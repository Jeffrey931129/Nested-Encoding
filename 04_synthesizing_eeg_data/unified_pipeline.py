"""
Unified Pipeline: End-to-End Encoding, Correlation Analysis, and Statistical Testing.

This script sequentially trains a DNN model to predict EEG responses,
computes the correlation and noise ceilings against biological test data,
and directly calculates the confidence intervals and statistical significance
in memory, avoiding redundant disk I/O operations.
"""

import os
import numpy as np
import random
import torch
from tqdm import tqdm
from sklearn.utils import resample
from copy import deepcopy
from scipy.stats import pearsonr as corr
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

# Local utility imports
from end_to_end_encoding_utils import load_images, load_eeg_data, create_dataloader
from custom_model import CustomModel
from nested_sgd import NestedSGD
from nested_adam import NestedAdam

# =============================================================================
# Configuration Class
# =============================================================================
class Args:
    def __init__(self):
        # Core modeling arguments
        self.sub = 1
        self.modeled_time_points = "all"
        self.dnn = "adam+nested"
        self.pretrained = True
        self.epochs = 200
        self.patience = 30
        self.lr = 0.00001
        self.weight_decay = 0.0
        self.momentum = 0.9
        
        # Nested optimizer specific arguments
        self.alpha = 0.1
        self.beta = (0.9, 0.999)  # Natively defined as a tuple
        self.freq = (1, 2, 4)
        self.chunk_size = (1, 2, 4)
        self.batch_size = 32
        
        # I/O arguments
        self.project_dir = "project_directory"
        
        # Combined analysis arguments
        self.corr_n_iter = 1000
        self.stats_n_iter = 10000

def main():
    # Initialize the configuration object directly
    args = Args()

    print(">>> Unified End-to-End Encoding, Correlation & Stats Pipeline <<<")
    print("\nInput parameters:")
    for key, val in vars(args).items():
        print("{:20} {}".format(key, val))

    # =============================================================================
    # 1. Setup Environment & Random Seeds
    # =============================================================================
    seed = 20200220
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =============================================================================
    # 2. Load the images (X) and the EEG data (y)
    # =============================================================================
    train_img_concepts = np.arange(1654)
    img_per_concept = 10
    val_concepts = np.sort(resample(train_img_concepts, replace=False, n_samples=100))
    idx_val = np.zeros((len(train_img_concepts) * img_per_concept), dtype=bool)
    for i in val_concepts:
        idx_val[i * img_per_concept : i * img_per_concept + img_per_concept] = True

    print("\n", "=" * 10, "Load Image", "=" * 10)
    X_train, X_val, X_test = load_images(args, idx_val)

    print("\n", "=" * 10, "Load EEG Data", "=" * 10)
    y_train, y_val, y_test, ch_names, times = load_eeg_data(args, idx_val)

    # =============================================================================
    # 3. Model Initialization
    # =============================================================================
    num_models = 1
    eeg_channels = y_test.shape[1]
    eeg_time_points = y_test.shape[2]
    # print(f"EEG Channel: {eeg_channels}, EEG Time Point: {eeg_time_points}")
    out_features = y_test.shape[1] * y_test.shape[2]
    synthetic_data = np.zeros((y_test.shape))
    best_epochs = np.zeros((num_models))

    train_dl, val_dl, test_dl = create_dataloader(
        args, 0, g_cpu, X_train, X_val, X_test, y_train, y_val, y_test
    )

    model = CustomModel(num_channels=eeg_channels, time_points=eeg_time_points, hidden_dim=512, num_layers=3)
    # print(model)
    model.to(device)

    f_fast, f_mid, f_slow = args.freq
    c_fast, c_mid, c_slow = args.chunk_size

    param_fast = [
        {"params": model.features[0:4].parameters(), "lr": args.lr * 0.1},
        {"params": model.feature_projection.parameters(), "lr": args.lr},
        {"params": model.lstm_layers[2].parameters(), "lr": args.lr},
        {"params": model.channel_decoder.parameters(), "lr": args.lr},
    ]
    param_mid = [
        {"params": model.features[4:9].parameters(), "lr": args.lr * 0.1},
        {"params": model.lstm_layers[1].parameters(), "lr": args.lr},
    ]
    param_slow = [
        {"params": model.features[9:13].parameters(), "lr": args.lr * 0.1},
        {"params": model.lstm_layers[0].parameters(), "lr": args.lr},
    ]

    if args.dnn == "gradient":
        opt_fast = torch.optim.SGD(param_fast, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        opt_mid, opt_slow = None, None
    elif args.dnn == "gradient+nested":
        opt_fast = NestedSGD(param_fast, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, alpha=args.alpha, chunk_size=c_fast)
        opt_mid = NestedSGD(param_mid, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, alpha=args.alpha, chunk_size=c_mid)
        opt_slow = NestedSGD(param_slow, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, alpha=args.alpha, chunk_size=c_slow)
    elif args.dnn == "adam":
        opt_fast = torch.optim.Adam(param_fast, lr=args.lr, weight_decay=args.weight_decay)
        opt_mid, opt_slow = None, None
    elif args.dnn == "adam+nested":
        opt_fast = NestedAdam(param_fast, lr=args.lr, weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta, chunk_size=c_fast, freq=f_fast)
        opt_mid = NestedAdam(param_mid, lr=args.lr, weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta, chunk_size=c_mid, freq=f_mid)
        opt_slow = NestedAdam(param_slow, lr=args.lr, weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta, chunk_size=c_slow, freq=f_slow)

    loss_fn = torch.nn.MSELoss().to(device)
    scaler = torch.amp.GradScaler("cuda")
    torch.backends.cudnn.benchmark = True

    # =============================================================================
    # 4. Training Loop
    # =============================================================================
    print("\n", "=" * 10, "Training Model", "=" * 10)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model = None

    # Progress bar for Epochs within this specific combination
    pbar = tqdm(range(args.epochs), unit="epoch", leave=False)

    for epoch in pbar:
        # --- Training ---
        model.train()
        train_loss = 0.0
        global_step_offset = epoch * len(train_dl)
        for batch_idx, (X, y) in enumerate(train_dl):
            current_step = global_step_offset + batch_idx + 1
            X, y = X.to(device), y.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(X).squeeze()
                    loss = loss_fn(pred, y)
                scaler.scale(loss).backward()

                scaler.step(opt_fast)
                opt_fast.zero_grad()
                if opt_mid is not None:
                    scaler.step(opt_mid)
                    opt_mid.zero_grad()
                if opt_slow is not None:
                    scaler.step(opt_slow)
                    opt_slow.zero_grad()

                scaler.update()
            else:
                pred = model(X).squeeze()
                loss = loss_fn(pred, y)
                loss.backward()

                opt_fast.step()
                opt_fast.zero_grad()
                if opt_mid is not None:
                    opt_mid.step()
                    opt_mid.zero_grad()
                if opt_slow is not None:
                    opt_slow.step()
                    opt_slow.zero_grad()

            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_dl.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                pred = model(X).squeeze()
                v_loss = loss_fn(pred, y)
                val_loss += v_loss.item() * X.size(0)
        val_loss /= len(val_dl.dataset)

        # Update progress bar info
        pbar.set_postfix(
            {"Val_Loss": f"{val_loss:.4f}", "Best": f"{best_val_loss:.4f}"}
        )
        
        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_model = deepcopy(model)
            best_epochs[0] = epoch + 1
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
            
    print(f"Best Epochs: {best_epochs[0]}, Best Loss: {best_val_loss:.4f}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Generate synthetic EEG test data
    best_model.to(device)
    best_model.eval()
    
    # Initialize a flat array to accumulate batch predictions
    bio_test_shape = (len(test_dl.dataset), out_features)
    synthetic_data_flat = np.zeros(bio_test_shape)
    
    ptr = 0
    with torch.no_grad():
        for X, y in test_dl:
            X = X.to(device)
            batch_size = X.size(0)
            preds = best_model(X).detach().cpu().numpy()
            synthetic_data_flat[ptr:ptr+batch_size] = preds
            ptr += batch_size

    # Reshape back to the original y_test shape
    synthetic_data = np.reshape(synthetic_data_flat, synthetic_data.shape)
    synthetic_data_dict = {"all_time_points": synthetic_data}

    del best_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # =============================================================================
    # 5. Correlation Analysis (In-Memory Processing)
    # =============================================================================
    print("\n", "=" * 10, "Correlation Analysis", "=" * 10)
    # Load biological EEG test data
    bio_data_dir = os.path.join("eeg_dataset", "preprocessed_data", f"sub-{args.sub:02d}", "preprocessed_eeg_test.npy")
    bio_data = np.load(os.path.join(args.project_dir, bio_data_dir), allow_pickle=True).item()
    bio_test = bio_data["preprocessed_eeg_data"]
    del bio_data

    # We evaluate model against the FULL average data to avoid data leakage
    bio_data_avg_all = np.mean(bio_test, 1)

    correlation = {layer: np.zeros((bio_test.shape[2], bio_test.shape[3])) for layer in synthetic_data_dict.keys()}

    # 5.1 Model vs Full Data Correlation
    print("Computing model correlations against full average data...")
    for t in tqdm(range(bio_test.shape[3])):
        for c in range(bio_test.shape[2]):
            for layer in synthetic_data_dict.keys():
                correlation[layer][c, t] = corr(synthetic_data_dict[layer][:, c, t], bio_data_avg_all[:, c, t])[0]

    # 5.2 Split-Half Reliability for Noise Ceiling
    nc_cache_dir = os.path.join(args.project_dir, "results", f"sub-{args.sub:02d}", "correlation_bound")
    nc_cache_path = os.path.join(nc_cache_dir, f"{args.corr_n_iter}.npy")

    if os.path.exists(nc_cache_path):
        print(f"Loading cached noise ceiling bounds from {nc_cache_path}...")
        nc_cache_data = np.load(nc_cache_path, allow_pickle=True).item()
        noise_ceiling_low = nc_cache_data["noise_ceiling_low"]
        noise_ceiling_up = nc_cache_data["noise_ceiling_up"]
    else:
        print("Estimating split-half reliability for noise ceiling...")
        noise_ceiling_low_splits = np.zeros((args.corr_n_iter, bio_test.shape[2], bio_test.shape[3]))
        for i in tqdm(range(args.corr_n_iter)):
            shuffle_idx = resample(np.arange(0, bio_test.shape[1]), replace=False, n_samples=int(bio_test.shape[1] / 2))
            bio_data_avg_half_1 = np.mean(np.delete(bio_test, shuffle_idx, 1), 1)
            bio_data_avg_half_2 = np.mean(bio_test[:, shuffle_idx, :, :], 1)
    
            for t in range(bio_test.shape[3]):
                for c in range(bio_test.shape[2]):
                    noise_ceiling_low_splits[i, c, t] = corr(bio_data_avg_half_2[:, c, t], bio_data_avg_half_1[:, c, t])[0]
    
        # Helper function for Fisher Z-transform averaging
        def fisher_z_mean(corrs):
            corrs = np.clip(corrs, -1 + 1e-7, 1 - 1e-7)
            return np.tanh(np.mean(np.arctanh(corrs), axis=0))
    
        # Average split-half reliability across iterations using Fisher Z
        noise_ceiling_low = fisher_z_mean(noise_ceiling_low_splits)
    
        # Apply Spearman-Brown formula to estimate full data reliability
        noise_ceiling_low = 2 * noise_ceiling_low / (1 + noise_ceiling_low)
    
        # Theoretical upper bound for full data is the square root of its reliability
        noise_ceiling_up = np.sqrt(np.clip(noise_ceiling_low, 0, None))
        
        os.makedirs(nc_cache_dir, exist_ok=True)
        np.save(nc_cache_path, {
            "noise_ceiling_low": noise_ceiling_low,
            "noise_ceiling_up": noise_ceiling_up
        })
        print(f"Saved noise ceiling bounds to {nc_cache_path}")

    # =============================================================================
    # 6. Statistical Significance & Bootstrapping (In-Memory Processing)
    # =============================================================================
    print("\n", "=" * 10, "Statistical Analysis", "=" * 10)
    # Compute differences (shape: channels, times)
    diff_noise_ceiling = {}
    for layer in correlation.keys():
        diff_noise_ceiling[layer] = noise_ceiling_low - correlation[layer]

    ci_lower, ci_upper = {}, {}
    ci_lower_diff, ci_upper_diff = {}, {}

    # Option 1: Across-channel within-subject statistics
    for layer in correlation.keys():
        time_points = correlation[layer].shape[1]
        ci_lower[layer], ci_upper[layer] = np.zeros(time_points), np.zeros(time_points)
        ci_lower_diff[layer], ci_upper_diff[layer] = np.zeros(time_points), np.zeros(time_points)
        
        for t in tqdm(range(time_points)):
            sample_dist = np.zeros(args.stats_n_iter)
            sample_dist_diff = np.zeros(args.stats_n_iter)
            for i in range(args.stats_n_iter):
                # Resampling across the channel dimension
                sample_dist[i] = np.mean(resample(correlation[layer][:, t]))
                sample_dist_diff[i] = np.mean(resample(diff_noise_ceiling[layer][:, t]))
            
            ci_lower[layer][t] = np.percentile(sample_dist, 2.5)
            ci_upper[layer][t] = np.percentile(sample_dist, 97.5)
            ci_lower_diff[layer][t] = np.percentile(sample_dist_diff, 2.5)
            ci_upper_diff[layer][t] = np.percentile(sample_dist_diff, 97.5)

    p_values, p_values_diff = {}, {}
    significance, significance_diff = {}, {}

    for layer in correlation.keys():
        time_points = correlation[layer].shape[1]
        p_values[layer] = np.ones(time_points)
        p_values_diff[layer] = np.ones(time_points)
        
        for t in range(time_points):
            # Fisher Z transform across channels before t-test
            # Clip correlation to prevent inf
            corr_clipped = np.clip(correlation[layer][:, t], -1 + 1e-7, 1 - 1e-7)
            diff_clipped = np.clip(diff_noise_ceiling[layer][:, t], -1 + 1e-7, 1 - 1e-7)
            
            fisher_values = np.arctanh(corr_clipped)
            fisher_values_diff = np.arctanh(diff_clipped)
            
            p_values[layer][t] = ttest_1samp(fisher_values, 0, alternative="greater")[1]
            p_values_diff[layer][t] = ttest_1samp(fisher_values_diff, 0, alternative="greater")[1]

        significance[layer] = multipletests(p_values[layer], 0.05, "bonferroni")[0]
        significance_diff[layer] = multipletests(p_values_diff[layer], 0.05, "bonferroni")[0]

    # For export backwards compatibility, we expand dims to simulate a (1, channels, times) subject array
    correlation_stat = {layer: np.expand_dims(correlation[layer], 0) for layer in correlation.keys()}
    diff_noise_ceiling_stat = {layer: np.expand_dims(diff_noise_ceiling[layer], 0) for layer in diff_noise_ceiling.keys()}
    nc_low_stat = np.expand_dims(noise_ceiling_low, 0)
    nc_up_stat = np.expand_dims(noise_ceiling_up, 0)

    # =============================================================================
    # 7. Final Output Export
    # =============================================================================
    stats_dict = {
        "correlation": correlation_stat,
        "ci_lower": ci_lower, "ci_upper": ci_upper, "significance": significance,
        "noise_ceiling_low": nc_low_stat, "noise_ceiling_up": nc_up_stat,
        "diff_noise_ceiling": diff_noise_ceiling_stat,
        "ci_lower_diff_noise_ceiling": ci_lower_diff, "ci_upper_diff_noise_ceiling": ci_upper_diff,
        "significance_diff_noise_ceiling": significance_diff,
        "times": times, "ch_names": ch_names,
    }

    stats_save_dir = os.path.join("experiment", "tmp")
    os.makedirs(stats_save_dir, exist_ok=True)
    save_path = os.path.join(stats_save_dir, ".npy")
    np.save(save_path, stats_dict)
    print(f"\nPipeline Complete! Final stats saved to {save_path}")

if __name__ == "__main__":
    main()