"""
Unified Pipeline: End-to-End Encoding, Correlation Analysis, and Statistical Testing.

This script sequentially trains a DNN model to predict EEG responses,
computes the correlation and noise ceilings against biological test data,
and directly calculates the confidence intervals and statistical significance
in memory, avoiding redundant disk I/O operations.
"""

import os
import time
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
from data_utils import load_images, load_eeg_data, create_dataloader
from model import CustomModel
from nested_sgd import NestedSGD
from nested_adam import NestedAdam

# =============================================================================
# Configuration Class
# =============================================================================
class Args:
    def __init__(self):
        # Core modeling arguments
        self.sub = 1
        self.dnn = "adam+nested"
        self.epochs = 1
        self.patience = 30
        self.lr = 1e-5
        self.weight_decay = 0.0
        self.momentum = 0.9
        
        # Nested optimizer specific arguments
        self.alpha = 0.5
        self.beta = (0.9, 0.999, 0.9)  # Natively defined as a tuple
        self.freq = (1, 8, 16)
        self.chunk_size = (8, 8, 8)
        self.batch_size = 32
        
        # I/O arguments
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        self.project_dir = os.path.join(root_dir, "data")
        
        # Combined analysis arguments
        self.corr_n_iter = 1000

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
    cache_dir = os.path.join(args.project_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    img_cache_path = os.path.join(cache_dir, "image_data_cache.pt")

    if os.path.exists(img_cache_path):
        print(f"Loading cached image data from {img_cache_path}...")
        img_cache_data = torch.load(img_cache_path, weights_only=False)
        X_train = img_cache_data["X_train"]
        X_val = img_cache_data["X_val"]
        X_test = img_cache_data["X_test"]
    else:
        X_train, X_val, X_test = load_images(args, idx_val)
        print(f"Saving image data to {img_cache_path}...")
        torch.save({
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test
        }, img_cache_path)

    print("\n", "=" * 10, "Load EEG Data", "=" * 10)
    cache_dir = os.path.join(args.project_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"sub-{args.sub:02d}_eeg_data_avg.pt")

    if os.path.exists(cache_path):
        print(f"Loading cached averaged EEG data from {cache_path}...")
        cache_data = torch.load(cache_path, weights_only=False)
        y_train = cache_data["y_train"]
        y_val = cache_data["y_val"]
        y_test = cache_data["y_test"]
        ch_names = cache_data["ch_names"]
        times = cache_data["times"]
    else:
        y_train, y_val, y_test, ch_names, times = load_eeg_data(args, idx_val)
        print(f"Saving averaged EEG data to {cache_path}...")
        torch.save({
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "ch_names": ch_names,
            "times": times
        }, cache_path)

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

    model = CustomModel(num_channels=eeg_channels, time_points=eeg_time_points, hidden_dim=512, num_layers=1)
    # print(model)
    model.to(device)

    param_fast = [
        {"params": model.features.parameters(), "lr": args.lr * 0.1},
        {"params": model.feature_projection[1].parameters(), "lr": args.lr},
        {"params": model.lstm_layers.parameters(), "lr": args.lr},
        {"params": model.channel_decoder.parameters(), "lr": args.lr},
    ]
    param_mid = [
        {"params": model.feature_projection[4].parameters(), "lr": args.lr},
    ]
    param_slow = [
        {"params": model.feature_projection[6].parameters(), "lr": args.lr},
    ]

    params_list = [param_fast, param_mid, param_slow]

    if args.dnn == "gradient":
        opts = [torch.optim.SGD(param_fast, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)]
    elif args.dnn == "gradient+nested":
        opts = [NestedSGD(p, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, alpha=args.alpha, chunk_size=args.chunk_size[i]) for i, p in enumerate(params_list)]
    elif args.dnn == "adam":
        opts = [torch.optim.AdamW(param_fast, lr=args.lr, weight_decay=args.weight_decay)]
    elif args.dnn == "adam+nested":
        opts = [NestedAdam(p, lr=args.lr, weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta, chunk_size=args.chunk_size[i], freq=1) for i, p in enumerate(params_list)]

    loss_fn = torch.nn.MSELoss().to(device)
    torch.backends.cudnn.benchmark = True

    # =============================================================================
    # 4. Training Loop
    # =============================================================================
    print("\n", "=" * 10, "Training Model", "=" * 10)
    start_time = time.time()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model = None

    # Progress bar for Epochs within this specific combination
    pbar = tqdm(range(args.epochs), unit="epoch", leave=False)
    X_buffer, y_buffer = [None] * args.freq[-1], [None] * args.freq[-1]

    for epoch in pbar:
        # --- Training ---
        model.train()
        global_step_offset = epoch * len(train_dl)
        for batch_idx, (X, y) in enumerate(train_dl):
            current_step = global_step_offset + batch_idx + 1
            data_idx = current_step % args.freq[-1]
            X_buffer[data_idx], y_buffer[data_idx] = X.to(device), y.to(device)

            should_update = [current_step % f == 0 for f in args.freq]
            update_idx = None
            for i in range(len(should_update)-1, -1, -1):
                if should_update[i]:
                    update_idx = i 
                    break
            if update_idx != None:
                while update_idx != None:
                    next_update_idx = None
                    for i in range(update_idx-1, -1, -1):
                        if should_update[i]:
                            next_update_idx = i
                            break
                    end = args.freq[next_update_idx] if next_update_idx != None else 0
                    X = torch.cat([X_buffer[i%args.freq[-1]] for i in range(data_idx-args.freq[update_idx]+1, data_idx-end+1)], dim=0)
                    y = torch.cat([y_buffer[i%args.freq[-1]] for i in range(data_idx-args.freq[update_idx]+1, data_idx-end+1)], dim=0)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = model(X).squeeze()
                        loss = loss_fn(pred, y)
                    loss.backward()
                    for i in range(update_idx-1, -1, -1):
                        opts[i].zero_grad()
                    update_idx = next_update_idx

                for i in range(len(opts)):
                    if current_step % args.freq[i] == 0:
                        opts[i].step()

                for opt in opts:
                    opt.zero_grad()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
    print(f"Training Time: {time.time() - start_time:.2f} seconds")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # =============================================================================
    # 5. Model Inference & Correlation Analysis (In-Memory Processing)
    # =============================================================================
    print("\n", "=" * 10, "Model Inference & Correlation Analysis", "=" * 10)
    # Generate synthetic EEG test data
    best_model.to(device)
    best_model.eval()
    
    ptr = 0
    test_loss = 0.0
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(device), y.to(device)
            batch_size = X.size(0)
            
            # Compute test loss and predictions (avoid duplicate model call)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = best_model(X)
                t_loss = loss_fn(pred.squeeze(), y)
            test_loss += t_loss.item() * batch_size
            
            # Store predictions directly reshaped to avoid unnecessary flat arrays
            preds = pred.detach().float().cpu().numpy().reshape(batch_size, eeg_channels, eeg_time_points)
            synthetic_data[ptr:ptr+batch_size] = preds
            ptr += batch_size
            
    test_loss /= len(test_dl.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    synthetic_data_dict = {"all_time_points": synthetic_data}

    del best_model
    if device == "cuda":
        torch.cuda.empty_cache()
    # We already have the averaged test data loaded as `y_test` at the beginning
    bio_data_avg_all = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    
    num_channels = bio_data_avg_all.shape[1]
    num_times = bio_data_avg_all.shape[2]

    correlation = {}

    # 5.1 Model vs Full Data Correlation (Vectorized)
    print("Computing model correlations against full average data...")
    for layer, synth_data in synthetic_data_dict.items():
        # Vectorized Pearson correlation across samples (axis=0)
        synth_mean = np.mean(synth_data, axis=0, keepdims=True)
        bio_mean = np.mean(bio_data_avg_all, axis=0, keepdims=True)
        
        synth_centered = synth_data - synth_mean
        bio_centered = bio_data_avg_all - bio_mean
        
        cov = np.sum(synth_centered * bio_centered, axis=0)
        synth_var = np.sum(synth_centered ** 2, axis=0)
        bio_var = np.sum(bio_centered ** 2, axis=0)
        
        denom = np.sqrt(synth_var * bio_var)
        valid = denom > 0
        
        corr_matrix = np.zeros_like(denom)
        corr_matrix[valid] = cov[valid] / denom[valid]
        correlation[layer] = corr_matrix

    # 5.2 Split-Half Reliability for Noise Ceiling
    nc_cache_dir = os.path.join(args.project_dir, "cache")
    nc_cache_path = os.path.join(nc_cache_dir, f"sub-{args.sub:02d}_noise_ceiling_{args.corr_n_iter}.npy")

    if os.path.exists(nc_cache_path):
        print(f"Loading cached noise ceiling bounds from {nc_cache_path}...")
        nc_cache_data = np.load(nc_cache_path, allow_pickle=True).item()
        noise_ceiling_low = nc_cache_data["noise_ceiling_low"]
        noise_ceiling_up = nc_cache_data["noise_ceiling_up"]
    else:
        # Load raw biological EEG test data ONLY when cache doesn't exist
        print("Loading raw test data for noise ceiling estimation...")
        bio_data_dir = os.path.join("eeg_dataset", "preprocessed_data", f"sub-{args.sub:02d}", "preprocessed_eeg_test.npy")
        bio_data = np.load(os.path.join(args.project_dir, bio_data_dir), allow_pickle=True).item()
        bio_test = bio_data["preprocessed_eeg_data"]
        del bio_data

        print("Estimating split-half reliability for noise ceiling...")
        noise_ceiling_low_splits = np.zeros((args.corr_n_iter, num_channels, num_times))
        for i in tqdm(range(args.corr_n_iter)):
            shuffle_idx = resample(np.arange(0, bio_test.shape[1]), replace=False, n_samples=int(bio_test.shape[1] / 2))
            bio_data_avg_half_1 = np.mean(np.delete(bio_test, shuffle_idx, 1), 1)
            bio_data_avg_half_2 = np.mean(bio_test[:, shuffle_idx, :, :], 1)
    
            # Vectorized correlation across samples (axis=0)
            h1_mean = np.mean(bio_data_avg_half_1, axis=0, keepdims=True)
            h2_mean = np.mean(bio_data_avg_half_2, axis=0, keepdims=True)
            
            h1_centered = bio_data_avg_half_1 - h1_mean
            h2_centered = bio_data_avg_half_2 - h2_mean
            
            cov = np.sum(h1_centered * h2_centered, axis=0)
            h1_var = np.sum(h1_centered ** 2, axis=0)
            h2_var = np.sum(h2_centered ** 2, axis=0)
            
            denom = np.sqrt(h1_var * h2_var)
            valid = denom > 0
            
            corr_matrix = np.zeros_like(denom)
            corr_matrix[valid] = cov[valid] / denom[valid]
            noise_ceiling_low_splits[i] = corr_matrix
    
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
    p_values = {}
    significance = {}

    for layer in correlation.keys():
        time_points = correlation[layer].shape[1]
        p_values[layer] = np.ones(time_points)
        
        for t in range(time_points):
            # Fisher Z transform across channels before t-test
            # Clip correlation to prevent inf
            corr_clipped = np.clip(correlation[layer][:, t], -1 + 1e-7, 1 - 1e-7)
            
            fisher_values = np.arctanh(corr_clipped)
            
            p_values[layer][t] = ttest_1samp(fisher_values, 0, alternative="greater")[1]

        significance[layer] = multipletests(p_values[layer], 0.05, "bonferroni")[0]

    # For export backwards compatibility, we expand dims to simulate a (1, channels, times) subject array
    correlation_stat = {layer: np.expand_dims(correlation[layer], 0) for layer in correlation.keys()}

    # =============================================================================
    # 7. Final Output Export
    # =============================================================================
    stats_dict = {
        "correlation": correlation_stat,
        "significance": significance,
        "times": times,
        "test_loss": test_loss,
    }

    stats_save_dir = os.path.join("experiment", "tmp")
    os.makedirs(stats_save_dir, exist_ok=True)
    save_path = os.path.join(stats_save_dir, ".npy")
    np.save(save_path, stats_dict)
    print(f"\nPipeline Complete! Final stats saved to {save_path}")

if __name__ == "__main__":
    main()