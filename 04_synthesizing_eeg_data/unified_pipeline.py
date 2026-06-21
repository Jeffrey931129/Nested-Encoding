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
from custom_alexnet import CustomAlexNet
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
        self.lr = 0.001
        self.weight_decay = 0.0
        self.momentum = 0.9
        
        # Nested optimizer specific arguments
        self.alpha = 0.9
        self.beta = (0.9, 0.999)  # Natively defined as a tuple
        self.gamma = 0.1
        self.chunk_size = 10
        self.batch_size = 32
        
        # I/O arguments
        self.project_dir = "project_directory"
        
        # Combined analysis arguments
        self.corr_n_iter = 100
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
    out_features = y_test.shape[1] * y_test.shape[2]
    synthetic_data = np.zeros((y_test.shape))
    best_epochs = np.zeros((num_models))

    train_dl, val_dl, test_dl = create_dataloader(
        args, 0, g_cpu, X_train, X_val, X_test, y_train, y_val, y_test
    )

    model = CustomAlexNet(num_classes=out_features)
    # print(model)
    model.to(device)

    f_fast = 1
    f_mid = 4
    f_slow = 8

    param_fast = [
        {"params": model.features[0:4].parameters(), "lr": args.lr * 0.1},
        {"params": model.classifier[6].parameters()},
    ]
    param_mid = [
        {"params": model.features[4:9].parameters(), "lr": args.lr * 0.1},
        {"params": model.classifier[4].parameters()},
    ]
    param_slow = [
        {"params": model.features[9:13].parameters(), "lr": args.lr * 0.1},
        {"params": model.classifier[1].parameters()},
    ]

    if args.dnn == "gradient":
        opt_fast = torch.optim.SGD(param_fast, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        opt_mid, opt_slow = None, None
    elif args.dnn == "gradient+nested":
        opt_fast = NestedSGD(param_fast, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, alpha=args.alpha, chunk_size=args.chunk_size)
        opt_mid = NestedSGD(param_mid, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, alpha=args.alpha, chunk_size=args.chunk_size)
        opt_slow = NestedSGD(param_slow, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, alpha=args.alpha, chunk_size=args.chunk_size)
    elif args.dnn == "adam":
        opt_fast = torch.optim.Adam(param_fast, lr=args.lr, weight_decay=args.weight_decay)
        opt_mid, opt_slow = None, None
    elif args.dnn == "adam+nested":
        opt_fast = NestedAdam(param_fast, lr=args.lr, weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta, gamma=args.gamma, chunk_size=args.chunk_size)
        opt_mid = NestedAdam(param_mid, lr=args.lr, weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta, gamma=args.gamma, chunk_size=args.chunk_size)
        opt_slow = NestedAdam(param_slow, lr=args.lr, weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta, gamma=args.gamma, chunk_size=args.chunk_size)

    loss_fn = torch.nn.MSELoss().to(device)
    scaler = torch.amp.GradScaler("cuda")
    torch.backends.cudnn.benchmark = True

    # =============================================================================
    # 4. Training Loop
    # =============================================================================
    print("\n", "=" * 10, "Training Model", "=" * 10)
    best_val_loss = float("inf")
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

                if current_step % f_fast == 0:
                    scaler.step(opt_fast)
                    opt_fast.zero_grad()
                if opt_mid is not None and current_step % f_mid == 0:
                    scaler.step(opt_mid)
                    opt_mid.zero_grad()
                if opt_slow is not None and current_step % f_slow == 0:
                    scaler.step(opt_slow)
                    opt_slow.zero_grad()

                scaler.update()
            else:
                pred = model(X).squeeze()
                loss = loss_fn(pred, y)
                loss.backward()

                if current_step % f_fast == 0:
                    opt_fast.step()
                    opt_fast.zero_grad()
                if opt_mid is not None and current_step % f_mid == 0:
                    opt_mid.step()
                    opt_mid.zero_grad()
                if opt_slow is not None and current_step % f_slow == 0:
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
        
        if val_loss < best_val_loss:
            best_model = deepcopy(model)
            best_epochs[0] = epoch + 1
            best_val_loss = val_loss
            
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

    correlation = {layer: np.zeros((args.corr_n_iter, bio_test.shape[2], bio_test.shape[3])) for layer in synthetic_data_dict.keys()}
    noise_ceiling_low = np.zeros((args.corr_n_iter, bio_test.shape[2], bio_test.shape[3]))
    noise_ceiling_up = np.zeros((args.corr_n_iter, bio_test.shape[2], bio_test.shape[3]))

    bio_data_avg_all = np.mean(bio_test, 1)

    for i in tqdm(range(args.corr_n_iter)):
        shuffle_idx = resample(np.arange(0, bio_test.shape[1]), replace=False, n_samples=int(bio_test.shape[1] / 2))
        bio_data_avg_half_1 = np.mean(np.delete(bio_test, shuffle_idx, 1), 1)
        bio_data_avg_half_2 = np.mean(bio_test[:, shuffle_idx, :, :], 1)

        for t in range(bio_test.shape[3]):
            for c in range(bio_test.shape[2]):
                for layer in synthetic_data_dict.keys():
                    correlation[layer][i, c, t] = corr(synthetic_data_dict[layer][:, c, t], bio_data_avg_half_1[:, c, t])[0]
                noise_ceiling_low[i, c, t] = corr(bio_data_avg_half_2[:, c, t], bio_data_avg_half_1[:, c, t])[0]
                noise_ceiling_up[i, c, t] = corr(bio_data_avg_all[:, c, t], bio_data_avg_half_1[:, c, t])[0]

    # Average results across iterations
    for layer in synthetic_data_dict.keys():
        correlation[layer] = np.mean(correlation[layer], 0)
    noise_ceiling_low = np.mean(noise_ceiling_low, 0)
    noise_ceiling_up = np.mean(noise_ceiling_up, 0)

    # =============================================================================
    # 6. Statistical Significance & Bootstrapping (In-Memory Processing)
    # =============================================================================
    print("\n", "=" * 10, "Statistical Analysis", "=" * 10)
    # Expand dimensions to simulate multiple subjects array shape (1, channels, times)
    correlation_stat = {}
    diff_noise_ceiling = {}
    nc_low_stat = np.expand_dims(noise_ceiling_low, 0)
    nc_up_stat = np.expand_dims(noise_ceiling_up, 0)
    
    for layer in correlation.keys():
        correlation_stat[layer] = np.expand_dims(correlation[layer], 0)
        diff_noise_ceiling[layer] = nc_low_stat - correlation_stat[layer]

    ci_lower, ci_upper = {}, {}
    ci_lower_diff, ci_upper_diff = {}, {}

    for layer in correlation_stat.keys():
        time_points = correlation_stat[layer].shape[2]
        ci_lower[layer], ci_upper[layer] = np.zeros(time_points), np.zeros(time_points)
        ci_lower_diff[layer], ci_upper_diff[layer] = np.zeros(time_points), np.zeros(time_points)
        
        for t in tqdm(range(time_points)):
            sample_dist = np.zeros(args.stats_n_iter)
            sample_dist_diff = np.zeros(args.stats_n_iter)
            for i in range(args.stats_n_iter):
                # Resampling across the subject dimension (which is index 0)
                sample_dist[i] = np.mean(resample(np.mean(correlation_stat[layer][:, :, t], 1)))
                sample_dist_diff[i] = np.mean(resample(np.mean(diff_noise_ceiling[layer][:, :, t], 1)))
            
            ci_lower[layer][t] = np.percentile(sample_dist, 2.5)
            ci_upper[layer][t] = np.percentile(sample_dist, 97.5)
            ci_lower_diff[layer][t] = np.percentile(sample_dist_diff, 2.5)
            ci_upper_diff[layer][t] = np.percentile(sample_dist_diff, 97.5)

    p_values, p_values_diff = {}, {}
    significance, significance_diff = {}, {}

    for layer in correlation_stat.keys():
        time_points = correlation_stat[layer].shape[2]
        p_values[layer] = np.ones(time_points)
        p_values_diff[layer] = np.ones(time_points)
        
        for t in range(time_points):
            fisher_values = np.arctanh(np.mean(correlation_stat[layer][:, :, t], 1))
            fisher_values_diff = np.arctanh(np.mean(diff_noise_ceiling[layer][:, :, t], 1))
            
            if len(fisher_values) < 2:
                p_values[layer][t] = 1.0
                p_values_diff[layer][t] = 1.0
            else:
                p_values[layer][t] = ttest_1samp(fisher_values, 0, alternative="greater")[1]
                p_values_diff[layer][t] = ttest_1samp(fisher_values_diff, 0, alternative="greater")[1]

        significance[layer] = multipletests(p_values[layer], 0.05, "bonferroni")[0]
        significance_diff[layer] = multipletests(p_values_diff[layer], 0.05, "bonferroni")[0]

    # =============================================================================
    # 7. Final Output Export
    # =============================================================================
    stats_dict = {
        "correlation": correlation_stat,
        "ci_lower": ci_lower, "ci_upper": ci_upper, "significance": significance,
        "noise_ceiling_low": nc_low_stat, "noise_ceiling_up": nc_up_stat,
        "diff_noise_ceiling": diff_noise_ceiling,
        "ci_lower_diff_noise_ceiling": ci_lower_diff, "ci_upper_diff_noise_ceiling": ci_upper_diff,
        "significance_diff_noise_ceiling": significance_diff,
        "times": times, "ch_names": ch_names,
    }

    stats_save_dir = os.path.join(args.project_dir, "results", f"sub-{args.sub:02d}", "stats", f"dnn-{args.dnn}")
    os.makedirs(stats_save_dir, exist_ok=True)
    np.save(os.path.join(stats_save_dir, "correlation_stats.npy"), stats_dict)
    print(f"\nPipeline Complete! Final stats saved to {os.path.join(stats_save_dir, 'correlation_stats.npy')}")

if __name__ == "__main__":
    main()