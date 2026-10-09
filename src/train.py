"""
Training Script.

This script trains the visual-to-EEG encoding model and evaluates neural
prediction accuracy against biological noise ceilings.
"""

import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
from scipy.stats import ttest_1samp
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from data_utils import (create_dataloader, data_dir, experiment_dir,
                        load_eeg_data, load_images)
from models import AlexEEGNet, AlexNetLite
from nested_adam import NestedAdam


# Configuration Class
class Args:
    def __init__(self):
        # Core modeling arguments
        self.sub = 1
        self.model = "AlexNetLite"
        self.optim = "NestedAdam"
        self.epochs = 200
        self.patience = 30
        self.lr = 5e-6
        self.weight_decay = 0.05
        self.batch_size = 32

        # Nested optimizer specific arguments
        self.alpha = 0.0
        self.beta = (0.95, 0.9, 0.999)
        self.freq = (1, 1, 1)
        self.macro_period = (1, 1, 1)

        # Combined analysis arguments
        self.corr_n_iter = 1000


def main():
    args = Args()
    print("Input parameters:")
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
    print("\n", "=" * 10, "Load Image", "=" * 10)
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    img_cache_path = os.path.join(cache_dir, "image_data_cache.pt")

    if os.path.exists(img_cache_path):
        print(f"Loading cached image data from {img_cache_path}...")
        img_cache_data = torch.load(img_cache_path, weights_only=False)
        X_train = img_cache_data["X_train"]
        X_val = img_cache_data["X_val"]
        X_test = img_cache_data["X_test"]
    else:
        X_train, X_val, X_test = load_images()
        print(f"Saving image data to {img_cache_path}...")
        torch.save({"X_train": X_train, "X_val": X_val, "X_test": X_test}, img_cache_path)

    print("\n", "=" * 10, "Load EEG Data", "=" * 10)
    cache_dir = os.path.join(data_dir, "cache")
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
        y_train, y_val, y_test, ch_names, times = load_eeg_data(args.sub)
        print(f"Saving averaged EEG data to {cache_path}...")
        torch.save({"y_train": y_train, "y_val": y_val, "y_test": y_test, "ch_names": ch_names, "times": times}, cache_path)

    # =============================================================================
    # 3. Model Initialization
    # =============================================================================
    eeg_channels = y_test.shape[1]
    eeg_time_points = y_test.shape[2]
    model = args.model
    optim = args.optim
    epochs = args.epochs
    patience = args.patience
    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    alpha = args.alpha
    beta = args.beta
    freq = args.freq
    macro_period = args.macro_period

    train_dl, val_dl, test_dl = create_dataloader(batch_size, g_cpu, X_train, X_val, X_test, y_train, y_val, y_test)

    if model == "AlexNetLite":
        model = AlexNetLite(num_channels=eeg_channels, time_points=eeg_time_points)
        param = [
            {"params": model.features.parameters(), "lr": lr * 0.1},
            {"params": model.classifier.parameters(), "lr": lr},
        ]
        params_list = [param]
    elif model == "AlexEEGNet":
        model = AlexEEGNet(num_channels=eeg_channels, time_points=eeg_time_points)
        param_fast = [
            {"params": model.features.parameters(), "lr": lr * 0.1},
            {"params": model.classifier[1].parameters(), "lr": lr},
            {"params": model.lstm.parameters(), "lr": lr},
            {"params": model.channel_decoder.parameters(), "lr": lr},
        ]
        param_mid = [
            {"params": model.classifier[4].parameters(), "lr": lr},
        ]
        param_slow = [
            {"params": model.classifier[6].parameters(), "lr": lr},
        ]
        params_list = [param_fast, param_mid, param_slow]
    model.to(device)

    if optim == "Adam":
        opts = [torch.optim.AdamW(p, lr=lr, weight_decay=weight_decay, betas=(beta[0], beta[2])) for p in params_list]
    elif optim == "NestedAdam":
        opts = [NestedAdam(p, lr=lr, weight_decay=weight_decay, alpha=alpha, beta=beta, macro_period=macro_period[i]) for i, p in enumerate(params_list)]

    loss_fn = torch.nn.MSELoss(reduction="sum").to(device)
    torch.backends.cudnn.benchmark = True

    # =============================================================================
    # 4. Training Loop
    # =============================================================================
    print("\n", "=" * 10, "Training Model", "=" * 10)
    start_time = time.time()
    best_model = None
    best_val_loss = float("inf")
    epochs_no_improve = 0
    freq = freq if optim == "NestedAdam" else (1,) * len(freq)

    pbar = tqdm(range(epochs), unit="epoch", leave=False)
    for epoch in pbar:
        # --- Training ---
        model.train()
        
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
        
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(X)
                loss = loss_fn(pred.squeeze(), y)
            loss = loss / X.size(0) / eeg_channels / eeg_time_points
            loss.backward()
        
            for opt in opts:
                opt.step()
                opt.zero_grad()
        
        # --- Validation ---
        model.eval()
        with torch.no_grad():
            X, y = next(iter(val_dl))
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(X)
                v_loss = loss_fn(pred.squeeze(), y)
            val_loss = v_loss.item() / (len(val_dl.dataset) * eeg_channels * eeg_time_points)

        pbar.set_postfix({"Val_Loss": f"{val_loss:.6f}", "Best": f"{best_val_loss:.6f}"})

        if val_loss < best_val_loss:
            best_model = deepcopy(model)
            best_epochs = epoch + 1
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"Best Epochs: {best_epochs}, Best Loss: {best_val_loss:.6f}")
    print(f"Training Time: {time.time() - start_time:.2f} seconds")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # =============================================================================
    # 5. Model Inference & Correlation Analysis (In-Memory Processing)
    # =============================================================================
    print("\n", "=" * 10, "Model Inference & Correlation Analysis", "=" * 10)
    best_model.to(device)
    best_model.eval()

    with torch.no_grad():
        X, y = next(iter(test_dl))
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = best_model(X)
            t_loss = loss_fn(pred.squeeze(), y)
        test_loss = t_loss.item() / (len(test_dl.dataset) * eeg_channels * eeg_time_points)

        synthetic_data = pred.detach().float().cpu().numpy().reshape(len(test_dl.dataset), eeg_channels, eeg_time_points)
    print(f"Test Loss: {test_loss:.6f}")

    synthetic_data_dict = {"all_time_points": synthetic_data}

    del best_model
    if device == "cuda":
        torch.cuda.empty_cache()

    bio_data_avg_all = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    num_channels = bio_data_avg_all.shape[1]
    num_times = bio_data_avg_all.shape[2]

    correlation = {}

    # 5.1 Model vs Full Data Correlation (Vectorized)
    print("Computing model correlations against full average data...")
    for layer, synth_data in synthetic_data_dict.items():
        synth_mean = np.mean(synth_data, axis=0, keepdims=True)
        bio_mean = np.mean(bio_data_avg_all, axis=0, keepdims=True)

        synth_centered = synth_data - synth_mean
        bio_centered = bio_data_avg_all - bio_mean

        cov = np.sum(synth_centered * bio_centered, axis=0)
        synth_var = np.sum(synth_centered**2, axis=0)
        bio_var = np.sum(bio_centered**2, axis=0)

        denom = np.sqrt(synth_var * bio_var)
        valid = denom > 0

        corr_matrix = np.zeros_like(denom)
        corr_matrix[valid] = cov[valid] / denom[valid]
        correlation[layer] = corr_matrix

    # 5.2 Split-Half Reliability for Noise Ceiling
    nc_cache_dir = os.path.join(data_dir, "cache")
    nc_cache_path = os.path.join(nc_cache_dir, f"sub-{args.sub:02d}_noise_ceiling_{args.corr_n_iter}.npy")

    if os.path.exists(nc_cache_path):
        print(f"Loading cached noise ceiling bounds from {nc_cache_path}...")
        nc_cache_data = np.load(nc_cache_path, allow_pickle=True).item()
        noise_ceiling_low = nc_cache_data["noise_ceiling_low"]
        noise_ceiling_up = nc_cache_data["noise_ceiling_up"]
    else:
        print("Loading raw test data for noise ceiling estimation...")
        bio_data_dir = os.path.join("eeg_dataset", "preprocessed_data", f"sub-{args.sub:02d}", "preprocessed_eeg_test.npy")
        bio_data = np.load(os.path.join(data_dir, bio_data_dir), allow_pickle=True).item()
        bio_test = bio_data["preprocessed_eeg_data"]
        del bio_data

        print("Estimating split-half reliability for noise ceiling...")
        noise_ceiling_low_splits = np.zeros((args.corr_n_iter, num_channels, num_times))
        for i in tqdm(range(args.corr_n_iter)):
            shuffle_idx = resample(np.arange(0, bio_test.shape[1]), replace=False, n_samples=int(bio_test.shape[1] / 2))
            bio_data_avg_half_1 = np.mean(np.delete(bio_test, shuffle_idx, 1), 1)
            bio_data_avg_half_2 = np.mean(bio_test[:, shuffle_idx, :, :], 1)

            h1_mean = np.mean(bio_data_avg_half_1, axis=0, keepdims=True)
            h2_mean = np.mean(bio_data_avg_half_2, axis=0, keepdims=True)

            h1_centered = bio_data_avg_half_1 - h1_mean
            h2_centered = bio_data_avg_half_2 - h2_mean

            cov = np.sum(h1_centered * h2_centered, axis=0)
            h1_var = np.sum(h1_centered**2, axis=0)
            h2_var = np.sum(h2_centered**2, axis=0)

            denom = np.sqrt(h1_var * h2_var)
            valid = denom > 0

            corr_matrix = np.zeros_like(denom)
            corr_matrix[valid] = cov[valid] / denom[valid]
            noise_ceiling_low_splits[i] = corr_matrix

        def fisher_z_mean(corrs):
            corrs = np.clip(corrs, -1 + 1e-7, 1 - 1e-7)
            return np.tanh(np.mean(np.arctanh(corrs), axis=0))

        noise_ceiling_low = fisher_z_mean(noise_ceiling_low_splits)
        noise_ceiling_low = 2 * noise_ceiling_low / (1 + noise_ceiling_low)
        noise_ceiling_up = np.sqrt(np.clip(noise_ceiling_low, 0, None))

        os.makedirs(nc_cache_dir, exist_ok=True)
        np.save(nc_cache_path, {"noise_ceiling_low": noise_ceiling_low, "noise_ceiling_up": noise_ceiling_up})
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
            corr_clipped = np.clip(correlation[layer][:, t], -1 + 1e-7, 1 - 1e-7)
            fisher_values = np.arctanh(corr_clipped)
            p_values[layer][t] = ttest_1samp(fisher_values, 0, alternative="greater")[1]
        significance[layer] = multipletests(p_values[layer], 0.05, "bonferroni")[0]
    correlation_stat = {layer: np.expand_dims(correlation[layer], 0) for layer in correlation.keys()}

    # =============================================================================
    # 7. Final Output Export
    # =============================================================================
    stats_dict = {
        "args": vars(args),
        "correlation": correlation_stat,
        "significance": significance,
        "times": times,
        "test_loss": test_loss,
        "predictions": synthetic_data_dict["all_time_points"],
    }

    stats_save_dir = experiment_dir
    os.makedirs(stats_save_dir, exist_ok=True)
    save_path = os.path.join(stats_save_dir, ".npy")
    np.save(save_path, stats_dict)
    print(f"\nComplete! Final stats saved to {save_path}")


if __name__ == "__main__":
    main()
