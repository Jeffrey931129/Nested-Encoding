"""
Hyperparameter Tuning Script.

This script performs grid search across hyperparameter combinations to
identify the best-performing model configuration based on validation loss.
"""

import os
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from data_utils import (create_dataloader, data_dir, experiment_dir,
                        load_eeg_data, load_images)
from model import AlexEEGNet
from nested_adam import NestedAdam


# Configuration Class
class Args:
    def __init__(self):
        self.sub = 1
        self.model = "AlexEEGNet"
        self.optim = "NestedAdam"
        self.epochs = 200
        self.patience = 15

        self.hyperparameter_space = {
            "lr": [5e-06],
            "weight_decay": [5e-2],
            "batch_size": [32],
            "alpha": [0.5, 1.0, 5.0],
            "beta": [(0.95, 0.9, 0.999)],
            "freq": [(1, 2, 4), (1, 4, 8), (1, 4, 16), (1, 8, 64)],
            "chunk_size": [(1, 2, 4), (1, 4, 8), (8, 8, 8), (64, 64, 64)],
        }


def main():
    args = Args()
    print("=" * 10, f"Hyperparameter Tuning ({args.model} + {args.optim})", "=" * 10)
    # =============================================================================
    # 1. Setup Environment and Random Seeds
    # =============================================================================
    seed = 20200220
    g_cpu = torch.Generator()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =============================================================================
    # 2. Initialize Logging
    # =============================================================================
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = experiment_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{formatted_time}.log")
    detail_log_file = os.path.join(log_dir, f"{formatted_time}_detail.log")

    # =============================================================================
    # 3. Load the images (X) and the EEG data (y)
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
    # 4. Hyperparameter Search Loop
    # =============================================================================
    with open(log_file, "w") as f, open(detail_log_file, "w") as fd:
        f.write(f">> {args.model} + {args.optim} <<\n")
        f.write("-" * 50 + "\n")
        fd.write(f">> {args.model} + {args.optim} <<\n")
        fd.write("-" * 50 + "\n")

        eeg_channels = y_test.shape[1]
        eeg_time_points = y_test.shape[2]
        grid = list(ParameterGrid(args.hyperparameter_space))
        best_overall_loss = float("inf")
        best_overall_config = None

        for idx, config in enumerate(grid):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            g_cpu.manual_seed(seed)

            combo_str = f"[{idx+1}/{len(grid)}] Config: {config}"
            print(f"\n{combo_str}")
            f.write(f"\n{combo_str}\n")
            fd.write(f"\n{combo_str}\n")
            f.flush()
            fd.flush()

            model = args.model
            optim = args.optim
            epochs = args.epochs
            patience = args.patience
            lr = config.get("lr", 1e-5)
            weight_decay = config.get("weight_decay", 0.0)
            batch_size = config.get("batch_size", 32)
            alpha = config.get("alpha", 0.5)
            beta = config.get("beta", (0.9, 0.9, 0.999))
            freq = config.get("freq", (1, 8, 16))
            chunk_size = config.get("chunk_size", (8, 8, 8))

            train_dl, val_dl, test_dl = create_dataloader(batch_size, g_cpu, X_train, X_val, X_test, y_train, y_val, y_test)

            model = AlexEEGNet(num_channels=eeg_channels, time_points=eeg_time_points)
            model.to(device)

            param_fast = [
                {"params": model.features.parameters(), "lr": lr / freq[0] * 0.1},
                {"params": model.classifier[1].parameters(), "lr": lr / freq[0]},
                {"params": model.lstm.parameters(), "lr": lr / freq[0]},
                {"params": model.channel_decoder.parameters(), "lr": lr / freq[0]},
            ]
            param_mid = [
                {"params": model.classifier[4].parameters(), "lr": lr / freq[1]},
            ]
            param_slow = [
                {"params": model.classifier[6].parameters(), "lr": lr / freq[2]},
            ]

            params_list = [param_fast, param_mid, param_slow]

            if optim == "AdamW":
                opts = [torch.optim.AdamW(p, lr=lr, weight_decay=weight_decay, betas=beta) for p in params_list]
            elif optim == "NestedAdam":
                opts = [NestedAdam(p, lr=lr, weight_decay=weight_decay, alpha=alpha, beta=beta, chunk_size=chunk_size[i]) for i, p in enumerate(params_list)]

            loss_fn = nn.MSELoss(reduction="sum").to(device)
            torch.backends.cudnn.benchmark = True

            best_model = None
            best_val_loss = float("inf")
            epochs_no_improve = 0

            pbar = tqdm(range(epochs), desc=f"Combo {idx+1}", unit="epoch", leave=False)
            for epoch in pbar:
                # --- Training ---
                model.train()
                train_loss = 0.0

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

                    train_loss += loss.item() * X.size(0)
                train_loss /= len(train_dl.dataset)

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
                fd.write(f"    Epoch {epoch + 1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}\n")
                fd.flush()

                if val_loss < best_val_loss:
                    best_model = deepcopy(model)
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

            best_model.to(device)
            best_model.eval()
            with torch.no_grad():
                X, y = next(iter(test_dl))
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = best_model(X)
                    t_loss = loss_fn(pred.squeeze(), y)
                test_loss = t_loss.item() / (len(test_dl.dataset) * eeg_channels * eeg_time_points)

            del model
            del best_model
            if device == "cuda":
                torch.cuda.empty_cache()

            print(f"-> Best Val Loss: {best_val_loss:.6f} | Test Loss: {test_loss:.6f} (Stopped at Epoch {epoch+1})")
            f.write(f"-> Best Val Loss: {best_val_loss:.6f} | Test Loss: {test_loss:.6f} (Stopped at Epoch {epoch+1})\n")
            fd.write(f"-> Best Val Loss: {best_val_loss:.6f} | Test Loss: {test_loss:.6f} (Stopped at Epoch {epoch+1})\n")

            if test_loss < best_overall_loss:
                best_overall_loss = test_loss
                best_overall_config = config
                f.write(">>> New Best Found! <<<\n")
                fd.write(">>> New Best Found! <<<\n")

            f.write("\n")
            fd.write("\n")
            f.flush()
            fd.flush()

        summary_str = "=" * 50 + "\n"
        summary_str += "Optimization Completed\n"
        summary_str += f"Best Test Loss: {best_overall_loss:.6f}\n"
        summary_str += "Optimal Hyperparameters:\n"
        for k, v in best_overall_config.items():
            summary_str += f"  - {k}: {v}\n"
        summary_str += "=" * 50 + "\n"

        print(summary_str)
        f.write(summary_str)
        fd.write(summary_str)


if __name__ == "__main__":
    main()
