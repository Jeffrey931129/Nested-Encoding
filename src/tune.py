"""
Hyperparameter Tuning Script.

This script performs a grid search over hyperparameter combinations using 
a train/validation split for the model (e.g., AlexEEGNet) with the specified optimizer (e.g., NestedAdam, AdamW).
"""

import os
import numpy as np
import random
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.utils import resample

# Local utility imports
from data_utils import load_images, load_eeg_data, create_dataloader, data_dir, experiment_dir
from model import AlexEEGNet
from nested_adam import NestedAdam

# =============================================================================
# Configuration Class
# =============================================================================
class Args:
    def __init__(self):
        # Core defaults
        self.sub = 1
        self.model = "AlexEEGNet"
        self.optim = "NestedAdam"
        self.epochs = 200
        self.patience = 15

        self.hyperparameter_space = {
            "lr": [5e-6, 1e-5, 5e-5],
            "batch_size": [16, 32],
            "alpha": [0.0],
            "beta": [(0.85, 0.999, 0.9), (0.9, 0.999, 0.9), (0.95, 0.999, 0.9)],
            "chunk_size": [(1, 1, 1)],
            "freq": [(1, 1, 1)],
            "weight_decay": [0.0, 1e-4, 1e-2],
        }

def main():
    args = Args()

    print(f">>> Hyperparameter Tuning ({args.optim} + {args.model}) <<<")
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
    # 2. Initialize Log File
    # =============================================================================
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    
    log_dir = os.path.join(experiment_dir, "tmp")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{formatted_time}.log")
    detail_log_file = os.path.join(log_dir, f"{formatted_time}_detail.log")
    print(f"\nTuning started. Results will be saved to:\n  - {log_file}\n  - {detail_log_file}")

    # =============================================================================
    # 3. Load the images (X) and the EEG data (y)
    # =============================================================================
    train_img_concepts = np.arange(1654)
    img_per_concept = 10
    val_concepts = np.sort(resample(train_img_concepts, replace=False, n_samples=100))
    idx_val = np.zeros((len(train_img_concepts) * img_per_concept), dtype=bool)
    for i in val_concepts:
        idx_val[i * img_per_concept : i * img_per_concept + img_per_concept] = True

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
        X_train, X_val, X_test = load_images(args, idx_val)
        print(f"Saving image data to {img_cache_path}...")
        torch.save({
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test
        }, img_cache_path)

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
        y_train, y_val, y_test, ch_names, times = load_eeg_data(args, idx_val)
        print(f"Saving averaged EEG data to {cache_path}...")
        torch.save({
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "ch_names": ch_names,
            "times": times
        }, cache_path)

    eeg_channels = y_test.shape[1]
    eeg_time_points = y_test.shape[2]

    # =============================================================================
    # 4. Define Grid
    # =============================================================================
    grid = list(ParameterGrid(args.hyperparameter_space))

    # =============================================================================
    # 5. Search Loop
    # =============================================================================
    with open(log_file, "w") as f, open(detail_log_file, "w") as fd:
        f.write(f">> {args.model} + {args.optim} <<\n")
        f.write("-" * 50 + "\n")

        fd.write(f">> {args.model} + {args.optim} <<\n")
        fd.write("-" * 50 + "\n")

        best_overall_loss = float("inf")
        best_overall_config = None

        for idx, config in enumerate(grid):
            combo_str = f"[{idx+1}/{len(grid)}] Config: {config}"
            print(f"\n{combo_str}")
            f.write(f"\n{combo_str}\n")
            fd.write(f"\n{combo_str}\n")
            f.flush()
            fd.flush()

            # Create DataLoaders for this batch_size
            args.batch_size = config["batch_size"]
            train_dl, val_dl, _ = create_dataloader(
                args, 0, g_cpu, X_train, X_val, X_test, y_train, y_val, y_test
            )

            # Run Training (Inlined)
            patience = args.patience
            combo_id = idx + 1
            
            model = AlexEEGNet(num_channels=eeg_channels, time_points=eeg_time_points)
            model.to(device)

            # Configure parameter groups exactly like train.py
            param_fast = [
                {"params": model.features.parameters(), "lr": config["lr"] * 0.1},
                {"params": model.classifier[1].parameters(), "lr": config["lr"]},
                {"params": model.lstm.parameters(), "lr": config["lr"]},
                {"params": model.channel_decoder.parameters(), "lr": config["lr"]},
            ]
            param_mid = [
                {"params": model.classifier[4].parameters(), "lr": config["lr"]},
            ]
            param_slow = [
                {"params": model.classifier[6].parameters(), "lr": config["lr"]},
            ]

            params_list = [param_fast, param_mid, param_slow]

            freq = config.get("freq", (1, 8, 16))
            if args.optim != "NestedAdam":
                freq = (1,) * len(freq)
            chunk_size = config.get("chunk_size", (8, 8, 8))
            alpha = config.get("alpha", 0.5)
            beta = config.get("beta", (0.9, 0.999, 0.9))
            lr = config.get("lr", 1e-5)
            weight_decay = config.get("weight_decay", 0.0)

            if args.optim == "AdamW":
                opts = [torch.optim.AdamW(p, lr=lr, weight_decay=weight_decay, betas=beta) for p in params_list]
            elif args.optim == "NestedAdam":
                opts = [
                    NestedAdam(p, lr=lr, weight_decay=weight_decay, alpha=alpha, beta=beta, chunk_size=chunk_size[i]) 
                    for i, p in enumerate(params_list)
                ]
            else:
                raise ValueError(f"Unsupported optimizer: {args.optim}")

            loss_fn = nn.MSELoss().to(device)
            torch.backends.cudnn.benchmark = True

            best_val_loss = float("inf")
            epochs_no_improve = 0

            # Progress bar for Epochs within this specific combination
            pbar = tqdm(range(args.epochs), desc=f"Combo {combo_id}", unit="epoch", leave=False)

            for epoch in pbar:
                # --- Training ---
                model.train()
                train_loss = 0.0
                global_step_offset = epoch * len(train_dl)
                
                # Buffer setup matching train.py
                X_buffer, y_buffer = [None] * freq[-1], [None] * freq[-1]
                
                for batch_idx, (X, y) in enumerate(train_dl):
                    current_step = global_step_offset + batch_idx + 1
                    data_idx = current_step % freq[-1]
                    X_buffer[data_idx], y_buffer[data_idx] = X.to(device), y.to(device)

                    update_indices = [i for i, f in enumerate(freq) if current_step % f == 0]
                    
                    if update_indices:
                        for idx_update in range(len(update_indices) - 1, -1, -1):
                            curr_i = update_indices[idx_update]
                            next_i = update_indices[idx_update - 1] if idx_update > 0 else None
                            end = freq[next_i] if next_i is not None else 0
                            indices = [i % freq[-1] for i in range(data_idx - freq[curr_i] + 1, data_idx - end + 1)]

                            if indices:
                                X = torch.cat([X_buffer[i] for i in indices], dim=0)
                                y = torch.cat([y_buffer[i] for i in indices], dim=0)
                                
                                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                    pred = model(X).squeeze()
                                    loss = loss_fn(pred, y)
                                loss.backward()
                                
                                for i in range(curr_i):
                                    opts[i].zero_grad()

                        for i in update_indices:
                            opts[i].step()

                        for opt in opts:
                            opt.zero_grad()
                    
                    with torch.no_grad():
                        X, y = X_buffer[data_idx].to(device), y_buffer[data_idx].to(device)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            pred = model(X).squeeze()
                            loss = loss_fn(pred, y)
                        train_loss += loss.item() * X.size(0)
                train_loss /= len(train_dl.dataset)

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

                fd.write(
                    f"    Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
                )
                fd.flush()

                # Early Stopping Logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

            best_val = best_val_loss
            epochs_run = epoch + 1
            
            print(f"-> Best Val Loss: {best_val:.4f} (Stopped at Epoch {epochs_run})")
            f.write(f"-> Best Val Loss: {best_val:.4f} (Stopped at Epoch {epochs_run})\n")
            fd.write(f"-> Best Val Loss: {best_val:.4f} (Stopped at Epoch {epochs_run})\n")

            if best_val_loss < best_overall_loss:
                best_overall_loss = best_val_loss
                best_overall_config = config
                f.write(">>> New Best Found! <<<\n")
                fd.write(">>> New Best Found! <<<\n")

            f.write("\n")
            fd.write("\n")
            f.flush()
            fd.flush()

        # =============================================================================
        # 6. Final Statistics Summary
        # =============================================================================
        summary_str = "=" * 50 + "\n"
        summary_str += "Optimization Completed\n"
        summary_str += f"Best Validation Loss: {best_overall_loss:.4f}\n"
        summary_str += "Optimal Hyperparameters:\n"
        for k, v in best_overall_config.items():
            summary_str += f"  - {k}: {v}\n"
        summary_str += "=" * 50 + "\n"

        print(summary_str)
        f.write(summary_str)
        fd.write(summary_str)


if __name__ == "__main__":
    main()
