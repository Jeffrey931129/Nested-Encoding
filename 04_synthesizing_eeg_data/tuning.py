import os
import numpy as np
import random
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.utils import resample
from nested_sgd import NestedSGD
from nested_adam import NestedAdam
from custom_model import CustomModel

# Import your custom utilities
from end_to_end_encoding_utils import load_images
from end_to_end_encoding_utils import load_eeg_data
from end_to_end_encoding_utils import create_dataloader


# =============================================================================
# Configuration Class
# =============================================================================
class Args:
    def __init__(self):
        # Core defaults
        self.sub = 1
        self.modeled_time_points = "all"
        self.model = "adam+nested"  
        self.batch_size = 32
        
        # I/O arguments
        self.project_dir = "project_directory"


# =============================================================================
# Core Training Function with Progress Bar and Early Stopping
# =============================================================================
def train_and_evaluate(
    config,
    train_dl,
    val_dl,
    device,
    eeg_channels,
    eeg_time_points,
    combo_id,
    model_type="gradient",
    max_epochs=200,
    patience=15,
    log_file=None,
):
    model = CustomModel(num_channels=eeg_channels, time_points=eeg_time_points, hidden_dim=512, num_layers=3).to(device)

    # Configure parameter groups
    param_fast = [
        {"params": model.features[0:4].parameters(), "lr": config["lr"] * 0.1},
        {"params": model.feature_projection.parameters(), "lr": config["lr"]},
        {"params": model.lstm_layers[2].parameters(), "lr": config["lr"]},
        {"params": model.channel_decoder.parameters(), "lr": config["lr"]},
    ]
    param_mid = [
        {"params": model.features[4:9].parameters(), "lr": config["lr"] * 0.1},
        {"params": model.lstm_layers[1].parameters(), "lr": config["lr"]},
    ]
    param_slow = [
        {"params": model.features[9:13].parameters(), "lr": config["lr"] * 0.1},
        {"params": model.lstm_layers[0].parameters(), "lr": config["lr"]},
    ]

    # Select optimizer based on model_type
    if model_type == "gradient":
        opt_fast = torch.optim.SGD(
            param_fast + param_mid + param_slow,
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
        opt_mid = None
        opt_slow = None
    elif model_type == "gradient+nested":
        opt_fast = NestedSGD(
            param_fast,
            lr=config["lr"],
            momentum=config["momentum"],
            alpha=config["alpha"],
            chunk_size=config["chunk_size"],
            weight_decay=config["weight_decay"],
        )
        opt_mid = NestedSGD(
            param_mid,
            lr=config["lr"],
            momentum=config["momentum"],
            alpha=config["alpha"],
            chunk_size=config["chunk_size"],
            weight_decay=config["weight_decay"],
        )
        opt_slow = NestedSGD(
            param_slow,
            lr=config["lr"],
            momentum=config["momentum"],
            alpha=config["alpha"],
            chunk_size=config["chunk_size"],
            weight_decay=config["weight_decay"],
        )
    elif model_type == "adam":
        opt_fast = torch.optim.Adam(
            param_fast + param_mid + param_slow,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        opt_mid = None
        opt_slow = None
    elif model_type == "adam+nested":
        opt_fast = NestedAdam(
            param_fast,
            lr=config["lr"],
            beta=config["beta"],
            eps=config["eps"],
            alpha=config["alpha"],
            chunk_size=config["chunk_size"],
            weight_decay=config["weight_decay"],
        )
        opt_mid = NestedAdam(
            param_mid,
            lr=config["lr"],
            beta=config["beta"],
            eps=config["eps"],
            alpha=config["alpha"],
            chunk_size=config["chunk_size"],
            weight_decay=config["weight_decay"],
        )
        opt_slow = NestedAdam(
            param_slow,
            lr=config["lr"],
            beta=config["beta"],
            eps=config["eps"],
            alpha=config["alpha"],
            chunk_size=config["chunk_size"],
            weight_decay=config["weight_decay"],
        )

    loss_fn = nn.MSELoss().to(device)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best_val_loss = float("inf")
    epochs_no_improve = 0

    f_fast, f_mid, f_slow = config.get("freq", (1, 1, 1))

    # Progress bar for Epochs within this specific combination
    pbar = tqdm(range(max_epochs), desc=f"Combo {combo_id}", unit="epoch", leave=False)

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

            train_loss += loss.item()

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

        if log_file:
            log_file.write(
                f"      Epoch {epoch + 1:03d} | Train Loss: {train_loss / len(train_dl):.4f} | Val Loss: {val_loss:.4f}\n"
            )
            log_file.flush()

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
    return best_val_loss, epoch + 1


# =============================================================================
# Main Execution Logic
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize Log File with Timestamp
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    log_file = f"{formatted_time}.log"
    detail_log_file = f"{formatted_time}_detail.log"
    print(f"Tuning started. Results will be saved to: {log_file} and {detail_log_file}")

    # 2. Setup Data (Reusing your logic)
    args = Args()

    # Seeds for reproducibility
    seed = 20200220
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)

    train_img_concepts = np.arange(1654)
    val_concepts = np.sort(resample(train_img_concepts, replace=False, n_samples=100))
    idx_val = np.zeros((16540), dtype=bool)
    for i in val_concepts:
        idx_val[i * 10 : i * 10 + 10] = True

    X_train, X_val, X_test = load_images(args, idx_val)
    y_train, y_val, y_test, _, _ = load_eeg_data(args, idx_val)
    eeg_channels = y_test.shape[1]
    eeg_time_points = y_test.shape[2]

    # 3. Define Grid
    if args.model == "gradient":
        hyperparameter_space = {
            "lr": [1e-3, 5e-4, 1e-4],
            "batch_size": [32, 64],
            "momentum": [0.9, 0.95],
            "weight_decay": [0, 1e-3, 1e-4],
            "freeze_conv_base": [True, False],
        }
    elif args.model == "gradient+nested":
        hyperparameter_space = {
            "lr": [1e-3],
            "batch_size": [32],
            "momentum": [0.95],
            "weight_decay": [0],
            "freeze_conv_base": [False],
            "alpha": [0.9],
            "chunk_size": [10, 25],
            "freq": [(1, 2, 4), (1, 4, 8)],
        }
    elif args.model == "adam":
        hyperparameter_space = {
            "lr": [1e-4, 5e-5, 1e-5],
            "batch_size": [32, 64],
            "weight_decay": [0, 1e-4, 1e-5],
        }
    elif args.model == "adam+nested":
        hyperparameter_space = {
            "lr": [1e-3, 1e-4, 1e-5],
            "batch_size": [32, 64],
            "alpha": [0.1, 0.5, 0.9],
            "beta": [(0.9, 0.999)],
            "eps": [1e-8],
            "chunk_size": [4, 10, 25],
            "freq": [(1, 1, 1), (1, 2, 4), (1, 4, 8)],
            "weight_decay": [0.0, 1e-4],
        }
    grid = list(ParameterGrid(hyperparameter_space))

    # 4. Search Loop
    with open(log_file, "w") as f, open(detail_log_file, "w") as fd:
        f.write(f">> {args.model} <<\n")
        f.write("-" * 50 + "\n")

        fd.write(f">> {args.model} <<\n")
        fd.write("-" * 50 + "\n")

        best_overall_loss = float("inf")
        best_overall_config = None  # Added tracking for the best configuration

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

            # Run Training
            best_val, epochs_run = train_and_evaluate(
                config,
                train_dl,
                val_dl,
                device,
                eeg_channels,
                eeg_time_points,
                combo_id=idx + 1,
                model_type=args.model,
                log_file=fd,
            )

            # Log results
            f.write(
                f"-> Best Val Loss: {best_val:.4f} (Stopped at Epoch {epochs_run})\n"
            )
            fd.write(
                f"-> Best Val Loss: {best_val:.4f} (Stopped at Epoch {epochs_run})\n"
            )
            print(f"-> Best Val Loss: {best_val:.4f} (Stopped at Epoch {epochs_run})")

            if best_val < best_overall_loss:
                best_overall_loss = best_val
                best_overall_config = config  # Save the best configuration
                f.write(">>> New Best Found! <<<\n")
                fd.write(">>> New Best Found! <<<\n")

            f.write("\n")
            fd.write("\n")
            f.flush()  # Ensure it writes to disk immediately
            fd.flush()

        # 5. Final Statistics Summary
        summary_str = "=" * 50 + "\n"
        summary_str += "Optimization Completed\n"
        summary_str += f"Best Validation Loss: {best_overall_loss:.4f}\n"
        summary_str += "Optimal Hyperparameters:\n"
        for k, v in best_overall_config.items():
            summary_str += f"  - {k}: {v}\n"
        summary_str += "=" * 50 + "\n"

        # Print the final summary to the console
        print(summary_str)
        # Write the final summary to the log file
        f.write(summary_str)
        fd.write(summary_str)
