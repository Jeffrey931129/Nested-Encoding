import argparse
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

# Import your custom utilities
from end_to_end_encoding_utils import load_images
from end_to_end_encoding_utils import load_eeg_data
from end_to_end_encoding_utils import create_dataloader


# =============================================================================
# Standardized AlexNet for EEG Prediction
# =============================================================================
class CustomAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =============================================================================
# Core Training Function with Progress Bar and Early Stopping
# =============================================================================
def train_and_evaluate(
    config,
    train_dl,
    val_dl,
    device,
    out_features,
    combo_id,
    model_type="gradient",
    max_epochs=200,
    patience=15,
):
    model = CustomAlexNet(num_classes=out_features).to(device)

    # Configure parameter groups based on freezing strategy
    if config["freeze_conv_base"]:
        for param in model.features.parameters():
            param.requires_grad = False
        param_fast = [{"params": model.classifier[1].parameters()}]
        param_mid = [{"params": model.classifier[4].parameters()}]
        param_slow = [{"params": model.classifier[6].parameters()}]
    else:
        param_fast = [
            {"params": model.features.parameters(), "lr": config["lr"] * 0.1},
            {"params": model.classifier[1].parameters()},
        ]
        param_mid = [{"params": model.classifier[4].parameters()}]
        param_slow = [{"params": model.classifier[6].parameters()}]

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
    print(f"Tuning started. Results will be saved to: {log_file}")

    # 2. Setup Data (Reusing your logic)
    class Args:
        pass

    args = Args()
    args.sub, args.modeled_time_points, args.project_dir = 1, "all", "project_directory"
    args.model = "gradient+nested"

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
    out_features = y_test.shape[1] * y_test.shape[2]

    # 3. Define Grid
    if args.model == "gradient":
        hyperparameter_space = {
            "lr": [1e-3, 5e-4, 1e-4],  # Shifted up based on your previous logs
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
    grid = list(ParameterGrid(hyperparameter_space))

    # 4. Search Loop
    with open(log_file, "w") as f:
        f.write(f">> {args.model} <<\n")
        f.write("-" * 50 + "\n")
        best_overall_loss = float("inf")
        best_overall_config = None  # Added tracking for the best configuration

        for idx, config in enumerate(grid):
            combo_str = f"[{idx+1}/{len(grid)}] Config: {config}"
            print(f"\n{combo_str}")

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
                out_features,
                combo_id=idx + 1,
                model_type=args.model,
            )

            # Log results
            log_entry = f"{combo_str}\n-> Best Val Loss: {best_val:.4f} (Stopped at Epoch {epochs_run})\n"
            f.write(log_entry)

            if best_val < best_overall_loss:
                best_overall_loss = best_val
                best_overall_config = config  # Save the best configuration
                f.write(">>> New Best Found! <<<\n")

            f.write("\n")
            f.flush()  # Ensure it writes to disk immediately

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
