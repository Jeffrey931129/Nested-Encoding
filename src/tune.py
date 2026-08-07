import os
import numpy as np
import random
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.utils import resample
from nested_sgd import NestedSGD
from nested_adam import NestedAdam
from model import CustomModel

# Import your custom utilities
from data_utils import load_images
from data_utils import load_eeg_data
from data_utils import create_dataloader

# =============================================================================
# Configuration Class
# =============================================================================
class Args:
    def __init__(self):
        # Core defaults
        self.sub = 1
        self.model = "AlexNet+NestedAdam"  
        self.batch_size = 32
        
        # I/O arguments
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        self.project_dir = os.path.join(root_dir, "data")


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
    if model_type == "AlexNet+NestedAdam":
        import torchvision.models as models
        model = models.alexnet(pretrained=False)
        
        # Keep original AlexNet structure, add a new head mapping 1000 -> EEG dimensions
        layers = list(model.classifier.children())
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.Linear(1000, eeg_channels * eeg_time_points))
        model.classifier = nn.Sequential(*layers)
        
        model = model.to(device)

        # Configure parameter groups for AlexNet
        param_fast = [
            {"params": model.features.parameters(), "lr": config["lr"] * 0.1},
            {"params": model.classifier[0:3].parameters(), "lr": config["lr"]},
            {"params": model.classifier[9:10].parameters(), "lr": config["lr"]},
        ]
        param_mid = [
            {"params": model.classifier[3:6].parameters(), "lr": config["lr"]},
        ]
        param_slow = [
            {"params": model.classifier[6:9].parameters(), "lr": config["lr"]},
        ]
    else:
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
    f_fast, f_mid, f_slow = config.get("freq", (1, 1, 1))
    c_fast, c_mid, c_slow = config.get("chunk_size", (1, 1, 1))

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
            chunk_size=c_fast,
            weight_decay=config["weight_decay"],
        )
        opt_mid = NestedSGD(
            param_mid,
            lr=config["lr"],
            momentum=config["momentum"],
            alpha=config["alpha"],
            chunk_size=c_mid,
            weight_decay=config["weight_decay"],
        )
        opt_slow = NestedSGD(
            param_slow,
            lr=config["lr"],
            momentum=config["momentum"],
            alpha=config["alpha"],
            chunk_size=c_slow,
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
    elif model_type in ["adam+nested", "AlexNet+NestedAdam"]:
        opt_fast = NestedAdam(
            param_fast,
            lr=config["lr"],
            beta=config["beta"],
            eps=config["eps"],
            alpha=config["alpha"],
            chunk_size=c_fast,
            weight_decay=config["weight_decay"],
            freq=f_fast,
        )
        opt_mid = NestedAdam(
            param_mid,
            lr=config["lr"],
            beta=config["beta"],
            eps=config["eps"],
            alpha=config["alpha"],
            chunk_size=c_mid,
            weight_decay=config["weight_decay"],
            freq=f_mid,
        )
        opt_slow = NestedAdam(
            param_slow,
            lr=config["lr"],
            beta=config["beta"],
            eps=config["eps"],
            alpha=config["alpha"],
            chunk_size=c_slow,
            weight_decay=config["weight_decay"],
            freq=f_slow,
        )

    loss_fn = nn.MSELoss().to(device)

    best_val_loss = float("inf")
    epochs_no_improve = 0

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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                f"        Epoch {epoch + 1:03d} | Train Loss: {train_loss / len(train_dl):.4f} | Val Loss: {val_loss:.4f}\n"
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
    
    log_dir = os.path.join("experiment", "tmp")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{formatted_time}.log")
    detail_log_file = os.path.join(log_dir, f"{formatted_time}_detail.log")
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
    # Load ALL training data into memory at once to avoid disk I/O per fold
    idx_val_all_false = np.zeros((16540), dtype=bool)
    
    X_all, _, X_test = load_images(args, idx_val_all_false)
    y_all, _, y_test, _, _ = load_eeg_data(args, idx_val_all_false)
    
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
            "chunk_size": [(10, 10, 10), (25, 25, 25)],
            "freq": [(1, 2, 4), (1, 4, 8)],
        }
    elif args.model == "adam":
        hyperparameter_space = {
            "lr": [1e-4, 5e-5, 1e-5],
            "batch_size": [32, 64],
            "weight_decay": [0, 1e-4, 1e-5],
        }
    elif args.model in ["adam+nested", "AlexNet+NestedAdam"]:
        hyperparameter_space = {
            "lr": [5e-6, 1e-5, 2e-5],
            "batch_size": [32],
            "alpha": [0.0, 0.5, 1.0, 2.0],
            "beta": [(0.9, 0.999)],
            "eps": [1e-8],
            "chunk_size": [(4, 4, 4), (8, 8, 8), (16, 16, 16)],
            "freq": [(1, 4, 8), (1, 4, 16), (1, 4, 16)],
            "weight_decay": [0.0, 1e-5],
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

            # Set up 5-Fold Cross Validation
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            fold_val_losses = []
            
            for fold_idx, (train_index, val_index) in enumerate(kf.split(train_img_concepts)):
                print(f"  --- Fold {fold_idx + 1}/5 ---")
                fd.write(f"    --- Fold {fold_idx + 1}/5 ---\n")
                
                # Create validation mask for this fold
                val_mask = np.zeros(16540, dtype=bool)
                for i in val_index:
                    val_mask[i * 10 : i * 10 + 10] = True
                    
                # Slice the in-memory data
                X_train_fold = [X_all[i] for i in range(16540) if not val_mask[i]]
                X_val_fold = [X_all[i] for i in range(16540) if val_mask[i]]
                y_train_fold = y_all[~val_mask]
                y_val_fold = y_all[val_mask]

                # Create DataLoaders for this batch_size and fold
                args.batch_size = config["batch_size"]
                train_dl, val_dl, _ = create_dataloader(
                    args, 0, g_cpu, X_train_fold, X_val_fold, X_test, y_train_fold, y_val_fold, y_test
                )

                # Run Training for this fold
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
                
                fold_val_losses.append(best_val)
                print(f"  -> Fold {fold_idx + 1} Best Val Loss: {best_val:.4f} (Stopped at Epoch {epochs_run})")
                f.write(f"    Fold {fold_idx + 1} Best Val Loss: {best_val:.4f} (Stopped at Epoch {epochs_run})\n")
                fd.write(f"    Fold {fold_idx + 1} Best Val Loss: {best_val:.4f} (Stopped at Epoch {epochs_run})\n")

            # Average Validation Loss across 5 folds
            avg_val_loss = np.mean(fold_val_losses)

            # Log results
            f.write(
                f"-> Average Val Loss: {avg_val_loss:.4f}\n"
            )
            fd.write(
                f"-> Average Val Loss: {avg_val_loss:.4f}\n"
            )
            print(f"-> Average Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_overall_loss:
                best_overall_loss = avg_val_loss
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
