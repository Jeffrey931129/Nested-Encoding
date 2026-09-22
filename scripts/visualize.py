import ast
import math
import os
import re
import sys
import time
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Add src directory and repository root to sys.path to allow importing modules from src
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from data_utils import data_dir, experiment_dir, figure_dir
except ImportError:
    data_dir = os.path.join(root_dir, "data")
    experiment_dir = os.path.join(root_dir, "experiment")
    figure_dir = os.path.join(root_dir, "figures")


# =============================================================================
# Configuration Class
# =============================================================================
class Args:
    def __init__(self):
        # I/O arguments
        self.sub = 1
        
        # Optional: Specify which DNN folders to process (list of strings or None)
        self.target_dnns = None
        


args = Args()

# =============================================================================
# Plotting Configuration
# =============================================================================
matplotlib.rcParams["font.sans-serif"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 20
plt.rc("xtick", labelsize=20)
plt.rc("ytick", labelsize=20)
matplotlib.rcParams["axes.linewidth"] = 2
matplotlib.rcParams["xtick.major.width"] = 2
matplotlib.rcParams["ytick.major.width"] = 2
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
color_noise_ceiling = (150 / 255, 150 / 255, 150 / 255)

import matplotlib.cm as cm

cmap = matplotlib.colormaps.get_cmap("tab20")

# =============================================================================
# Main Logic
# =============================================================================

if not os.path.exists(experiment_dir):
    print(f"Error: Experiment directory {experiment_dir} not found.")
    exit()

dir = experiment_dir

# --- Step 1: Collect all data ---
all_plottable_data = []

for root_path, dirs, files in os.walk(dir):
    rel_path = os.path.relpath(root_path, dir)
    if rel_path != '.':
        top_level_dir = rel_path.split(os.sep)[0]
        if args.target_dnns and top_level_dir not in args.target_dnns:
            continue
            
    for file in files:
        if file.endswith(".npy"):
            fpath = os.path.join(root_path, file)
            
            if rel_path == '.':
                folder_label = "experiment"
            else:
                folder_label = rel_path.replace(os.sep, " | ")
                
            try:
                data = np.load(fpath, allow_pickle=True).item()

                if "correlation" not in data or "times" not in data:
                    continue
                keys = list(data["correlation"].keys())
                if not keys:
                    continue

                target_key = keys[0]
                
                if file == ".npy":
                    full_label = folder_label
                else:
                    full_label = f"{folder_label} | {os.path.splitext(file)[0]}"

                all_plottable_data.append(
                    {
                        "label": full_label,
                        "data": data,
                        "key": target_key,
                        "dnn_group": folder_label,
                    }
                )

            except Exception as e:
                print(f"  Cannot read {os.path.basename(fpath)}: {e}")

if not all_plottable_data:
    print("Error: No valid data was read.")
    exit()



times = all_plottable_data[0]["data"]["times"]
num_total = len(all_plottable_data)

min_len = min(
    [len(item["data"]["significance"][item["key"]]) for item in all_plottable_data]
)
sig_matrix = np.zeros((num_total, min_len))

for i, item in enumerate(all_plottable_data):
    s_data = item["data"]["significance"][item["key"]][:min_len]
    for t in range(len(s_data)):
        if s_data[t] == False:
            sig_matrix[i, t] = -100
        else:
            sig_matrix[i, t] = -0.085 + (
                abs(i + 4.25 - num_total) / (num_total * 2 + 10) * 1.75
            )

# =============================================================================
# Plot 1: Comparison Line Plot (All in One)
# =============================================================================
# Set figsize to 32x18 and DPI to 120 for 4K resolution (3840x2160)
fig1 = plt.figure(figsize=(32, 18))
unique_groups = set([item["dnn_group"].split(" | ")[0] for item in all_plottable_data])
plt.title(f"Model Comparison ({len(unique_groups)} Groups)", fontsize=30, pad=20)

plt.plot(
    [-10, 10], [0, 0], "k--", [0, 0], [10, -10], "k--", label="_nolegend_", linewidth=3
)

for i, item in enumerate(all_plottable_data):
    color = cmap(i / num_total) if num_total > 1 else cmap(0)
    data_dict = item["data"]
    key = item["key"]
    label = item["label"]
    
    if "test_loss" in data_dict:
        label += f" | Loss: {data_dict['test_loss']:.6f}"

    corr = data_dict["correlation"][key]

    if corr.ndim == 3:
        mean_corr = np.mean(np.mean(corr, 0), 0)
    elif corr.ndim == 2:
        mean_corr = np.mean(corr, 0)
    else:
        mean_corr = corr

    p_len = min(len(times), len(mean_corr))

    plt.plot(times[:p_len], mean_corr[:p_len], color=color, linewidth=3, label=label)

nc_cache_dir = os.path.join(data_dir, "cache")
prefix = f"sub-{args.sub:02d}_noise_ceiling_"
nc_files = [f for f in os.listdir(nc_cache_dir) if f.startswith(prefix) and f.endswith(".npy")] if os.path.exists(nc_cache_dir) else []

if nc_files:
    # Prioritize selecting the cache file with the highest number of iterations
    nc_files.sort(key=lambda x: int(x[len(prefix):-4]) if x[len(prefix):-4].isdigit() else 0, reverse=True)
    nc_cache_path = os.path.join(nc_cache_dir, nc_files[0])
    nc_cache_data = np.load(nc_cache_path, allow_pickle=True).item()
    nc_low = nc_cache_data["noise_ceiling_low"]
    nc_up = nc_cache_data["noise_ceiling_up"]
    
    if nc_low.ndim == 3:
        nc_low = np.mean(np.mean(nc_low, 0), 0)
        nc_up = np.mean(np.mean(nc_up, 0), 0)
    elif nc_low.ndim == 2:
        nc_low = np.mean(nc_low, 0)
        nc_up = np.mean(nc_up, 0)
        
    plt.fill_between(
        times[: len(nc_low)],
        nc_low,
        nc_up,
        color=color_noise_ceiling,
        alpha=0.3,
        label=f"Noise Ceiling (iters: {os.path.splitext(nc_files[0])[0]})",
    )

plt.xlabel("Time (s)", fontsize=24)
plt.ylabel("Pearson's $r$", fontsize=24)

xticks = [-0.2, 0, 0.2, 0.4, 0.6, max(times)]
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, round(max(times), 1)]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))
plt.ylim(bottom=-0.1, top=1)

plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout()

# Save Plot 1 as JPG, setting DPI=300 for high resolution
out_dir = figure_dir
os.makedirs(out_dir, exist_ok=True)
plot1_filename = os.path.join(out_dir, "model_comparison.jpg")
plt.savefig(plot1_filename, format="jpg", dpi=120)

plt.close(fig1)  # Free memory

# =============================================================================
# Plot 2: Per-Channel Temporal Dynamics (Refactored using plt.figure)
# =============================================================================
first_data_corr = all_plottable_data[0]["data"]["correlation"][
    all_plottable_data[0]["key"]
]
if first_data_corr.ndim == 3:
    num_channels = first_data_corr.shape[1]
elif first_data_corr.ndim == 2:
    num_channels = first_data_corr.shape[0]
else:
    num_channels = 0

if num_channels == 0:
    print("Cannot determine the number of channels or incorrect tensor dimension.")
else:
    # --- [Critical Protection Mechanism] Prevent rendering crash due to high subplot density ---
    # On a 2K resolution (2560x1600) canvas, exceeding 64 subplots (16 rows x 4 columns) causes visual clutter or errors
    MAX_CHANNELS = 64
    if num_channels > MAX_CHANNELS:
        print(
            f"Warning: Number of channels ({num_channels}) is too large. To ensure successful 2K image rendering, only the first {MAX_CHANNELS} channels will be plotted."
        )
        plot_channels = MAX_CHANNELS
    else:
        plot_channels = num_channels

    cols = 4
    rows = int(math.ceil(plot_channels / cols))

    # Re-call plt.figure and force figsize to 32x18 and DPI to 120 for 4K resolution (3840x2160)
    fig2 = plt.figure(figsize=(32, 18))
    fig2.suptitle(
        f"Time-Resolved Encoding Performance per Channel", fontsize=20, y=1.05
    )

    ch_names = ['Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2']

    for c in range(plot_channels):
        # Manually create subplots one by one to replace the original plt.subplots
        ax = fig2.add_subplot(rows, cols, c + 1)
        
        ch_title = ch_names[c] if c < len(ch_names) else f"Channel {c}"
        ax.set_title(ch_title, fontsize=12)
        ax.plot([min(times), max(times)], [0, 0], "k--", linewidth=1.5)

        for i, item in enumerate(all_plottable_data):
            color = cmap(i / num_total) if num_total > 1 else cmap(0)
            data_dict = item["data"]
            key = item["key"]
            label = item["label"]
            raw_corr = data_dict["correlation"][key]

            if raw_corr.ndim == 3:
                chan_time_data = np.mean(raw_corr, axis=0)
            elif raw_corr.ndim == 2:
                chan_time_data = raw_corr
            else:
                continue

            time_course = chan_time_data[c, :]
            p_len = min(len(times), len(time_course))

            ax.plot(
                times[:p_len],
                time_course[:p_len],
                color=color,
                linewidth=1.5,
                label=label if c == 0 else "",
            )

        # Set Y-axis label (only for the leftmost column)
        if c % cols == 0:
            ax.set_ylabel("Pearson's $r$", fontsize=10)
            ax.tick_params(axis="y", labelsize=10)

        # Set X-axis label (only for the bottommost row)
        if c >= (rows - 1) * cols or (c + cols) >= plot_channels:
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, max(times)])
            ax.set_xticklabels([-0.2, 0, 0.2, 0.4, 0.6, round(max(times), 1)])
            ax.tick_params(axis="x", labelsize=10)

    # Extract legend
    handles, labels = fig2.axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if "" in by_label:
        del by_label[""]

    fig2.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(4, len(by_label)),
        frameon=False,
    )

    # Adjust layout and include exception handling as a precaution
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    except Exception as e:
        print(f"Layout adjustment warning: {e} (Ignored to ensure image output)")

    plot2_filename = os.path.join(out_dir, "channel_dynamics.jpg")
    plt.savefig(plot2_filename, format="jpg", dpi=120)

    plt.close(fig2)  # Free memory

# =============================================================================
# Plot 3 & 4: Hyperparameter Configuration Mean and Variance
# =============================================================================

exp_dir = experiment_dir

config_pattern = re.compile(r"\[\d+/\d+\] Config:\s*(\{.*\})")
loss_pattern = re.compile(r"-> (?:Best|Average) Val Loss:\s*([0-9.]+)")

all_results = []
if os.path.exists(exp_dir):
    for root_path, dirs, files in os.walk(exp_dir):
        rel_path = os.path.relpath(root_path, exp_dir)
        if rel_path == '.':
            continue
        else:
            parts = rel_path.split(os.sep)
            if len(parts) >= 2:
                folder_name = f"{parts[0]} | {parts[1]}"
            else:
                folder_name = parts[0]
            
        for file in files:
            if file.endswith(".log") and not file.endswith("detail.log"):
                log_path = os.path.join(root_path, file)
                current_config = None
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        config_match = config_pattern.search(line)
                        if config_match:
                            try:
                                current_config = ast.literal_eval(config_match.group(1))
                            except:
                                current_config = None
                            continue
                        
                        loss_match = loss_pattern.search(line)
                        if loss_match and current_config is not None:
                            val_loss = float(loss_match.group(1))
                            
                            # Create a stable string representation for the config
                            sorted_items = sorted(current_config.items(), key=lambda x: x[0])
                            config_str = str(sorted_items)
                            
                            all_results.append({
                                "folder_name": folder_name,
                                "config_str": config_str,
                                "best_val_loss": val_loss
                            })
                            current_config = None

plot3_filename = os.path.join(out_dir, "hyperparam_analysis.jpg")

if all_results:
    df = pd.DataFrame(all_results)
    
    # Compute mean and std for EACH subfolder
    folder_stats = df.groupby('folder_name')['best_val_loss'].agg(['mean', 'std']).reset_index()
    
    folder_names = sorted(folder_stats['folder_name'].unique())
    num_folders = len(folder_names)
    cmap_hp = plt.get_cmap("tab20")
    
    # --- Plot 3: Mean and Std as points on two vertical lines ---
    fig3, ax1 = plt.subplots(figsize=(32, 18))
    plt.title(r"Validation Loss Performance per Subfolder", fontsize=30, pad=20)
    
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Mean", "Standard Deviation"], fontsize=24)
    
    # Draw two vertical lines
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax1.axvline(x=1, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    
    ax2 = ax1.twinx()
    
    for i, fname in enumerate(folder_names):
        color = cmap_hp(i / num_folders) if num_folders > 1 else cmap_hp(0)
        
        row = folder_stats[folder_stats['folder_name'] == fname].iloc[0]
        f_mean = row['mean']
        f_std = row['std'] if not pd.isna(row['std']) else 0.0
        
        # Plot an invisible line to create a line in the legend
        ax1.plot([], [], color=color, linewidth=3, label=fname)
        
        # Plot Mean on ax1 (x=0) without a label so the dot doesn't show in the legend
        ax1.plot([0], [f_mean], marker='_', markersize=50, markeredgewidth=5, color=color, linestyle='None')
        # Plot Std on ax2 (x=1)
        ax2.plot([1], [f_std], marker='_', markersize=50, markeredgewidth=5, color=color, linestyle='None')
    
    ax1.set_ylabel("Mean Validation Loss", fontsize=24)
    ax2.set_ylabel("Standard Deviation", fontsize=24)
    ax1.tick_params(axis='y', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    
    ax1.legend(fontsize=20, loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)
    
    plt.tight_layout()
    plt.savefig(plot3_filename, format="jpg", dpi=120)

    plt.close(fig3)
else:
    print("No valid log data found in the experiment folder.")

# =============================================================================
# Plot 4: Random Test Prediction to Nearest Training Sample Match
# =============================================================================
import torch

valid_data_with_preds = [item for item in all_plottable_data if "predictions" in item["data"]]

if not valid_data_with_preds:
    print("No predictions found in the log data. Skipping Plot 4.")
    print("Hint: Rerun train.py with the updated script to save predictions to .npy files.")
else:
    cache_dir = os.path.join(data_dir, "cache")
    img_cache_path = os.path.join(cache_dir, "image_data_cache.pt")
    eeg_cache_path = os.path.join(cache_dir, f"sub-{args.sub:02d}_eeg_data_avg.pt")
    
    if os.path.exists(img_cache_path) and os.path.exists(eeg_cache_path):
        print("Loading cached image and EEG data for Plot 4...")
        img_cache_data = torch.load(img_cache_path, weights_only=False)
        X_train = img_cache_data["X_train"]
        X_test = img_cache_data["X_test"]
        
        eeg_cache_data = torch.load(eeg_cache_path, weights_only=False)
        y_train = eeg_cache_data["y_train"]
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()
            
        # 1. Randomly pick test predictions based on the first model (assuming uniform test set)
        num_test_samples = valid_data_with_preds[0]["data"]["predictions"].shape[0]
        # Ensure we don't index out of bounds in case of shape mismatch
        num_test_samples = min(num_test_samples, len(X_test))
        num_samples_to_plot = min(3, num_test_samples)
        rand_indices = np.random.choice(num_test_samples, num_samples_to_plot, replace=False)
        
        # 2. Figure setup
        num_models = len(valid_data_with_preds)
        plots_per_sample = 1 + num_models
        total_plots = num_samples_to_plot * plots_per_sample
        
        cols = min(5, plots_per_sample) if plots_per_sample <= 5 else min(4, total_plots)
        rows = math.ceil(total_plots / cols)
        
        fig4, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        fig4.suptitle("Test Prediction vs. Most Similar Training Sample by Model", fontsize=24)
        
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
            
        # Hide any unused subplots
        for ax in axes_flat[total_plots:]:
            ax.axis('off')
            
        def imshow(img_tensor, ax, title):
            # De-normalize image
            img = img_tensor.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            # Use wrap for long titles
            import textwrap
            wrapped_title = "\n".join(textwrap.wrap(title, width=35))
            # Pad to 3 lines to ensure consistent image alignment
            newlines = wrapped_title.count('\n')
            if newlines < 2:
                wrapped_title += '\n' * (2 - newlines)
            ax.set_title(wrapped_title, fontsize=14)
            ax.axis('off')
            
        # Precompute common elements for Pearson Correlation
        y_train_flat = y_train.reshape(y_train.shape[0], -1)
        y_train_mean = np.mean(y_train_flat, axis=1, keepdims=True)
        y_train_centered = y_train_flat - y_train_mean
        y_var = np.sum(y_train_centered ** 2, axis=1)

        # 3. Find most similar training sample for each model
        plot_idx = 0
        for rand_idx in rand_indices:
            imshow(X_test[rand_idx], axes_flat[plot_idx], f"Target Test Image\n(Idx: {rand_idx})")
            plot_idx += 1
            
            for idx, item in enumerate(valid_data_with_preds):
                pred_data = item["data"]["predictions"]
                if rand_idx >= pred_data.shape[0]:
                    print(f"Warning: Model {item['label']} has less predictions than rand_idx. Skipping.")
                    axes_flat[plot_idx].axis('off')
                    plot_idx += 1
                    continue
                    
                test_pred = pred_data[rand_idx]  # shape: (channels, time_points)
                test_pred_flat = test_pred.reshape(1, -1)
                test_pred_mean = np.mean(test_pred_flat, axis=1, keepdims=True)
                test_pred_centered = test_pred_flat - test_pred_mean
                
                cov = np.sum(y_train_centered * test_pred_centered, axis=1)
                test_var = np.sum(test_pred_centered ** 2, axis=1)
                
                denom = np.sqrt(y_var * test_var)
                valid = denom > 0
                corr = np.zeros_like(denom)
                corr[valid] = cov[valid] / denom[valid]
                
                best_train_idx = np.argmax(corr)
                best_corr_val = corr[best_train_idx]
                
                label = item["label"]
                if "test_loss" in item["data"]:
                    label += f" | Loss: {item['data']['test_loss']:.6f}"
                    
                title = f"{label}\nTrain Match: {best_train_idx} (r={best_corr_val:.3f})"
                imshow(X_train[best_train_idx], axes_flat[plot_idx], title)
                plot_idx += 1
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot4_filename = os.path.join(out_dir, "prediction_similarity.jpg")
        plt.savefig(plot4_filename, format="jpg", dpi=120)
        plt.close(fig4)
        print(f"Plot 4 saved to {plot4_filename}")
    else:
        print("Missing cached data (.pt files). Cannot plot prediction similarity.")