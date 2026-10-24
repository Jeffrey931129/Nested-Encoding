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
# Configuration Class & Helper Functions
# =============================================================================
class Args:
    def __init__(self):
        # Subject selection: int (e.g. 1), list of ints (e.g. [1, 2]), str ("sub-01"), or None for all subjects
        self.sub = None

        # Optional: Specify which DNN folders to process (list of strings or None)
        self.target_dnns = None

        # Overall averaging method across subjects: "fisher_z" (recommended in neuroscience) or "arithmetic"
        self.average_method = "fisher_z"

        # Whether to display standard error of the mean (SEM) shaded region in the overall plot
        self.show_sem = True


def normalize_target_subs(sub_arg):
    if sub_arg is None:
        return None
    if isinstance(sub_arg, int):
        return {f"sub-{sub_arg:02d}"}
    if isinstance(sub_arg, str):
        if sub_arg.lower().startswith("sub-"):
            return {sub_arg.lower()}
        try:
            return {f"sub-{int(sub_arg):02d}"}
        except ValueError:
            return {sub_arg}
    if isinstance(sub_arg, (list, tuple, set)):
        normalized = set()
        for item in sub_arg:
            if isinstance(item, int):
                normalized.add(f"sub-{item:02d}")
            elif isinstance(item, str):
                if item.lower().startswith("sub-"):
                    normalized.add(item.lower())
                else:
                    try:
                        normalized.add(f"sub-{int(item):02d}")
                    except ValueError:
                        normalized.add(item)
        return normalized
    return None


def fisher_z_mean(corrs, axis=0):
    corrs = np.asarray(corrs)
    corrs = np.clip(corrs, -1 + 1e-7, 1 - 1e-7)
    return np.tanh(np.mean(np.arctanh(corrs), axis=axis))


def load_noise_ceiling(sub_id, data_dir):
    nc_cache_dir = os.path.join(data_dir, "cache")
    if not os.path.exists(nc_cache_dir):
        return None, None, None
    prefix = f"sub-{sub_id:02d}_noise_ceiling_"
    nc_files = [f for f in os.listdir(nc_cache_dir) if f.startswith(prefix) and f.endswith(".npy")]
    if not nc_files:
        return None, None, None

    # Prioritize selecting the cache file with the highest number of iterations
    nc_files.sort(
        key=lambda x: int(x[len(prefix) : -4]) if x[len(prefix) : -4].isdigit() else 0,
        reverse=True,
    )
    best_file = nc_files[0]
    nc_cache_path = os.path.join(nc_cache_dir, best_file)
    try:
        nc_cache_data = np.load(nc_cache_path, allow_pickle=True).item()
        nc_low = nc_cache_data["noise_ceiling_low"]
        nc_up = nc_cache_data["noise_ceiling_up"]

        if nc_low.ndim == 3:
            nc_low = np.mean(np.mean(nc_low, 0), 0)
            nc_up = np.mean(np.mean(nc_up, 0), 0)
        elif nc_low.ndim == 2:
            nc_low = np.mean(nc_low, 0)
            nc_up = np.mean(nc_up, 0)

        return nc_low, nc_up, best_file
    except Exception as e:
        print(f"  Warning: Cannot load noise ceiling from {nc_cache_path}: {e}")
        return None, None, None


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

cmap = matplotlib.colormaps.get_cmap("tab20")

# =============================================================================
# Main Logic
# =============================================================================

if not os.path.exists(experiment_dir):
    print(f"Error: Experiment directory {experiment_dir} not found.")
    exit()

dir = experiment_dir
target_subs = normalize_target_subs(args.sub)

# --- Step 1: Collect all data ---
all_collected_data = []

for root_path, dirs, files in os.walk(dir):
    rel_path = os.path.relpath(root_path, dir)
    if rel_path != '.':
        top_level_dir = rel_path.split(os.sep)[0]
        if args.target_dnns and top_level_dir not in args.target_dnns:
            continue

    for file in files:
        if file.endswith(".npy"):
            fpath = os.path.join(root_path, file)

            sub_match = re.search(r"sub-(\d+)", file, re.IGNORECASE)
            if sub_match:
                sub_tag = f"sub-{int(sub_match.group(1)):02d}"
            else:
                sub_tag = "other"

            if target_subs is not None and sub_tag not in target_subs:
                continue

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

                model_label = folder_label
                test_loss = data.get("test_loss")
                if test_loss is not None and isinstance(test_loss, (int, float)):
                    model_label += f" | Loss: {test_loss:.6f}"

                # Compute channel-averaged correlation
                corr = data["correlation"][target_key]
                if corr.ndim == 3:
                    chan_avg_corr = np.mean(np.mean(corr, 0), 0)
                elif corr.ndim == 2:
                    chan_avg_corr = np.mean(corr, 0)
                else:
                    chan_avg_corr = corr

                all_collected_data.append(
                    {
                        "label": full_label,
                        "model_label": model_label,
                        "model_id": folder_label,
                        "sub": sub_tag,
                        "data": data,
                        "key": target_key,
                        "chan_avg_corr": chan_avg_corr,
                        "times": data["times"],
                        "test_loss": test_loss,
                        "dnn_group": folder_label,
                    }
                )

            except Exception as e:
                print(f"  Cannot read {os.path.basename(fpath)}: {e}")

if not all_collected_data:
    print("Error: No valid data was read.")
    exit()

# Group data by subject and by model
data_by_sub = defaultdict(list)
data_by_model = defaultdict(list)
for item in all_collected_data:
    data_by_sub[item["sub"]].append(item)
    data_by_model[item["model_id"]].append(item)

# Determine consistent color mapping for models across all subjects
all_model_ids = sorted(list(data_by_model.keys()))
num_models = len(all_model_ids)
model_color_map = {
    m_id: cmap(i / max(1, num_models - 1)) if num_models > 1 else cmap(0)
    for i, m_id in enumerate(all_model_ids)
}

out_dir = figure_dir
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# Plot 1: Per-Subject Model Comparison Plots (All Channels Combined)
# =============================================================================
for sub_tag in sorted(data_by_sub.keys()):
    sub_items = data_by_sub[sub_tag]
    times = sub_items[0]["times"]

    # Set figsize to 32x18 and DPI to 120 for 4K resolution (3840x2160)
    fig = plt.figure(figsize=(32, 18))
    plt.title(f"Model Comparison - {sub_tag.upper()} ({len(sub_items)} Models)", fontsize=30, pad=20)

    plt.plot([-10, 10], [0, 0], "k--", [0, 0], [10, -10], "k--", label="_nolegend_", linewidth=3)

    for item in sub_items:
        color = model_color_map.get(item["model_id"], cmap(0))
        mean_corr = item["chan_avg_corr"]
        p_len = min(len(times), len(mean_corr))
        plt.plot(times[:p_len], mean_corr[:p_len], color=color, linewidth=3, label=item["model_label"])

    # Subject-specific noise ceiling
    sub_num_match = re.search(r"\d+", sub_tag)
    if sub_num_match:
        sub_id = int(sub_num_match.group())
        nc_low, nc_up, nc_file = load_noise_ceiling(sub_id, data_dir)
        if nc_low is not None:
            plt.fill_between(
                times[: len(nc_low)],
                nc_low,
                nc_up,
                color=color_noise_ceiling,
                alpha=0.3,
                label=f"Noise Ceiling (iters: {os.path.splitext(nc_file)[0]})",
            )
        else:
            print(f"  Info: No noise ceiling cache found for {sub_tag}")

    plt.xlabel("Time (s)", fontsize=24)
    plt.ylabel("Pearson's $r$", fontsize=24)

    xticks = [-0.2, 0, 0.2, 0.4, 0.6, max(times)]
    xlabels = [-0.2, 0, 0.2, 0.4, 0.6, round(max(times), 1)]
    plt.xticks(ticks=xticks, labels=xlabels)
    plt.xlim(left=min(times), right=max(times))
    plt.ylim(bottom=-0.1, top=1)

    plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    plt.tight_layout()

    out_filename = os.path.join(out_dir, f"model_comparison_{sub_tag}.jpg")
    plt.savefig(out_filename, format="jpg", dpi=120)
    plt.close(fig)
    print(f"Saved: {out_filename}")


# =============================================================================
# Plot 2: Overall Cross-Subject Comparison Plot (All Channels Combined)
# =============================================================================
overall_times = all_collected_data[0]["times"]

fig_overall = plt.figure(figsize=(32, 18))
method_str = "Fisher z-Mean" if args.average_method == "fisher_z" else "Arithmetic Mean"
plt.title(f"Model Comparison - Overall Cross-Subject Average ({method_str})", fontsize=30, pad=20)
plt.plot([-10, 10], [0, 0], "k--", [0, 0], [10, -10], "k--", label="_nolegend_", linewidth=3)

avg_func = fisher_z_mean if args.average_method == "fisher_z" else np.mean
summary_records = []

for m_id in all_model_ids:
    items = data_by_model[m_id]
    color = model_color_map.get(m_id, cmap(0))

    # Align time dimension
    min_len = min(len(item["chan_avg_corr"]) for item in items)
    p_len = min(len(overall_times), min_len)
    stacked_corrs = np.array([item["chan_avg_corr"][:p_len] for item in items])

    if args.average_method == "fisher_z":
        mean_curve = fisher_z_mean(stacked_corrs, axis=0)
    else:
        mean_curve = np.mean(stacked_corrs, axis=0)

    # Compute standard error of the mean (SEM) across subjects
    if len(items) > 1:
        sem_curve = np.std(stacked_corrs, axis=0, ddof=1) / np.sqrt(len(items))
    else:
        sem_curve = np.zeros(p_len)

    # Format label
    losses = [it["test_loss"] for it in items if it["test_loss"] is not None]
    label = f"{m_id} (N={len(items)})"
    if losses:
        label += f" | Mean Loss: {np.mean(losses):.6f}"

    plt.plot(overall_times[:p_len], mean_curve[:p_len], color=color, linewidth=3, label=label)

    # Plot SEM shaded region
    if args.show_sem and len(items) > 1:
        plt.fill_between(
            overall_times[:p_len],
            mean_curve[:p_len] - sem_curve[:p_len],
            mean_curve[:p_len] + sem_curve[:p_len],
            color=color,
            alpha=0.15,
        )

    # Compute summary metrics across specific time windows
    times_aligned = np.array(overall_times[:p_len])
    peak_idx = np.argmax(mean_curve)
    peak_r = float(mean_curve[peak_idx])
    peak_time = float(times_aligned[peak_idx])

    mask_0_01 = (times_aligned >= 0.0) & (times_aligned <= 0.1)
    mask_0_02 = (times_aligned >= 0.0) & (times_aligned <= 0.2)
    mask_0_05 = (times_aligned >= 0.0) & (times_aligned <= 0.5)

    r_0_01 = float(avg_func(mean_curve[mask_0_01])) if np.any(mask_0_01) else np.nan
    r_0_02 = float(avg_func(mean_curve[mask_0_02])) if np.any(mask_0_02) else np.nan
    r_0_05 = float(avg_func(mean_curve[mask_0_05])) if np.any(mask_0_05) else np.nan
    r_full = float(avg_func(mean_curve))

    summary_records.append(
        {
            "Model": m_id,
            "Peak r": peak_r,
            "Peak Time (s)": peak_time,
            "0~0.1s r": r_0_01,
            "0~0.2s r": r_0_02,
            "0~0.5s r": r_0_05,
            "Full Window r": r_full,
        }
    )

# Average Noise Ceiling across available subjects
nc_lows = []
nc_ups = []
for sub_tag in sorted(data_by_sub.keys()):
    sub_num_match = re.search(r"\d+", sub_tag)
    if sub_num_match:
        sub_id = int(sub_num_match.group())
        nc_l, nc_u, _ = load_noise_ceiling(sub_id, data_dir)
        if nc_l is not None and nc_u is not None:
            nc_lows.append(nc_l)
            nc_ups.append(nc_u)

if nc_lows and nc_ups:
    min_nc_len = min(min(len(x) for x in nc_lows), min(len(y) for y in nc_ups), len(overall_times))
    nc_lows_aligned = np.array([x[:min_nc_len] for x in nc_lows])
    nc_ups_aligned = np.array([y[:min_nc_len] for y in nc_ups])

    avg_nc_low = fisher_z_mean(nc_lows_aligned, axis=0)
    avg_nc_up = fisher_z_mean(nc_ups_aligned, axis=0)

    nc_label = f"Noise Ceiling (Average across {len(nc_lows)} sub{'s' if len(nc_lows) > 1 else ''})"
    plt.fill_between(
        overall_times[:min_nc_len],
        avg_nc_low,
        avg_nc_up,
        color=color_noise_ceiling,
        alpha=0.3,
        label=nc_label,
    )

plt.xlabel("Time (s)", fontsize=24)
plt.ylabel("Pearson's $r$", fontsize=24)

xticks = [-0.2, 0, 0.2, 0.4, 0.6, max(overall_times)]
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, round(max(overall_times), 1)]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(overall_times), right=max(overall_times))
plt.ylim(bottom=-0.1, top=1)

plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout()

overall_filename = os.path.join(out_dir, "model_comparison_overall.jpg")
plt.savefig(overall_filename, format="jpg", dpi=120)
plt.close(fig_overall)
print(f"Saved: {overall_filename}")

# =============================================================================
# Summary Table: Overall Cross-Subject Performance Summary
# =============================================================================
if summary_records:
    df_summary = pd.DataFrame(summary_records)
    print("\n" + "=" * 90)
    print(f"Overall Cross-Subject Performance Summary ({method_str})")
    print("=" * 90)

    display_df = df_summary.copy()
    display_df["Peak r"] = display_df["Peak r"].map("{:.4f}".format)
    display_df["Peak Time (s)"] = display_df["Peak Time (s)"].map("{:.2f}".format)
    display_df["0~0.1s r"] = display_df["0~0.1s r"].map("{:.4f}".format)
    display_df["0~0.2s r"] = display_df["0~0.2s r"].map("{:.4f}".format)
    display_df["0~0.5s r"] = display_df["0~0.5s r"].map("{:.4f}".format)
    display_df["Full Window r"] = display_df["Full Window r"].map("{:.4f}".format)

    print(display_df.to_string(index=False))
    print("=" * 90 + "\n")

    summary_csv_path = os.path.join(out_dir, "model_comparison_overall_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved: {summary_csv_path}")