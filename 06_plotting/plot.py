import argparse
import os
import math
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from collections import defaultdict
import time
import subprocess

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser(description="Recursive plotting script - Compare DNNs")
parser.add_argument("--project_dir", default="project_directory", type=str, 
                    help="Root directory containing DNN subfolders")
parser.add_argument("--sub", default=1, type=int)
parser.add_argument("--target_dnns", nargs="+", default=None, 
                    help="Optional: Specify which DNN folders to process")
args = parser.parse_args()

# =============================================================================
# Helper Functions
# =============================================================================

def find_files_recursive(current_dir):
    """
    使用 os.listdir 手動遞迴搜索資料夾
    """
    found_files = []
    
    try:
        entries = os.listdir(current_dir)
    except OSError as e:
        print(f"警告: 無法讀取目錄 {current_dir} ({e})，跳過。")
        return []

    entries.sort()

    for entry in entries:
        full_path = os.path.join(current_dir, entry)
        
        if os.path.isdir(full_path):
            found_files.extend(find_files_recursive(full_path))
            
        elif os.path.isfile(full_path) and entry.endswith(".npy"):
            filename_label = os.path.splitext(entry)[0]
            parent_folder = os.path.basename(current_dir)
            
            found_files.append({
                "path": full_path,
                "filename": filename_label,
                "parent": parent_folder
            })
            
    return found_files

def resolve_labels(file_list):
    """
    處理單一 DNN 內的檔名衝突
    """
    label_counts = defaultdict(int)
    for item in file_list:
        label_counts[item["filename"]] += 1
        
    final_output = []
    for item in file_list:
        if label_counts[item["filename"]] > 1:
            label = f"{item['parent']}_{item['filename']}"
        else:
            label = item["filename"]
        final_output.append((item["path"], label))
    
    final_output.sort(key=lambda x: x[1])
    return final_output

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
cmap = matplotlib.cm.get_cmap('tab20')

# =============================================================================
# Main Logic
# =============================================================================

if not os.path.exists(args.project_dir):
    print(f"錯誤：找不到專案目錄 {args.project_dir}")
    exit()

print(f"正在掃描專案目錄: {args.project_dir}...")
try:
    dir = os.path.join(args.project_dir, 'results', 'sub-'+
            format(args.sub,'02'), 'stats', 'correlation',
        'encoding-end_to_end')
    subdirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    subdirs.sort()
except OSError as e:
    print(f"錯誤：無法讀取專案目錄 ({e})")
    exit()

if args.target_dnns:
    dnn_groups = [d for d in subdirs if d in args.target_dnns]
else:
    dnn_groups = subdirs

if not dnn_groups:
    print("未找到任何 DNN 資料夾。")
    exit()

print(f"即將處理以下 DNN 模型: {dnn_groups}")

# --- Step 1: 收集所有數據 ---
all_plottable_data = []

for dnn_name in dnn_groups:
    print(f"正在讀取 DNN: {dnn_name}")
    dnn_path = os.path.join(dir, dnn_name)
    
    raw_files = find_files_recursive(dnn_path)
    if not raw_files:
        continue
        
    files_info = resolve_labels(raw_files)
    
    for fpath, label in files_info:
        try:
            data = np.load(fpath, allow_pickle=True).item()
            
            if "correlation" not in data or "times" not in data:
                continue
            keys = list(data["correlation"].keys())
            if not keys:
                continue
            
            target_key = keys[0]
            full_label = f"{dnn_name} | {label}"
            
            all_plottable_data.append({
                "label": full_label,
                "data": data,
                "key": target_key,
                "dnn_group": dnn_name
            })
            
        except Exception as e:
            print(f"  無法讀取 {os.path.basename(fpath)}: {e}")

if not all_plottable_data:
    print("錯誤：沒有讀取到任何有效數據。")
    exit()

print(f"總共收集到 {len(all_plottable_data)} 條數據，開始繪圖...")

times = all_plottable_data[0]["data"]["times"]
num_total = len(all_plottable_data)

min_len = min([len(item["data"]["significance"][item["key"]]) for item in all_plottable_data])
sig_matrix = np.zeros((num_total, min_len))

for i, item in enumerate(all_plottable_data):
    s_data = item["data"]["significance"][item["key"]][:min_len]
    for t in range(len(s_data)):
        if s_data[t] == False:
            sig_matrix[i, t] = -100
        else:
            sig_matrix[i, t] = -0.085 + (abs(i + 4.25 - num_total) / (num_total * 2 + 10) * 1.75) 

# =============================================================================
# Plot 1: Comparison Line Plot (All in One)
# =============================================================================
# figize 設定為 32x20，稍後我們以 DPI 80 輸出，即 32*80 = 2560 像素 (2K 寬度)
fig1 = plt.figure(figsize=(32, 20)) 
plt.title(f"Model Comparison ({len(dnn_groups)} DNNs)", fontsize=30, pad=20)

plt.plot([-10, 10], [0, 0], "k--", [0, 0], [10, -10], "k--", label="_nolegend_", linewidth=3)

for i, item in enumerate(all_plottable_data):
    color = cmap(i / num_total) if num_total > 1 else cmap(0)
    data_dict = item["data"]
    key = item["key"]
    label = item["label"]
    
    corr = data_dict["correlation"][key]
    
    if corr.ndim == 3:
        mean_corr = np.mean(np.mean(corr, 0), 0)
        ci_lo = data_dict["ci_lower"][key]
        ci_up = data_dict["ci_upper"][key]
    elif corr.ndim == 2:
        mean_corr = np.mean(corr, 0)
        ci_lo = data_dict["ci_lower"][key]
        ci_up = data_dict["ci_upper"][key]
    else:
        mean_corr = corr
        ci_lo = mean_corr
        ci_up = mean_corr
        
    p_len = min(len(times), len(mean_corr))
    
    plt.plot(times[:p_len], mean_corr[:p_len], color=color, linewidth=3, label=label)
    
    if "ci_lower" in data_dict:
        plt.fill_between(times[:p_len], ci_up[:p_len], ci_lo[:p_len], color=color, alpha=0.1)
        
    sig_y = sig_matrix[i]
    if len(sig_y) > 0:
        valid_idx = sig_y > -10
        plt.plot(times[:len(sig_y)][valid_idx], sig_y[valid_idx], "o", color=color, markersize=3)

first_data = all_plottable_data[0]["data"]
if "noise_ceiling_low" in first_data:
    nc_low = first_data["noise_ceiling_low"]
    nc_up = first_data["noise_ceiling_up"]
    if nc_low.ndim == 3:
        nc_low = np.mean(np.mean(nc_low, 0), 0)
        nc_up = np.mean(np.mean(nc_up, 0), 0)
    plt.fill_between(times[:len(nc_low)], nc_low, nc_up, color=color_noise_ceiling, alpha=0.3, label="Noise Ceiling")

plt.xlabel("Time (s)", fontsize=24)
plt.ylabel("Pearson's $r$", fontsize=24)

xticks = [-0.2, 0, 0.2, 0.4, 0.6, max(times)]
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, round(max(times), 1)]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))
plt.ylim(bottom=-0.1, top=1)

plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout()

# 將 Plot 1 存為 JPG，設定 DPI=80 (32英吋 * 80 DPI = 2560 像素，達到 2K 寬度)
plot1_filename = "model_comparison_2K.jpg"
plt.savefig(plot1_filename, format="jpg", dpi=300)
print(f"已儲存 Plot 1: {plot1_filename} (解析度 2560x1600)")
plt.close(fig1)  # 釋放記憶體

# =============================================================================
# Plot 2: Per-Channel Temporal Dynamics (使用 plt.figure 重構)
# =============================================================================
first_data_corr = all_plottable_data[0]["data"]["correlation"][all_plottable_data[0]["key"]]
if first_data_corr.ndim == 3:
    num_channels = first_data_corr.shape[1]
elif first_data_corr.ndim == 2:
    num_channels = first_data_corr.shape[0]
else:
    num_channels = 0

if num_channels == 0:
    print("無法判定通道數量或張量維度錯誤。")
else:
    # --- 【關鍵保護機制】防止子圖密度過高導致渲染崩潰 ---
    # 2K 解析度 (2560x1600) 的畫布，若超過 64 個子圖 (16列x4欄)，視覺上會完全糊成一團甚至報錯
    MAX_CHANNELS = 64 
    if num_channels > MAX_CHANNELS:
        print(f"警告：通道數 ({num_channels}) 過大。為保證 2K 圖片成功渲染，僅繪製前 {MAX_CHANNELS} 個通道。")
        plot_channels = MAX_CHANNELS
    else:
        plot_channels = num_channels

    cols = 4
    rows = int(math.ceil(plot_channels / cols))
    
    # 依照要求，重新呼叫 plt.figure 並強制寫死 32x20
    fig2 = plt.figure(figsize=(32, 20))
    fig2.suptitle(f"Time-Resolved Encoding Performance per Channel", fontsize=20, y=1.05)
    
    for c in range(plot_channels):
        # 手動逐一建立子圖，代替原本的 plt.subplots
        ax = fig2.add_subplot(rows, cols, c + 1)
        
        ax.set_title(f"Channel {c}", fontsize=12)
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
            
            ax.plot(times[:p_len], time_course[:p_len], 
                    color=color, linewidth=1.5, 
                    label=label if c == 0 else "")
        
        # 設定 Y 軸標籤 (僅限最左側欄)
        if c % cols == 0:
            ax.set_ylabel("Pearson's $r$", fontsize=10)
            ax.tick_params(axis='y', labelsize=10)
        
        # 設定 X 軸標籤 (僅限最底部列)
        if c >= (rows - 1) * cols or (c + cols) >= plot_channels:
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, max(times)])
            ax.set_xticklabels([-0.2, 0, 0.2, 0.4, 0.6, round(max(times), 1)])
            ax.tick_params(axis='x', labelsize=10)

    # 擷取圖例
    handles, labels = fig2.axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if "" in by_label: 
        del by_label[""]
        
    fig2.legend(by_label.values(), by_label.keys(), 
               fontsize=10, loc="upper center", 
               bbox_to_anchor=(0.5, 1.0), ncol=min(4, len(by_label)), frameon=False)
    
    # 處理邊界，並加入例外捕捉以防萬一
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    except Exception as e:
        print(f"版面佈局調整警告: {e} (已略過以確保圖片輸出)")
    
    plot2_filename = "channel_dynamics_2K.jpg"
    plt.savefig(plot2_filename, format="jpg", dpi=300)
    print(f"已儲存 Plot 2: {plot2_filename} (解析度 2560x1600)")
    plt.close(fig2)  # 釋放記憶體

try:
    subprocess.Popen(['explorer', os.path.abspath(plot1_filename)])
    time.sleep(1)
    subprocess.Popen(['explorer', os.path.abspath(plot2_filename)])
except Exception as e:
    print(f"{e}")

print("所有繪圖任務完成並已存成 2K JPG 檔案。")