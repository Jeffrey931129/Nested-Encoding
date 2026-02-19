import os
import numpy as np
import matplotlib.pyplot as plt
import mne  # 這是專門畫腦波圖的套件

# ================= 設定區 =================
# 請依照你的資料夾結構修改這裡
project_dir = '../project_directory' 
sub = 1
dnn = 'alexnet'
# =========================================

print("正在讀取數據...")

# 1. 讀取你算好的統計數據 (correlation_stats.npy)
data_path = os.path.join(project_dir, 'results', 'stats', 'correlation',
                         'encoding-linearizing', 'subjects-within', 
                         f'dnn-{dnn}', 'pretrained-True', 'layers-all', 
                         'n_components-01000', 'correlation_stats.npy')

if not os.path.exists(data_path):
    print(f"錯誤：找不到檔案 {data_path}")
    print("請確認路徑或參數設定是否正確。")
    exit()

results = np.load(data_path, allow_pickle=True).item()

# 2. 準備數據
avg_corr = np.mean(results['correlation']['all_layers'], axis=0)
times = results['times']
ch_names = results['ch_names']

# 3. 找出預測最準的「巔峰時刻」 (Peak Time)
global_mean_over_time = np.mean(avg_corr, axis=0)
peak_idx = np.argmax(global_mean_over_time)
peak_time = times[peak_idx]

print(f"發現最佳預測時間點：{peak_time:.3f} 秒 (第 {peak_idx} 個時間點)")

# 4. 設定 MNE 的頭皮資訊
montage = mne.channels.make_standard_montage('easycap-M1') 
info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
info.set_montage(montage)

# 5. 畫圖！
fig, ax = plt.subplots(figsize=(6, 6))

# --- 修正重點：使用 vlim=(min, max) 取代 vmin, vmax ---
im, _ = mne.viz.plot_topomap(
    avg_corr[:, peak_idx], 
    info, 
    axes=ax, 
    show=False,
    names=ch_names, 
    cmap='RdBu_r', 
    vlim=(0, 0.6)  # 這裡是新版的寫法
)

# 加個標題和 Colorbar
plt.title(f"Brain Encoding Performance\nModel: {dnn} | Time: {peak_time:.3f}s", fontsize=15)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Correlation (r)")

print("繪圖完成！請看視窗。")
plt.show()