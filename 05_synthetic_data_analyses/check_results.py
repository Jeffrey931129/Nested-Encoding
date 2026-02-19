import numpy as np
import os
import argparse

# 設定參數 (必須跟您跑 correlation.py 時的設定一樣！)
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--n_components', default=1000, type=int) # 注意確認此數值
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

# 建構檔案路徑 (這段邏輯是從 correlation.py 複製過來的)
save_dir = os.path.join(args.project_dir, 'results', 'sub-'+format(args.sub,'02'), 
    'correlation', 'encoding-linearizing', 'subjects-within', 
    'dnn-'+args.dnn, 'pretrained-True', 'layers-all', 
    'n_components-'+format(args.n_components,'05'))

file_path = r"C:\Users\Jeffrey\Downloads\eeg_encoding\project_directory\results\sub-01\correlation\encoding-end_to_end\dnn-alexnet\modeled_time_points-all\pretrained-True\lr-1e-05__wd-0e+00__bs-064\correlation.npy"

# 檢查檔案是否存在
if not os.path.exists(file_path):
    print(f"錯誤：找不到檔案！請檢查路徑：\n{file_path}")
    print("提示：您的 n_components 是不是設錯了？")
    exit()

# 讀取數據
print(f"正在讀取：{file_path}")
data = np.load(file_path, allow_pickle=True).item()
correlations = data['correlation'] # 這是一個字典，key 是層名稱
times = data['times']
ch_names = data['ch_names']

print("\n" + "="*40)
print(f"   受試者 {args.sub} - {args.dnn} 預測結果快報")
print("="*40)

# 遍歷每一層 (Layer) 的結果
for layer_name, corr_matrix in correlations.items():
    # corr_matrix 形狀通常是 (Channels x TimePoints)
    
    # 1. 計算整體平均相關係數
    avg_corr = np.mean(corr_matrix)
    
    # 2. 找出最強的相關係數
    max_corr = np.max(corr_matrix)
    
    # 3. 找出最強發生的位置
    max_idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
    best_ch = ch_names[max_idx[0]]
    best_time = times[max_idx[1]]

    print(f"\n層級: {layer_name}")
    print(f"  - 平均相關係數: {avg_corr:.4f}")
    print(f"  - 最高相關係數: {max_corr:.4f}")
    print(f"  - 最佳預測點: 通道 [{best_ch}] 在 {best_time:.3f} 秒")

print("\n" + "="*40)
print("解讀指南：")
print("1. 相關係數範圍 -1 到 1。")
print("2. 在 EEG Encoding 研究中，平均值通常在 0.1 ~ 0.3 就算不錯。")
print("3. 最高值如果能超過 0.5 代表某些特定腦區預測非常準確。")
print("="*40)