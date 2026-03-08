import torch
import torch.nn as nn

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=1700):
        super(CustomAlexNet, self).__init__()
        
        # 嚴謹的維度計算：確保每個通道獲得均等的特徵數量 (例如 1700 / 17 = 100)
        self.features_per_channel = num_classes // 17
        
        # 卷積特徵提取層
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
        
        # ==========================================
        # 區塊 1：對應早期視覺處理
        # ==========================================
        self.classifier_block1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True)
        )

        # 分支 1：枕葉 (Occipital) - 負責 O1, Oz, O2 (3 個通道)
        self.occipital_head = nn.Linear(4096, self.features_per_channel * 3)

        # ==========================================
        # 區塊 2：對應視覺注意力分配
        # ==========================================
        self.classifier_block2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

        # 分支 2：頂枕葉 (Parieto-Occipital) - 負責 PO7, PO3, POz, PO4, PO8 (5 個通道)
        self.parieto_occipital_head = nn.Linear(4096, self.features_per_channel * 5)

        # ==========================================
        # 區塊 3：對應視覺工作記憶
        # ==========================================
        # 為了與前兩層保持一致的神經元容量，這裡重新定義 block3 為隱藏層
        self.classifier_block3 = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

        # 分支 3：頂葉 (Parietal) - 負責 Pz, P3, P7, P4, P8, P1, P5, P6, P2 (9 個通道)
        self.parietal_head = nn.Linear(4096, self.features_per_channel * 9)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平操作，此時維度為 (Batch_Size, 256 * 6 * 6)

        # ---------------------------------------------------------
        # 階段一：提取第一層分類器特徵並分流至枕葉
        # ---------------------------------------------------------
        out1 = self.classifier_block1(x)
        pred_occ = self.occipital_head(out1)
        
        # ---------------------------------------------------------
        # 階段二：繼續深入網路並分流至頂枕葉
        # ---------------------------------------------------------
        out2 = self.classifier_block2(out1)
        pred_po = self.parieto_occipital_head(out2)
        
        # ---------------------------------------------------------
        # 階段三：通過最後的神經網路層並分流至頂葉
        # ---------------------------------------------------------
        out3 = self.classifier_block3(out2)
        pred_p = self.parietal_head(out3)
        
        # =========================================================
        # 重組 Output Tensor (張量拼接機制)
        # 原始通道: ['Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2']
        # =========================================================
        batch_size = x.size(0)
        
        # 1. 調整維度為 (batch_size, 通道數, 每個通道的特徵數)
        pred_occ = pred_occ.view(batch_size, 3, self.features_per_channel)
        pred_po = pred_po.view(batch_size, 5, self.features_per_channel)
        pred_p = pred_p.view(batch_size, 9, self.features_per_channel)
        
        # 2. 建立空張量以放置最終結果 (避免 In-place 操作破壞梯度)
        final_output = torch.empty(batch_size, 17, self.features_per_channel, device=x.device, dtype=x.dtype)
        
        # 3. 根據原始通道索引精準填入預測結果
        # 枕葉位於索引 3, 4, 5
        final_output[:, 3:6, :] = pred_occ
        
        # 頂枕葉位於索引 10, 11, 12, 13, 14
        final_output[:, 10:15, :] = pred_po
        
        # 頂葉散佈在前後位置，需分段填入：
        final_output[:, 0:3, :] = pred_p[:, 0:3, :]   # 對應 Pz, P3, P7 (索引 0~2)
        final_output[:, 6:10, :] = pred_p[:, 3:7, :]  # 對應 P4, P8, P1, P5 (索引 6~9)
        final_output[:, 15:17, :] = pred_p[:, 7:9, :] # 對應 P6, P2 (索引 15~16)
        
        # 4. 攤平回 (batch_size, out_features) 準備計算 MSE Loss
        final_output = final_output.view(batch_size, -1)
        
        return final_output

# 測試模型輸出與參數狀態
if __name__ == "__main__":
    model = CustomAlexNet(num_classes=1700)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"輸出張量形狀: {output.shape}")