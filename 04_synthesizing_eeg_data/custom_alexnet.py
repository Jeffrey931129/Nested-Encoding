import torch
import torch.nn as nn

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=1700):
        super(CustomAlexNet, self).__init__()
        
        # 特徵提取層 (引入 Batch Normalization 與 Grouped Convolutions)
        self.features = nn.Sequential(
            # Conv1: 維持大 Kernel 負責初始下採樣，這部分較難無損壓縮
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 將 5x5 分解為兩層 3x3，並使用 MobileNet 的 Depthwise Separable 概念
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 原本的參數怪獸，改用 Depthwise Separable Convolution
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, groups=192, bias=False),
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Conv4: 改用 Bottleneck 設計 (384 -> 128 -> 128 -> 256)
            nn.Conv2d(384, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=4, bias=False), # 進一步提升 groups 至 4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv5: 再次使用 Depthwise Separable Convolution
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 全局平均池化 (Global Average Pooling)
        # 將空間維度從 N x N 強制壓縮為 1 x 1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 重構的分類器 (拓撲結構平滑化)
        self.classifier = nn.Sequential(
            # 輸入特徵維度從 9216 驟降至 256
            nn.Linear(in_features=256, out_features=1024, bias=False),
            nn.BatchNorm1d(1024), # 全連接層也使用 BN 穩定梯度
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),    # 降低 Dropout 強度，避免特徵流失
            
            nn.Linear(in_features=1024, out_features=1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            # 輸出層直接映射至目標類別數
            nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平操作，此時維度為 (Batch_Size, 256)
        x = self.classifier(x)
        return x

# 測試模型輸出與參數狀態
if __name__ == "__main__":
    model = CustomAlexNet(num_classes=1700)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"輸出張量形狀: {output.shape}")