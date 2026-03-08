import torch
import torch.nn as nn

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=1700):
        super(CustomAlexNet, self).__init__()
        
        # 特徵提取層 (引入 Batch Normalization 與 Grouped Convolutions)
        self.features = nn.Sequential(
            # Conv1: 維持原始感受野，加入 BN 穩定初始特徵分佈
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 引入 groups=2，將卷積核切分為兩組，減少一半參數
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2, groups=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 負責跨通道資訊融合，不使用分組卷積
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Conv4: 再次使用 groups=2 降低維度災難帶來的參數膨脹
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv5: 使用 groups=2
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 全局平均池化 (Global Average Pooling)
        # 將空間維度從 N x N 強制壓縮為 1 x 1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        classifier_layers = []
        hidden_dim = 512
        num_hidden_layers = 9 # 前 9 層為隱藏層，第 10 層為輸出
        
        # 第 1 層 (Input: 256 -> Hidden: 512)
        classifier_layers.extend([
            nn.Linear(256, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1) # 建議降低 Dropout 機率
        ])
        
        # 第 2 到第 9 層 (Hidden: 512 -> Hidden: 512)
        for _ in range(num_hidden_layers - 1):
            classifier_layers.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1) 
            ])
            
        # 第 10 層 (輸出層, Hidden: 512 -> Output: 1700)
        classifier_layers.append(nn.Linear(hidden_dim, num_classes, bias=True))
        
        self.classifier = nn.Sequential(*classifier_layers)

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