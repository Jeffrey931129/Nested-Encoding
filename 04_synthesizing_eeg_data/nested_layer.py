import torch.nn as nn

class NestedOutputLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(NestedOutputLayer, self).__init__()
        
        # 定義中間層的維度，您可以根據需求調整
        hidden_dim_1 = 2048
        hidden_dim_2 = 1024
        
        # Level 1: Fast Neurons (高頻更新，最接近輸入) [cite: 41]
        self.fast_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim_1),
            nn.ReLU(),
        )
        
        # Level 2: Mid Frequency Neurons (中頻更新) [cite: 2]
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
        )
        
        # Level 3: Slow Neurons (低頻更新，最接近輸出) [cite: 2]
        self.slow_layer = nn.Linear(hidden_dim_2, out_features)

    def forward(self, x):
        # 巢狀結構：數據依次通過 Fast -> Mid -> Slow [cite: 293]
        x = self.fast_layer(x)
        x = self.mid_layer(x)
        x = self.slow_layer(x)
        return x