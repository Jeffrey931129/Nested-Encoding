import torch
import torch.nn as nn


class AlexEEGNet(nn.Module):
    """
    AlexEEGNet: A hybrid neural network combining CNN (AlexNet backbone),
    fully-connected projection layers, and an LSTM sequence model to decode
    visual stimuli into EEG responses across channels and time points.
    """

    def __init__(self, num_channels=17, time_points=100):
        super(AlexEEGNet, self).__init__()
        self.num_channels = num_channels
        self.time_points = time_points
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
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 9, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        self.channel_decoder = nn.Linear(512, num_channels)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        vis_embedding = self.classifier(x)
        rnn_input = vis_embedding.unsqueeze(1).repeat(1, self.time_points, 1)
        lstm_out, _ = self.lstm(rnn_input)
        out = self.channel_decoder(lstm_out)
        out = out.permute(0, 2, 1)
        out = torch.flatten(out, start_dim=1)
        return out


class AlexNetLite(nn.Module):
    """A parameter-reduced AlexNet baseline model for directly decoding visual stimuli into EEG responses."""

    def __init__(self, num_channels=17, time_points=100):
        super(AlexNetLite, self).__init__()
        self.num_channels = num_channels
        self.time_points = time_points
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 96, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 36, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_channels * time_points),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
