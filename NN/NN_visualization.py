import torch
from torchviz import make_dot
from torchinfo import summary

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_channels=64, n_classes=3, dropout=0.4):
        super().__init__()

        # Temporal filtering (interpretable)
        self.temporal = nn.Conv1d(n_channels, 32, kernel_size=25,
            padding=12, bias=False)
        self.bn1 = nn.BatchNorm1d(32)

        # Spatial filtering (channel mixing)
        self.spatial = nn.Conv1d(32, 64,
            kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)

        # Feature extraction
        self.conv = nn.Conv1d(64, 128,
            kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.temporal(x)))
        x = F.relu(self.bn2(self.spatial(x)))
        x = F.relu(self.bn3(self.conv(x)))

        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)

        return self.fc(x)

model = CNN(n_channels=64, n_classes=3)

x = torch.randn(1, 64, 300)  # example input
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("model_graph", format="png")

model = CNN(n_channels=64, n_classes=3)

summary(model, input_size=(1, 64, 300))