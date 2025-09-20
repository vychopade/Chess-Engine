# engine/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessEvaluator(nn.Module):
    def __init__(self, input_planes: int = 18):
        super().__init__()
        self.conv1 = nn.Conv2d(input_planes, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.head = nn.Sequential(
            nn.Flatten(),                    # 256 * 8 * 8
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x: [B, input_planes, 8, 8]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x)
        return torch.tanh(x)  # scale between -1 and 1
