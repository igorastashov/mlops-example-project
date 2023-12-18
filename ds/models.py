import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            ),  # 16 x 224 x 224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 32 x 224 x 224
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=28),  # 32 x 8 x 8
            nn.Flatten(),  # 32*8*8 = 2048
            nn.Linear(2048, 150),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
