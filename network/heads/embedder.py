import torch.nn as nn
import torch
from torchinfo import summary


class SMP_Embedder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(3))
        layers.append(nn.MaxPool2d(kernel_size=16,
                                   stride=4,
                                   ))

        layers.append(
            nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(3))
        layers.append(nn.AvgPool2d(kernel_size=16,
                                   stride=4,
                                   padding=1
                                   ))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(768, 512))

        self.model = nn.Sequential(*layers)

    def forward(self, masked_heatmaps_features):
        return self.model(masked_heatmaps_features)

    def print_details(self):
        pass
