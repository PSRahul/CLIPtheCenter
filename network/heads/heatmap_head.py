import copy

import torch
import torch.nn as nn

from torchinfo import summary


class SMP_HeatMapHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=int(cfg["smp"]["decoder_output_classes"]),
                out_channels=1,
                kernel_size=3,
                padding=1

            ))
        layers.append(nn.ReLU(inplace=True))
        """
        layers.append(nn.BatchNorm2d(8))
        layers.append(
            nn.Conv2d(
                in_channels=8,
                out_channels=4,
                kernel_size=3,
                padding=1,

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(4))
        layers.append(
            nn.Conv2d(
                in_channels=4,
                out_channels=1,
                kernel_size=3,
                padding=1,

            ))
        layers.append(nn.ReLU(inplace=True))
        """
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model.forward(x)
        x_like = torch.zeros_like(x)
        y = torch.max(x.view(x.shape[0], -1), dim=1)[0]
        for i in range(x.shape[0]):
            x_like[i] = y[i]
        return x / x_like

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 256, 96, 96))
