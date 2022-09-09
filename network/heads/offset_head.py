import torch.nn as nn

from torchinfo import summary


class EfficientnetConv2DT_OffSetHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        input_channels = [cfg["model"]["offset_head"]["input_num_filter"]]
        output_channels = cfg["model"]["offset_head"]["output_num_filter"]
        input_channels.extend(output_channels)

        kernel_size = cfg["model"]["offset_head"]["kernel_size"]

        layers.append(
            nn.Conv2d(
                in_channels=input_channels[0],
                out_channels=output_channels[0],
                kernel_size=kernel_size[0],
                stride=1,
                padding=1,

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(output_channels[0]))
        layers.append(
            nn.Conv2d(
                in_channels=input_channels[1],
                out_channels=output_channels[1],
                kernel_size=kernel_size[1],
                stride=1,

            ))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model.forward(x)

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 256, 96, 96))


class SMP_OffSetHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=cfg["smp"]["decoder_output_classes"],
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=1 / 2,

            ))
        layers.append(
            nn.UpsamplingBilinear2d(
                scale_factor=cfg["data"]["input_dimension"] / cfg["smp"]["decoder_output_dimension"]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model.forward(x)

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 256, 96, 96))
