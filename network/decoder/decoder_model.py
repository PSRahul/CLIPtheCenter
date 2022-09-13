import torch.nn as nn

from torchinfo import summary


class DecoderConvTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        input_channels = [cfg["model"]["decoder"]["input_num_filter"]]
        output_channels = cfg["model"]["decoder"]["output_num_filter"]
        input_channels.extend(output_channels)

        for i in range(cfg["model"]["decoder"]["num_layers"]):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=int(input_channels[i]),
                    out_channels=int(output_channels[i]),
                    kernel_size=3,
                    stride=3,

                ))
            layers.append(nn.BatchNorm2d(int(output_channels[i])))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Upsample((cfg["data"]["input_dimension"], cfg["data"]["input_dimension"])))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model.forward(x)

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 512, 10, 10))
