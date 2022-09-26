import torch
import torch.nn as nn

from torchinfo import summary


class EfficientnetConv2DT_BBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        input_channels = cfg["model"]["bbox_head"]["input_num_filter"]
        input_channels = [cfg["model"]["bbox_head"]["input_num_filter"]]
        output_channels = cfg["model"]["bbox_head"]["output_num_filter"]
        input_channels.extend(output_channels)

        kernel_size = cfg["model"]["bbox_head"]["kernel_size"]

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
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model.forward(x)

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 256, 96, 96))


class SMP_BBoxHead_with_Softmax(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bbox_w_model = self.get_model(cfg)
        self.bbox_h_model = self.get_model(cfg)

    def get_model(self, cfg):
        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=int(cfg["smp"]["decoder_output_classes"]),
                out_channels=8,
                kernel_size=3,
                padding=1

            ))
        layers.append(nn.ReLU(inplace=True))
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
                out_channels=int(cfg["smp"]["decoder_output_dimension"]),
                kernel_size=3,
                padding=1,

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Softmax(dim=1))

        model = nn.Sequential(*layers)
        return model

    def forward(self, x):
        w_heatmap_focal = self.bbox_w_model.forward(x)
        w_heatmap = torch.argmax(w_heatmap_focal, dim=1).unsqueeze(dim=1)
        h_heatmap_focal = self.bbox_h_model.forward(x)
        h_heatmap = torch.argmax(h_heatmap_focal, dim=1).unsqueeze(dim=1)
        heatmap = torch.cat([w_heatmap, h_heatmap], dim=1)

        return heatmap, w_heatmap_focal, h_heatmap_focal

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 256, 96, 96))


class SMP_BBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bbox_w_model = self.get_model(cfg)
        self.bbox_h_model = self.get_model(cfg)

    def get_model(self, cfg):
        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=int(cfg["smp"]["decoder_output_classes"]),
                out_channels=8,
                kernel_size=3,
                padding=1

            ))
        layers.append(nn.ReLU(inplace=True))
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
        model = nn.Sequential(*layers)
        return model

    def forward(self, x):
        w_heatmap = self.bbox_w_model.forward(x)
        h_heatmap = self.bbox_h_model.forward(x)
        heatmap = torch.cat([w_heatmap, h_heatmap], dim=1)

        return heatmap

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 256, 96, 96))
