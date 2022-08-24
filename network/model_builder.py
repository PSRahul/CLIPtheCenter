import torch.nn as nn
from torchinfo import summary

from network.decoder.decoder_model import DecoderConvTModel
from network.heads.bbox_head import BBoxHead
from network.heads.heatmap_head import HeatMapHead
from network.heads.offset_head import OffSetHead
from network.encoder.resnet18 import ResNet18Model
from network.encoder.efficientnetb3 import EfficientNetB3Model
from network.encoder.efficientnetb2 import EfficientNetB2Model


class DetectionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        encoder_model_name = globals()[cfg["model"]["encoder"]["encoder_name"]]
        self.encoder_model = encoder_model_name(cfg)
        self.decoder_model = DecoderConvTModel(cfg)
        self.heatmap_head = HeatMapHead(cfg)
        self.offset_head = OffSetHead(cfg)
        self.bbox_head = BBoxHead(cfg)

    def forward(self, x):
        x = self.encoder_model(x)
        return x
        # x = self.decoder_model(x)
        # output_heatmap = self.heatmap_head(x)
        # output_offset = self.offset_head(x)
        # output_bbox = self.bbox_head(x)
        # return output_heatmap, output_offset, output_bbox

    def print_details(self):
        batch_size = 32
        summary(self, input_size=(batch_size, 3, 384, 384))
        # summary(self, input_size=(batch_size, 512, 12, 12))
