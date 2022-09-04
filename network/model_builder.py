import torch.nn as nn
from torchinfo import summary
import torch
from network.decoder.decoder_model import DecoderConvTModel
from network.heads.bbox_head import BBoxHead
from network.heads.heatmap_head import HeatMapHead
from network.heads.offset_head import OffSetHead
from network.encoder.resnet18 import ResNet18Model
from network.encoder.efficientnetb3 import EfficientNetB3Model
from network.encoder.efficientnetb2 import EfficientNetB2Model
from network.encoder.efficientnetb0 import EfficientNetB0Model
from network.encoder.efficientnetb1 import EfficientNetB1Model
from network.encoder.efficientnetb4 import EfficientNetB4Model
from network.roi_classifier.roi_model import RoIModel


class DetectionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        encoder_model_name = globals()[cfg["model"]["encoder"]["encoder_name"]]
        self.encoder_model = encoder_model_name(cfg)
        self.decoder_model = DecoderConvTModel(cfg)
        self.heatmap_head = HeatMapHead(cfg)
        self.offset_head = OffSetHead(cfg)
        self.bbox_head = BBoxHead(cfg)
        self.roi_model = RoIModel(cfg)
        self.cfg = cfg

    def forward(self, image, image_id=torch.ones((32), device="cuda")):
        x = self.encoder_model(image)
        # return x
        x = self.decoder_model(x)
        # return x
        output_heatmap = self.heatmap_head(x)
        output_offset = self.offset_head(x)
        output_bbox = self.bbox_head(x)
        output_roi = self.roi_model(output_heatmap, output_bbox, output_offset, image_id)
        return output_heatmap, output_bbox, output_offset, output_roi

    def print_details(self):
        batch_size = 32
        summary(self, input_size=(batch_size, 3, 300, 300))
        # summary(self, input_size=(batch_size, 512, 12, 12))
