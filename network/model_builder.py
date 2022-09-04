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
from network.roi_classifier.clip_model import CLIPModel


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
        self.clip_model = CLIPModel(cfg)
        self.cfg = cfg

    def forward(self, batch, train_set=True):
        image = batch["image"].to(self.cfg["device"])
        image_path = batch["image_path"]
        image_id = batch['image_id'].to(self.cfg["device"])
        x = self.encoder_model(image)
        # return x
        x = self.decoder_model(x)
        # return x
        output_heatmap = self.heatmap_head(x)
        output_offset = self.offset_head(x)
        output_bbox = self.bbox_head(x)
        with torch.no_grad():
            output_roi = self.roi_model(output_heatmap, output_bbox, output_offset, image_id)
            output_clip = self.clip_model(image_path, output_roi, train_set=train_set)
            output_clip = torch.tensor(output_clip)

        return output_heatmap, output_bbox, output_offset, output_roi

    def print_details(self):
        batch_size = 32
        summary(self.encoder_model, input_size=(batch_size, 3, 300, 300))
        # summary(self, input_size=(batch_size, 512, 12, 12))
