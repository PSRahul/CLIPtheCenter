import torch.nn as nn
import torch
from torchinfo import summary
from network.models.EfficientnetConv2DT.utils import get_bounding_box_prediction
import clip


class RoIModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, output_heatmap, output_bbox, output_offset, image_id):
        with torch.no_grad():
            detections = get_bounding_box_prediction(self.cfg,
                                                     output_heatmap.detach(),
                                                     output_offset.detach(),
                                                     output_bbox.detach(),
                                                     image_id)

        return detections

    def print_details(self):
        pass
