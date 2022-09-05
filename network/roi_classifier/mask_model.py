import torch.nn as nn
import torch
from torchinfo import summary
from network.models.EfficientnetConv2DT.utils import get_bounding_box_prediction
import clip
from PIL import Image
import os


class MaskModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, output_roi):
        mask = torch.zeros(
            (output_roi.shape[0], self.cfg["heatmap"]["output_dimension"], self.cfg["heatmap"]["output_dimension"]))
        output_roi = output_roi[:, 1:5].int()
        for i in range(output_roi.shape[0]):
            bbox = output_roi[i]

            mask[i, bbox[1]:bbox[1] + bbox[3] + 1, bbox[0]:bbox[0] + bbox[2] + 1] = 1
        return mask

    def print_details(self):
        pass
