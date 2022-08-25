import os.path
import albumentations as A

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss.bbox_loss import calculate_bbox_loss
from loss.heatmap_loss import calculate_heatmap_loss
from loss.offset_loss import calculate_offset_loss
from trainer.trainer_visualisation import plot_heatmaps
import torch.nn as nn
from evaluation.eval_utils import _gather_output_feature, _transpose_and_gather_output_feature
import numpy as np


class VisModule():
    def __init__(self, cfg):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = "cpu"
        self.cfg = cfg
        
    def find_heatmap_peaks(self, output_heatmap, ):
        kernel = self.cfg["evaluation"]["heatmap_pooling_kernel"]
        pad = (kernel - 1) // 2

        output_heatmap_max = nn.functional.max_pool2d(
            output_heatmap, (kernel, kernel), stride=1, padding=pad)
        keep = (output_heatmap_max == output_heatmap).float()
        return output_heatmap * keep

    def get_topk_indexes(self, output_heatmap):
        k = self.cfg["evaluation"]["topk_k"]
        batch, num_classes, height, width = output_heatmap.size()
        # Returns the maximum K values and their location per class [N,C,H*W]
        topk_heatmap_value_per_class, topk_heatmap_index_per_class = torch.topk(
            output_heatmap.view(batch, num_classes, -1), k)
        # The coordinates of the largest K points on each N and each C, size is [N,C,K], storing the vertical and horizontal coordinates

        topk_heatmap_index_row_per_class = (topk_heatmap_index_per_class / width).int().float()
        topk_heatmap_index_column_per_class = (topk_heatmap_index_per_class % width).int().float()

        topk_heatmap_value, topk_heatmap_index = torch.topk(
            topk_heatmap_value_per_class.view(batch, -1), k)
        # Across second dimension, divide by the number of class. There are total of N filters. We choose top k scores per class. We so divide N by k to get the class correspondence.
        topk_classes = (topk_heatmap_index / k).int()

        topk_heatmap_index = _gather_output_feature(
            topk_heatmap_index_per_class.view(batch, -1, 1), topk_heatmap_index).view(batch, k)
        topk_heatmap_index_row = _gather_output_feature(topk_heatmap_index_row_per_class.view(batch, -1, 1),
                                                        topk_heatmap_index).view(batch, k)
        topk_heatmap_index_column = _gather_output_feature(topk_heatmap_index_column_per_class.view(batch, -1, 1),
                                                           topk_heatmap_index).view(batch, k)

        return topk_heatmap_value, topk_heatmap_index, topk_classes, topk_heatmap_index_row, topk_heatmap_index_column

    def get_topk_indexes_class_agnostic(self, output_heatmap):
        k = self.cfg["evaluation"]["topk_k"]
        output_heatmap = output_heatmap.squeeze(dim=1).to(self.device)

        batch, height, width = output_heatmap.size()
        topk_heatmap_value, topk_heatmap_index = torch.topk(
            output_heatmap.view(batch, -1), k)
        topk_heatmap_index_row = (topk_heatmap_index / width).int().float()
        topk_heatmap_index_column = (topk_heatmap_index % width).int().float()

        topk_classes = torch.zeros((batch, k), device=self.device)
        # Across second dimension, divide by the number of class. There are total of N filters. We choose top k scores per class. We so divide N by k to get the class correspondence.
        return topk_heatmap_value, topk_heatmap_index, topk_classes, topk_heatmap_index_row, topk_heatmap_index_column

    def process_output_heatmaps(self, output_heatmap):
        output_heatmap = torch.sigmoid(output_heatmap)
        output_heatmap = self.find_heatmap_peaks(output_heatmap)

        return self.get_topk_indexes_class_agnostic(
            output_heatmap)

    def get_bounding_box_prediction(self, output_heatmap, output_offset, output_bbox, image_id, original_image_shape):
        batch, num_classes, height, width = output_heatmap.size()
        k = self.cfg["evaluation"]["topk_k"]
        topk_heatmap_value, topk_heatmap_index, topk_classes, topk_heatmap_index_row, topk_heatmap_index_column = self.process_output_heatmaps(
            output_heatmap)
        output_heatmap = topk_heatmap_value
        output_offset = _transpose_and_gather_output_feature(output_offset, topk_heatmap_index)  # .view(batch, k, 2)
        output_bbox = _transpose_and_gather_output_feature(output_bbox, topk_heatmap_index)  # .view(batch, k, 2)

        topk_heatmap_index_column = topk_heatmap_index_column + output_offset[:, :, 0]
        topk_heatmap_index_row = topk_heatmap_index_row + output_offset[:, :, 1]

        # [32,10] -> [32,10,1]
        topk_heatmap_index_row = topk_heatmap_index_row.unsqueeze(dim=2)
        topk_heatmap_index_column = topk_heatmap_index_column.unsqueeze(dim=2)
        output_bbox_width = output_bbox[:, :, 0].unsqueeze(dim=2)
        output_bbox_height = output_bbox[:, :, 1].unsqueeze(dim=2)
        scores = output_heatmap.unsqueeze(dim=2)
        class_label = topk_classes.unsqueeze(dim=2)
        # [32] ->[32, 10, 1]
        image_id = torch.cat(k * [image_id.unsqueeze(dim=1)], dim=1).unsqueeze(dim=2)
        print(" ")

        # [32,10,4]
        bbox = torch.cat([topk_heatmap_index_column - output_bbox_width / 2,
                          topk_heatmap_index_row - output_bbox_height / 2,
                          # topk_heatmap_index_column + output_bbox_width / 2,
                          # topk_heatmap_index_row + output_bbox_height / 2,
                          output_bbox_width,
                          output_bbox_height]
                         , dim=2)

        scale_with_network_input_dimension = True
        if (scale_with_network_input_dimension):
            bbox_scale_with_network_input_dimension = bbox * self.cfg["data"]["input_dimension"] / self.cfg["heatmap"][
                "output_dimension"]
            bbox_scale_with_network_input_dimension = bbox_scale_with_network_input_dimension.int()

        scale_with_image_input_dimension = True
        if (scale_with_image_input_dimension):
            original_image_shape_ratio = torch.cat(k * [original_image_shape.unsqueeze(dim=1)], dim=1) / \
                                         self.cfg["heatmap"][
                                             "output_dimension"]
            bbox_scale_with_image_input_dimension = torch.zeros_like(bbox)
            bbox_scale_with_image_input_dimension[:, :, 0:2] = bbox[:, :, 0:2] * original_image_shape_ratio
            bbox_scale_with_image_input_dimension[:, :, 2:4] = bbox[:, :, 2:4] * original_image_shape_ratio
            bbox_scale_with_image_input_dimension = bbox_scale_with_image_input_dimension.int()

        # [32,10,7]
        detections_with_network_input_dimension = torch.cat(
            [image_id, bbox_scale_with_network_input_dimension, scores, class_label], dim=2)
        detections_scale_with_image_input_dimension = torch.cat(
            [image_id, bbox_scale_with_image_input_dimension, scores, class_label], dim=2)
        # [32,70]
        detections_with_network_input_dimension = detections_with_network_input_dimension.view(batch * k, 7)
        detections_scale_with_image_input_dimension = detections_scale_with_image_input_dimension.view(batch * k, 7)
        return detections_with_network_input_dimension, detections_scale_with_image_input_dimension
