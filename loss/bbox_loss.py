import torch
import torch.nn.functional as F

from network.models.EfficientnetConv2DT.utils import gather_output_array, transpose_and_gather_output_array


def calculate_bbox_loss_without_heatmap(predicted_bbox, groundtruth_bbox, flattened_index, num_objects, device):
    predicted_bbox = transpose_and_gather_output_array(predicted_bbox, flattened_index)
    bbox_loss = F.smooth_l1_loss(predicted_bbox, groundtruth_bbox, reduction="mean")
    return bbox_loss


def calculate_bbox_loss_without_heatmap(predicted_bbox, groundtruth_bbox, flattened_index, num_objects, device):
    predicted_bbox = transpose_and_gather_output_array(predicted_bbox, flattened_index)
    bbox_loss = F.smooth_l1_loss(predicted_bbox, groundtruth_bbox, reduction="mean")
    return bbox_loss
