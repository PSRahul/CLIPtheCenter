import torch
import torch.nn.functional as F

from network.models.efficientnet_conv2dT.utils import gather_output_array, transpose_and_gather_output_array


def calculate_bbox_loss(predicted_bbox, groundtruth_bbox, flattened_index, num_objects, device):
    predicted_bbox = transpose_and_gather_output_array(predicted_bbox, flattened_index)
    object_boolean_mask = torch.zeros((flattened_index.shape), device=device)
    for i in range(object_boolean_mask.shape[0]):
        object_boolean_mask[i, 0:int(num_objects[i])] = 1
    object_boolean_mask = object_boolean_mask.unsqueeze(2).expand_as(predicted_bbox).float()
    predicted_bbox, groundtruth_bbox = predicted_bbox * object_boolean_mask, groundtruth_bbox * object_boolean_mask
    bbox_loss = F.smooth_l1_loss(predicted_bbox, groundtruth_bbox, reduction="sum")
    bbox_loss = bbox_loss / object_boolean_mask.sum()
    return bbox_loss
