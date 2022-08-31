import torch
import torch.nn.functional as F
from network.models.EfficientnetConv2DT.utils import gather_output_array, transpose_and_gather_output_array


def calculate_offset_loss(predicted_offset, groundtruth_offset, flattened_index, num_objects, device):
    predicted_offset = transpose_and_gather_output_array(predicted_offset, flattened_index)
    object_boolean_mask = torch.zeros((flattened_index.shape), device=device)
    for i in range(object_boolean_mask.shape[0]):
        object_boolean_mask[i, 0:int(num_objects[i])] = 1
    object_boolean_mask = object_boolean_mask.unsqueeze(2).expand_as(predicted_offset).float()
    predicted_offset, groundtruth_offset = predicted_offset * object_boolean_mask, groundtruth_offset * object_boolean_mask
    offset_loss = F.smooth_l1_loss(predicted_offset, groundtruth_offset, reduction="sum")
    offset_loss = offset_loss / object_boolean_mask.sum()
    return offset_loss
