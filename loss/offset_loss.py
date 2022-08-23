import torch
import torch.nn.functional as F


def _gather_output_feature(output_feature, ind, mask=None):
    dim = output_feature.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    output_feature = output_feature.gather(1, ind.type(torch.int64))
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(output_feature)
        output_feature = output_feature[mask]
        output_feature = output_feature.view(-1, dim)
    return output_feature


def _transpose_and_gather_output_feature(output_feature, ind):
    output_feature = output_feature.permute(0, 2, 3, 1).contiguous()
    output_feature = output_feature.view(output_feature.size(0), -1, output_feature.size(3))
    output_feature = _gather_output_feature(output_feature, ind)
    return output_feature


def calculate_offset_loss(predicted_offset, groundtruth_offset, flattened_index, num_objects):
    predicted_offset = _transpose_and_gather_output_feature(predicted_offset, flattened_index)
    object_boolean_mask = torch.zeros((flattened_index.shape), device="cuda")
    for i in range(object_boolean_mask.shape[0]):
        object_boolean_mask[i, 0:int(num_objects[i])] = 1
    object_boolean_mask = object_boolean_mask.unsqueeze(2).expand_as(predicted_offset).float()
    predicted_offset, groundtruth_offset = predicted_offset * object_boolean_mask, groundtruth_offset * object_boolean_mask
    offset_loss = F.smooth_l1_loss(predicted_offset, groundtruth_offset, reduction="sum")
    offset_loss = offset_loss / object_boolean_mask.sum()
    return offset_loss
