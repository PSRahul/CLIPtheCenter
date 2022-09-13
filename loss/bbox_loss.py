import torch
import torch.nn.functional as F
import segmentation_models_pytorch
from network.models.EfficientnetConv2DT.utils import gather_output_array, transpose_and_gather_output_array


def calculate_bbox_loss_without_heatmap(predicted_bbox, groundtruth_bbox, flattened_index, num_objects, device):
    predicted_bbox = transpose_and_gather_output_array(predicted_bbox, flattened_index)
    bbox_loss = F.smooth_l1_loss(predicted_bbox, groundtruth_bbox, reduction="mean")
    return bbox_loss


def calculate_bbox_loss_with_heatmap(predicted_bbox, groundtruth_bbox, flattened_index, num_objects, device):
    ################# DEBUG

    predicted_width = predicted_bbox[:, 0, :, :].flatten(start_dim=1, end_dim=- 1)
    predicted_height = predicted_bbox[:, 1, :, :].flatten(start_dim=1, end_dim=- 1)

    groundtruth_width = groundtruth_bbox[:, 0, :, :].flatten(start_dim=1, end_dim=- 1)
    groundtruth_height = groundtruth_bbox[:, 1, :, :].flatten(start_dim=1, end_dim=- 1)

    bbox_loss_width = torch.nn.functional.mse_loss(input=predicted_width.float(),
                                                   target=groundtruth_width.float(),
                                                   reduction='mean')
    bbox_loss_height = torch.nn.functional.mse_loss(input=predicted_height.float(),
                                                    target=groundtruth_height.float(),
                                                    reduction='mean')
    bbox_loss = bbox_loss_height + bbox_loss_width
    return bbox_loss


def calculate_bbox_loss_with_heatmap_old(predicted_bbox, groundtruth_bbox, flattened_index, num_objects, device):
    ################# DEBUG
    groundtruth_bbox_np = groundtruth_bbox.detach().cpu().numpy()
    groundtruth_bbox_np = groundtruth_bbox_np[0, 0, :, :]

    heatmap_loss_fn_width = segmentation_models_pytorch.losses.FocalLoss(mode="multiclass")
    heatmap_loss_fn_height = segmentation_models_pytorch.losses.FocalLoss(mode="multiclass")
    predicted_width, predicted_height = predicted_bbox
    groundtruth_width = groundtruth_bbox[:, 0, :, :]
    groundtruth_height = groundtruth_bbox[:, 1, :, :]

    bbox_loss_width = heatmap_loss_fn_width(predicted_width, groundtruth_width)
    bbox_loss_height = heatmap_loss_fn_height(predicted_height, groundtruth_height)

    # bbox_loss_width = torch.nn.functional.mse_loss(input=predicted_bbox[:, 0, :, :].float(),
    #                                               target=groundtruth_bbox[:, 0, :, :].float(),
    #                                               reduction='mean')
    # bbox_loss_height = torch.nn.functional.mse_loss(input=predicted_bbox[:, 1, :, :].float(),
    #                                                target=groundtruth_bbox[:, 1, :, :].float(),
    #                                                reduction='mean')
    bbox_loss = bbox_loss_height + bbox_loss_width
    return bbox_loss
