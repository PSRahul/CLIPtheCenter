import torch
import torch.nn.functional as F
import segmentation_models_pytorch
from network.models.EfficientnetConv2DT.utils import gather_output_array, transpose_and_gather_output_array

import numpy as np

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

def calculate_bbox_loss_without_heatmap(predicted_bbox, groundtruth_bbox, flattened_index, num_objects, device):
    predicted_bbox = transpose_and_gather_output_array(predicted_bbox, flattened_index)
    bbox_loss = F.mse_loss(predicted_bbox.float(), groundtruth_bbox.float(), reduction="mean")
    return bbox_loss


def calculate_bbox_loss_with_giou(predicted_bbox, groundtruth_bbox, flattened_index, num_objects, device):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    #bboxes1, bboxes2
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form

    predicted_bbox = transpose_and_gather_output_array(predicted_bbox, flattened_index)
    predicted_bbox=predicted_bbox.squeeze(dim=1)
    groundtruth_bbox=groundtruth_bbox.squeeze(dim=1)

    bboxes1, bboxes2=predicted_bbox,groundtruth_bbox
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1
    hc = yyc2 - yyc1
    assert((wc > 0).all() and (hc > 0).all())
    area_enclose = wc * hc
    giou = iou - (area_enclose - wh) / area_enclose
    giou = (giou + 1.)/2.0 # resize from (-1,1) to (0,1)
    return giou
