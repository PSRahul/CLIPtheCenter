import torchvision
import torch
import segmentation_models_pytorch


def calculate_heatmap_loss(predicted_heatmap, groundtruth_heatmap):
    # heatmap_loss = torchvision.ops.sigmoid_focal_loss(inputs=predicted_heatmap, targets=groundtruth_heatmap,
    #                                   reduction="mean")
    # gradient_mask = torch.zeros_like(groundtruth_heatmap)
    # gradient_mask[groundtruth_heatmap != 0] = 1
    predicted_heatmap = predicted_heatmap.unsqueeze(dim=1)
    groundtruth_heatmap = groundtruth_heatmap.unsqueeze(dim=1)
    heatmap_loss_fn = segmentation_models_pytorch.losses.FocalLoss(mode="binary")
    heatmap_loss = heatmap_loss_fn(predicted_heatmap.float(), groundtruth_heatmap.float())
    # predicted_heatmap = predicted_heatmap * gradient_mask
    # groundtruth_heatmap = groundtruth_heatmap * gradient_mask
    # heatmap_loss = torch.nn.functional.mse_loss(input=predicted_heatmap.float(), target=groundtruth_heatmap.float(),
    #                                            reduction='mean')
    return heatmap_loss


def calculate_heatmap_loss(predicted_heatmap, groundtruth_heatmap):
    heatmap_loss = torch.nn.functional.mse_loss(input=predicted_heatmap.float(), target=groundtruth_heatmap.float(),
                                                reduction='mean')
    return heatmap_loss
