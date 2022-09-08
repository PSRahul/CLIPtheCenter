import torchvision
import torch


def calculate_heatmap_loss(predicted_heatmap, groundtruth_heatmap):
    # heatmap_loss = torchvision.ops.sigmoid_focal_loss(inputs=predicted_heatmap, targets=groundtruth_heatmap,
    #                                   reduction="mean")
    heatmap_loss = torch.nn.functional.mse_loss(input=predicted_heatmap, target=groundtruth_heatmap, reduction='mean')
    return heatmap_loss
