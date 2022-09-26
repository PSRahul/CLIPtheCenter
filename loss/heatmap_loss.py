import torchvision
import torch
import segmentation_models_pytorch



def calculate_heatmap_loss(predicted_heatmap, groundtruth_heatmap):
    predicted_heatmap = predicted_heatmap.unsqueeze(dim=1)
    groundtruth_heatmap = groundtruth_heatmap.unsqueeze(dim=1)
    heatmap_loss_fn = segmentation_models_pytorch.losses.FocalLoss(mode="binary",normalized=True)
    heatmap_loss = heatmap_loss_fn(predicted_heatmap.float(), groundtruth_heatmap.float())
    return heatmap_loss
def calculate_heatmap_loss(predicted_heatmap, groundtruth_heatmap):
    heatmap_loss = torch.nn.functional.mse_loss(input=predicted_heatmap.float(), target=groundtruth_heatmap.float(),
                                                reduction='mean')
    return heatmap_loss