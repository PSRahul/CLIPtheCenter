import torchvision


def calculate_heatmap_loss(predicted_heatmap, groundtruth_heatmap):
    heatmap_loss = torchvision.ops.sigmoid_focal_loss(inputs=predicted_heatmap, targets=groundtruth_heatmap,
                                                      reduction="mean")
    return heatmap_loss
