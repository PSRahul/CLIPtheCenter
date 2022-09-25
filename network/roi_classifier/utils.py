import torch
import copy


def get_binary_masks(cfg, output_roi):
    mask = torch.zeros(
        (output_roi.shape[0], cfg["heatmap"]["output_dimension"], cfg["heatmap"]["output_dimension"]))
    output_roi = output_roi[:, 1:5].int()
    for i in range(output_roi.shape[0]):
        bbox = output_roi[i]

        mask[i, bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = 1
    return mask


def get_masked_heatmaps(cfg, roi_heatmap, binary_mask, train_set=True):
    if (train_set):
        batch_size = cfg["data"]["train_batch_size"]
    else:
        batch_size = cfg["data"]["val_batch_size"]
    masked_roi_heatmaps = torch.zeros_like(binary_mask, device="cuda")
    for batch_index in range(batch_size):
        for detection_index in range(cfg["evaluation"]["topk_k"]):
            heatmap = roi_heatmap[batch_index].squeeze(0)
            index = cfg["evaluation"]["topk_k"] * batch_index + detection_index
            masked_roi_heatmaps[index] = heatmap * binary_mask[index]

    masked_roi_heatmaps = masked_roi_heatmaps.unsqueeze(dim=1)
    return masked_roi_heatmaps


def make_detections_valid(cfg, detections):
    detections_valid = copy.deepcopy(detections)
    detections_valid[detections_valid[:, 3] <= 10, 3] = 10
    detections_valid[detections_valid[:, 4] <= 10, 4] = 10
    for i in range(detections.shape[0]):
        x = detections_valid[i, 1] + detections_valid[i, 3]
        y = detections_valid[i, 2] + detections_valid[i, 4]
        if (x > cfg["heatmap"]["output_dimension"] - 1):
            detections_valid[i, 3] = cfg["heatmap"]["output_dimension"] - 1 - detections_valid[i, 1]
        if (y > cfg["heatmap"]["output_dimension"] - 1):
            detections_valid[i, 4] = cfg["heatmap"]["output_dimension"] - 1 - detections_valid[i, 2]
    return detections_valid

def make_detections_valid(cfg, detections):
    detections_valid = copy.deepcopy(detections)
    detections_valid[detections_valid[:, 1] <= 0, 1] = 0
    detections_valid[detections_valid[:, 2] <= 0, 2] = 0

    detections_valid[detections_valid[:, 3] <= 10, 3] = 10
    detections_valid[detections_valid[:, 4] <= 10, 4] = 10
    for i in range(detections.shape[0]):
        x = detections_valid[i, 1] + detections_valid[i, 3]
        y = detections_valid[i, 2] + detections_valid[i, 4]

        # Exceeds the boundary
        if (x > cfg["heatmap"]["output_dimension"] - 1):
            detections_valid[i, 3] = cfg["heatmap"]["output_dimension"] - 1 - detections_valid[i, 1]
            if(detections_valid[i, 3]<10):
                detections_valid[i, 1] = cfg["heatmap"]["output_dimension"] - 1-10
                detections_valid[i, 3]=10
        if (y > cfg["heatmap"]["output_dimension"] - 1):
            detections_valid[i, 4] = cfg["heatmap"]["output_dimension"] - 1 - detections_valid[i, 2]
            if (detections_valid[i, 4] < 10):
                detections_valid[i, 2] = cfg["heatmap"]["output_dimension"] - 1 - 10
                detections_valid[i, 4] = 10

    return detections_valid