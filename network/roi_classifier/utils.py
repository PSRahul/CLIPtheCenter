import torch


def get_masked_heatmaps(cfg, output_heatmap, output_mask):
    masked_heatmaps = torch.zeros_like(output_mask, device="cuda")
    for batch_index in range(cfg["data"]["train_batch_size"]):
        for detection_index in range(cfg["evaluation"]["topk_k"]):
            heatmap = output_heatmap[batch_index].squeeze(0)
            index = cfg["evaluation"]["topk_k"] * batch_index + detection_index
            masked_heatmaps[index] = heatmap * output_mask[index]

    masked_heatmaps = masked_heatmaps.unsqueeze(dim=1)
    return masked_heatmaps
