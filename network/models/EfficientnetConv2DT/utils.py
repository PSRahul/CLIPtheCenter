import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def gather_output_array(output_array, ind, mask=None):
    dim = output_array.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    output_array = output_array.gather(1, ind.type(torch.int64))
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(output_array)
        output_array = output_array[mask]
        output_array = output_array.view(-1, dim)
    return output_array


def transpose_and_gather_output_array(output_array, ind):
    output_array = output_array.permute(0, 2, 3, 1).contiguous()
    output_array = output_array.view(output_array.size(0), -1, output_array.size(3))
    output_array = gather_output_array(output_array, ind)
    return output_array


def find_heatmap_peaks(cfg, output_heatmap):
    kernel = cfg["evaluation"]["heatmap_pooling_kernel"]
    pad = (kernel - 1) // 2

    output_heatmap_max = nn.functional.max_pool2d(
        output_heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (output_heatmap_max == output_heatmap).float()
    return output_heatmap * keep


def get_topk_indexes_class_agnostic(cfg, output_heatmap):
    k = cfg["evaluation"]["topk_k"]
    output_heatmap = output_heatmap.squeeze(dim=1).to("cuda")

    batch, height, width = output_heatmap.size()
    topk_heatmap_value, topk_heatmap_index = torch.topk(
        output_heatmap.view(batch, -1), k)
    topk_heatmap_index_row = (topk_heatmap_index / width).int().float()
    topk_heatmap_index_column = (topk_heatmap_index % width).int().float()

    topk_classes = torch.zeros((batch, k), device="cuda")
    # Across second dimension, divide by the number of class. There are total of N filters. We choose top k scores per class. We so divide N by k to get the class correspondence.
    return topk_heatmap_value, topk_heatmap_index, topk_classes, topk_heatmap_index_row, topk_heatmap_index_column


def process_output_heatmaps(cfg, output_heatmap):
    if (cfg["debug"]):
        heatmap_np = output_heatmap.detach().cpu().squeeze(0).squeeze(0).numpy()
        plt.imshow(heatmap_np, cmap='Greys')
        plt.show()

    output_heatmap = torch.sigmoid(output_heatmap)
    if (cfg["debug"]):
        heatmap_np = output_heatmap.detach().cpu().squeeze(0).squeeze(0).numpy()
        plt.imshow(heatmap_np, cmap='Greys')
        plt.show()

    output_heatmap = output_heatmap / output_heatmap.max()
    if (cfg["debug"]):
        heatmap_np = output_heatmap.detach().cpu().squeeze(0).squeeze(0).numpy()
        plt.imshow(heatmap_np, cmap='Greys')
        plt.show()

    output_heatmap = find_heatmap_peaks(cfg, output_heatmap)

    if (cfg["debug"]):
        heatmap_np = output_heatmap.detach().cpu().squeeze(0).squeeze(0).numpy()
        plt.imshow(heatmap_np, cmap='Greys')
        plt.show()

    return get_topk_indexes_class_agnostic(cfg,
                                           output_heatmap)
