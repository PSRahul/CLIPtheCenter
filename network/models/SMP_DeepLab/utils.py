import numpy as np
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
        heatmap_np = output_heatmap.detach().cpu()[0].squeeze(0).numpy()
        plt.imshow(heatmap_np, cmap='Greys')
        plt.show()

    # output_heatmap = torch.sigmoid(output_heatmap)
    if (cfg["debug"]):
        heatmap_np = output_heatmap.detach().cpu()[0].squeeze(0).numpy()
        plt.imshow(heatmap_np, cmap='Greys')
        plt.show()

    # output_heatmap = output_heatmap / output_heatmap.max()
    if (cfg["debug"]):
        heatmap_np = output_heatmap.detach().cpu()[0].squeeze(0).numpy()
        plt.imshow(heatmap_np, cmap='Greys')
        plt.show()

    # output_heatmap = find_heatmap_peaks(cfg, output_heatmap)

    if (cfg["debug"]):
        heatmap_np = output_heatmap.detach().cpu()[0].squeeze(0).numpy()
        plt.imshow(heatmap_np, cmap='Greys')
        plt.show()

    return get_topk_indexes_class_agnostic(cfg,
                                           output_heatmap)


def get_bbox_from_heatmap(output_bbox, topk_heatmap_index_row, topk_heatmap_index_column, max_width=5):
    batch_size = output_bbox.shape[0]
    topk_size = topk_heatmap_index_column.shape[1]
    filtered_bbox = torch.zeros((batch_size, topk_size, 2), device="cuda")
    for batch_index in range(batch_size):
        for topk_index in range(topk_size):
            height = topk_heatmap_index_row[batch_index, topk_index].int().cpu()
            width = topk_heatmap_index_column[batch_index, topk_index].int().cpu()
            left = np.maximum(width - max_width, 0)
            right = np.minimum(width + max_width, output_bbox.shape[2])
            top = np.maximum(height - max_width, 0)
            bottom = np.minimum(height + max_width, output_bbox.shape[2])

            output_bbox_index = output_bbox[batch_index, :, top:bottom,
                                left:right]
            output_bbox_w_index = output_bbox_index[0]
            output_bbox_h_index = output_bbox_index[1]

            ############DEBUG
            output_bbox_w_index_np = output_bbox_w_index.cpu().detach().numpy()
            output_bbox_h_index_np = output_bbox_h_index.cpu().detach().numpy()
            ############DEBUG
            output_bbox_h_index = output_bbox_h_index.flatten()
            output_bbox_w_index = output_bbox_w_index.flatten()
            # print(torch.max(output_bbox_w_index, dim=0)[0])
            filtered_bbox[batch_index, topk_index, 0] = torch.max(output_bbox_w_index, dim=0)[0]
            filtered_bbox[batch_index, topk_index, 1] = torch.max(output_bbox_h_index, dim=0)[0]

    return filtered_bbox


def get_bounding_box_prediction(cfg, output_heatmap, output_bbox, image_id):
    batch, num_classes, height, width = output_heatmap.size()

    k = cfg["evaluation"]["topk_k"]

    topk_heatmap_value, topk_heatmap_index, topk_classes, topk_heatmap_index_row, topk_heatmap_index_column = process_output_heatmaps(
        cfg, output_heatmap)

    output_heatmap = topk_heatmap_value
    output_bbox = get_bbox_from_heatmap(output_bbox, topk_heatmap_index_row, topk_heatmap_index_column)
    # output_bbox = transpose_and_gather_output_array(output_bbox, topk_heatmap_index)  # .view(batch, k, 2)

    # [32,10] -> [32,10,1]
    topk_heatmap_index_row = topk_heatmap_index_row.unsqueeze(dim=2)
    topk_heatmap_index_column = topk_heatmap_index_column.unsqueeze(dim=2)
    output_bbox_width = output_bbox[:, :, 0].unsqueeze(dim=2)
    output_bbox_height = output_bbox[:, :, 1].unsqueeze(dim=2)
    scores = output_heatmap.unsqueeze(dim=2)
    class_label = topk_classes.unsqueeze(dim=2)
    # [32] ->[32, 10, 1]
    image_id = torch.cat(k * [image_id.unsqueeze(dim=1)], dim=1).unsqueeze(dim=2)

    # [32,10,4]
    bbox = torch.cat([topk_heatmap_index_column - output_bbox_width / 2,
                      topk_heatmap_index_row - output_bbox_height / 2,
                      # topk_heatmap_index_column + output_bbox_width / 2,
                      # topk_heatmap_index_row + output_bbox_height / 2,
                      output_bbox_width,
                      output_bbox_height]
                     , dim=2)
    bbox = bbox.int()

    # bbox = bbox.int()
    # [32,10,7]
    detections = torch.cat(
        [image_id, bbox, scores, class_label], dim=2)

    # [32,70]
    detections = detections.view(batch * k, 7)
    # detections = detections[
    #    detections[:, 5] >= float(self.cfg["evaluation"]["score_threshold"])]
    return detections
