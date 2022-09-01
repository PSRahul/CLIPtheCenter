import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms


def perform_nms(cfg, prediction):
    columns = [["image_id", "bbox_x", "bbox_y", "w", "h", "score", "class_label"]]

    df = pd.DataFrame(prediction, columns=columns)
    image_id_list = list(np.unique(df["image_id"].values))  # .tolist()  # .unique()  # .tolist()
    df["bbox_x+w"] = df["bbox_x"].values + df["w"].values
    df["bbox_y+h"] = df["bbox_y"].values + df["h"].values
    filtered_predictions = np.empty((0, 7))
    for image_id in image_id_list:
        df_id = df[df["image_id"].values == image_id]
        y1, x1 = df_id["bbox_y"].values, df_id["bbox_x"].values
        y2, x2 = df_id["bbox_y+h"].values, df_id["bbox_x+w"].values
        boxes = torch.tensor(np.hstack((x1, y1, x2, y2)))
        scores = torch.tensor(df_id["score"].values).squeeze(dim=1)
        nms_ind = nms(boxes, scores, iou_threshold=cfg["post_processing"]["nms_iou_threshold"]).numpy()

        # columns = [["image_id", "bbox_x", "bbox_y", "w", "h", "score", "class_label"]]
        # Filter based on the scores
        scores_filtered = scores[nms_ind]
        filter_by_nms_and_score_ind = scores_filtered > cfg["post_processing"]["score_threshold"]
        filter_by_nms_and_score_ind = filter_by_nms_and_score_ind.numpy()
        if (len(filter_by_nms_and_score_ind) != 0):
            nms_score_ind = nms_ind[filter_by_nms_and_score_ind]
            df_numpy = df_id.iloc[nms_score_ind].to_numpy()
            df_numpy = df_numpy[:, 0:7]
            filtered_predictions = np.vstack((filtered_predictions, df_numpy))

    return filtered_predictions
