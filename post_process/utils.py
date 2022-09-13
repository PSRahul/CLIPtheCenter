import argparse
import os
import shutil
import sys
from datetime import datetime
from post_process.coco_evaluation import calculate_coco_result
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import cv2
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from yaml.loader import SafeLoader
from post_process.torchmetric_evaluation import calculate_torchmetrics_mAP
from post_process.nms import perform_nms


def resize_predictions_image_size(cfg, dataset, prediction):
    for i in tqdm(range(prediction.shape[0])):
        image = dataset._load_image(int(prediction[i, 0]))
        image = np.array(image)
        height_scale = image.shape[0] / cfg["post_processing"]["model_output_shape"]
        width_scale = image.shape[1] / cfg["post_processing"]["model_output_shape"]

        # columns = [["image_id", "bbox_x", "bbox_y", "w", "h", "score", "class_label"]]
        prediction[i, 1], prediction[i, 3] = int(prediction[i, 1] * width_scale), int(prediction[i, 3] * width_scale)
        prediction[i, 2], prediction[i, 4] = int(prediction[i, 2] * height_scale), int(prediction[i, 4] * height_scale)
    return prediction


def assign_classes(clip_encodings, predictions):
    # ["image_id", "bbox_y", "bbox_x", "w", "h", "score", "class_label"]
    predictions, embeddings = predictions[:, 0:7], predictions[:, 7:]
    embeddings = torch.tensor(embeddings)
    clip_encodings = torch.tensor(clip_encodings)
    embeddings /= embeddings.norm(dim=-1, keepdim=True)
    clip_encodings /= clip_encodings.norm(dim=-1, keepdim=True)
    scores = torch.matmul(embeddings.float(), clip_encodings.T.float())
    top_probs, top_labels = scores.cpu().topk(1, dim=-1)
    classes = top_labels.numpy().ravel()
    predictions[:, 6] = classes
    return predictions
