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
import copy


def visualise_bbox(cfg, dataset, id, gt=None, pred=None, draw_gt=True, draw_pred=True,
                   resize_image_to_output_shape=False):
    image = dataset._load_image(id)

    image = np.array(image)
    if (resize_image_to_output_shape):
        image = cv2.resize(image,
                           (cfg["post_processing"]["model_output_shape"], cfg["post_processing"]["model_output_shape"]))

    fig, ax = plt.subplots()
    ax.imshow(image)

    if draw_pred:
        predictions_image = pred[pred[:, 0] == int(id)]
        print("Number of Predictions", predictions_image.shape[0])
        for i in range(predictions_image.shape[0]):
            bbox_i = predictions_image[i, 1:5]
            rect = patches.Rectangle(
                (bbox_i[0], bbox_i[1]), bbox_i[2],
                bbox_i[3], linewidth=2, edgecolor='r',
                facecolor='none')
            ax.add_patch(rect)

    if draw_gt:
        gt_image = gt[gt[:, 0] == int(id)]
        print("Number of GroundTruth Objects", gt_image.shape[0])
        for i in range(gt_image.shape[0]):
            bbox_i = gt_image[i, 1:5]
            rect = patches.Rectangle(
                (bbox_i[0], bbox_i[1]), bbox_i[2],
                bbox_i[3], linewidth=2, edgecolor='b',
                facecolor='none')
            ax.add_patch(rect)

    plt.show()
