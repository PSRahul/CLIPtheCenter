from matplotlib import pyplot as plt
import numpy as np
import yaml
from yaml.loader import SafeLoader
import argparse
import logging
import os
from datetime import datetime
import sys
import pathlib
import albumentations as A
import cv2
from data.dataset_module import DataModule
import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    BOX_COLOR = (255, 0, 0)  # Red
    TEXT_COLOR = (255, 255, 255)  # White

    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


def visualise_coco():
    with open("/configs/config.yaml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    cfg = load_config()
    coco_dataset = DataModule(cfg)
    image = coco_dataset.train_dataset._load_image(5)
    anns = coco_dataset.train_dataset._load_target(5)
    image = np.array(image)
    bounding_box_list = []
    class_list = []
    transform = A.Compose([
        A.RandomCrop(width=324, height=324),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='coco', min_area=1600, min_visibility=0.1, label_fields=['class_labels']))

    for ann in anns:
        bounding_box_list.append(ann['bbox'])
        class_list.append(ann['category_id'])

    transformed = transform(image=image, bboxes=bounding_box_list, class_labels=class_list)
    transformed_image = transformed['image']
    transformed_bounding_box_list = transformed['bboxes']
    transformed_class_list = transformed['class_labels']
    category_id_to_name = {11: 'cat', 16: 'dog'}

    visualize(image, bounding_box_list, class_list, category_id_to_name)
