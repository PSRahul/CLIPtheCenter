import argparse
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import yaml
from yaml.loader import SafeLoader
from tempfile import TemporaryFile

from data.dataset_module import DataModule
from network.model_builder import DetectionModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="metric_config.yaml")
    args = parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    return config


class Logger(object):
    def __init__(self, cfg, checkpoint_dir):
        self.terminal = sys.stdout
        self.log = open(checkpoint_dir, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def set_logging(cfg):
    now = datetime.now()
    date_save_string = now.strftime("%d%m%Y_%H%M")
    checkpoint_dir = os.path.join(
        "/home/psrahul/MasterThesis/repo/Phase3/CLIPandDetect/",
        cfg["logging"]["checkpoint_dir"],
        date_save_string,
    )
    print(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, "log.log")
    return log_file, checkpoint_dir


def get_original_image(test_dataset, index):
    index = test_dataset.ids[index]
    image = np.array(test_dataset._load_image(index))
    return image


def get_transformed_image(test_dataset, index):
    image, _, _, _ = np.array(test_dataset.get_transformed_image(index))
    return image


def get_bbox_from_groundtruth(test_dataset, index, cfg):
    bbox_list = []
    image_item = test_dataset[index]
    heatmap = image_item['heatmap']
    bbox = image_item['bbox']
    offset = image_item['offset']
    flattened_index = image_item['flattened_index']
    num_objects = image_item['num_objects']
    width = cfg["heatmap"]["output_dimension"]
    for bbox_index in range(num_objects):
        index = flattened_index[bbox_index]
        center_y = (index / width).int().float()
        center_x = (index % width).int().float()
        center_y = center_y + offset[bbox_index, 0]
        center_x = center_x + offset[bbox_index, 1]
        bbox_array = np.array([center_y - bbox[bbox_index, 0] / 2,
                               center_x - bbox[bbox_index, 1] / 2,
                               center_y + bbox[bbox_index, 0] / 2,
                               center_x + bbox[bbox_index, 1] / 2,
                               ])
        bbox_list.append(bbox_array)
    return np.array(bbox_list), num_objects


def main(cfg):
    detection_model = DetectionModel(cfg)
    coco_dataset = DataModule(cfg)
    test_dataset = coco_dataset.load_val_dataset()
    predictions = np.load(cfg["metrics"]["prediction_path"])
    for index in range(len(test_dataset)):
        index = 5
        id = test_dataset.ids[index]
        predictions_image = predictions[predictions[:, 0] == int(id)]
        predictions_image_bbox = predictions_image[:, 1:5]
        predictions_image_score = predictions_image[:, 5]
        predictions_image_class = predictions_image[:, 0]

        predictions_image_bbox[:, 2] = predictions_image_bbox[:, 1]
        predictions_image_bbox[:, 3] += predictions_image_bbox[:, 0]
        gt_bbox, num_objects = get_bbox_from_groundtruth(test_dataset, index, cfg)

    print("breakpoint")
    gt_class = np.zeros((gt_bbox.shape[0]))
    predictions_class = np.zeros((predictions_image_bbox.shape[0]))
    outfile = TemporaryFile()
    np.savez("/home/psrahul/MasterThesis/repo/Phase3/CLIPandDetect/Metrics/gt.npz", gt_class=gt_class, gt_bbox=gt_bbox)
    outfile = TemporaryFile()
    np.savez("/home/psrahul/MasterThesis/repo/Phase3/CLIPandDetect/Metrics/predictions.npz",
             predictions_class=predictions_class,
             predictions_bbox=predictions_image_bbox)


if __name__ == "__main__":
    args = get_args()
    cfg = load_config(args.c)
    log_file, checkpoint_dir = set_logging(cfg)
    shutil.copyfile(args.c, os.path.join(checkpoint_dir, "../configs/config.yaml"))

    sys.exit(main(cfg))
