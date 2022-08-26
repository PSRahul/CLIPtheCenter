import argparse
import os
import shutil
import sys
from datetime import datetime
import yaml
from yaml.loader import SafeLoader
import numpy as np
from data.dataset_module import DataModule
from network.model_builder import DetectionModel
from trainer.trainer_module import Trainer
from evaluation.evaluation_module import EvalMetrics
from evaluation.coco_eval import COCORunner
import matplotlib.pyplot as plt
import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# matplotlib.use('Agg')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="visualise/visualise_config.yaml")
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
    # plt.imshow(image)
    # plt.show()
    return image


def get_transformed_image(test_dataset, index):
    image, _, _, _ = np.array(test_dataset.get_transformed_image(index))
    # plt.imshow(image)
    # plt.show()
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
        # print("Centers", center_x, center_y)
        bbox_array = np.array([center_y - bbox[bbox_index, 0] / 2,
                               center_x - bbox[bbox_index, 1] / 2,
                               bbox[bbox_index, 0],
                               bbox[bbox_index, 1],
                               ])
        # bbox *= cfg["data"]["input_dimension"] / cfg["heatmap"][
        #    "output_dimension"]
        bbox_list.append(bbox_array)
    return bbox_list, num_objects


def draw_bbox_gt(test_dataset, index, cfg, checkpoint_dir):
    image = get_transformed_image(test_dataset, index)

    image = cv2.resize(image, (cfg["heatmap"]["output_dimension"], cfg["heatmap"]["output_dimension"]))
    fig, ax = plt.subplots()
    ax.imshow(image)

    bbox, num_objects = get_bbox_from_groundtruth(test_dataset, index, cfg)
    for i in range(num_objects):
        bbox_i = bbox[i] / cfg["heatmap"]["output_dimension"] * cfg["data"]["input_dimension"]
        # print(bbox_i)
        rect = patches.Rectangle(
            (bbox_i[1], bbox_i[0]), bbox_i[3],
            bbox_i[2], linewidth=1, edgecolor='r',
            facecolor='none')
        ax.add_patch(rect)
    # plt.savefig(os.path.join(checkpoint_dir, str(index) + ".png"))
    plt.close("all")
    # plt.show()


def draw_bbox_pred_heatmap_size(test_dataset, index, predictions):
    image = get_transformed_image(test_dataset, index)

    image = cv2.resize(image, (cfg["heatmap"]["output_dimension"], cfg["heatmap"]["output_dimension"]))
    fig, ax = plt.subplots()
    ax.imshow(image)

    bbox, num_objects = get_bbox_from_groundtruth(test_dataset, index, cfg)
    for i in range(num_objects):
        bbox_i = bbox[i]
        # print(bbox_i)
        rect = patches.Rectangle(
            (bbox_i[1], bbox_i[0]), bbox_i[3],
            bbox_i[2], linewidth=2, edgecolor='b',
            facecolor='none')
        ax.add_patch(rect)
    bbox_list = []
    num_objects = 0

    for i in range(predictions.shape[0]):
        bbox_array = predictions[i, 1:5]
        bbox_list.append(bbox_array)
        num_objects += 1
    for i in range(num_objects):
        bbox_i = bbox_list[i]
        # print(bbox_i)
        rect = patches.Rectangle(
            (bbox_i[1], bbox_i[0]), bbox_i[3],
            bbox_i[2], linewidth=2, edgecolor='r',
            facecolor='none')
        ax.add_patch(rect)
    plt.savefig(os.path.join(checkpoint_dir, str(index) + ".png"))
    # plt.show()
    plt.close("all")
    #


def draw_bbox_pred_input_size(test_dataset, index, predictions):
    image = get_transformed_image(test_dataset, index)

    image = cv2.resize(image, (cfg["heatmap"]["output_dimension"], cfg["heatmap"]["output_dimension"]))
    fig, ax = plt.subplots()
    ax.imshow(image)

    bbox, num_objects = get_bbox_from_groundtruth(test_dataset, index, cfg)
    for i in range(num_objects):
        bbox_i = bbox[i]  # * cfg["data"]["input_dimension"] / cfg["heatmap"]["output_dimension"]
        # print(bbox_i)
        rect = patches.Rectangle(
            (bbox_i[1], bbox_i[0]), bbox_i[3],
            bbox_i[2], linewidth=2, edgecolor='b',
            facecolor='none')
        ax.add_patch(rect)
    bbox_list = []
    num_objects = 0

    for i in range(predictions.shape[0]):
        bbox_array = predictions[i, 1:5]  # * cfg["data"]["input_dimension"] / cfg["heatmap"]["output_dimension"]
        bbox_list.append(bbox_array)
        num_objects += 1
    for i in range(num_objects):
        bbox_i = bbox_list[i]
        # print(bbox_i)
        rect = patches.Rectangle(
            (bbox_i[1], bbox_i[0]), bbox_i[3],
            bbox_i[2], linewidth=2, edgecolor='r',
            facecolor='none')
        ax.add_patch(rect)
    plt.savefig(os.path.join(checkpoint_dir, str(index) + ".png"))
    # plt.show()
    plt.close("all")
    #


def main(cfg):
    detection_model = DetectionModel(cfg)
    coco_dataset = DataModule(cfg)
    test_dataset = coco_dataset.load_val_dataset()

    # for index in range(len(test_dataset)):
    #    draw_bbox(test_dataset, index, cfg, checkpoint_dir)

    predictions = np.load(cfg["visualise"]["prediction_path"])
    for index in range(len(test_dataset)):
        id = test_dataset.ids[index]
        predictions_image = predictions[predictions[:, 0] == int(id)]
        # predictions_image = predictions_image[predictions_image[:, 5] >= float(cfg["visualise"]["score_threshold"])]
        if (predictions_image.shape[0] != 0):
            draw_bbox_pred_input_size(test_dataset, index, predictions_image)


if __name__ == "__main__":
    args = get_args()
    cfg = load_config(args.c)
    log_file, checkpoint_dir = set_logging(cfg)
    sys.stdout = Logger(cfg, log_file)
    print("Log_directory : ", checkpoint_dir)
    shutil.copyfile(args.c, os.path.join(checkpoint_dir, "../config.yaml"))

    sys.exit(main(cfg))
