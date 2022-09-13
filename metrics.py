import argparse
import copy
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
from post_process.utils import resize_predictions_image_size, assign_classes
from post_process.visualise import visualise_bbox


# matplotlib.use('Agg')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="configs/metrics.yaml")
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


def get_groundtruths(dataset, show_image=False):
    gt = np.empty((0, 7))
    for index in range(len(dataset)):
        image_id = dataset.ids[index]
        image, anns = dataset[index]
        image = np.array(image)
        bounding_box_list = []
        class_list = []
        for ann in anns:
            bounding_box_list.append(ann['bbox'])
            class_list.append(ann['category_id'])

        if (show_image):
            bbox = bounding_box_list[0]
            bbox = [int(x) for x in bbox]
            print(bbox)
            image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
            plt.imshow(image)
            plt.show()
            break
        image_id = np.array(image_id)
        bounding_box_list = np.array(bounding_box_list)
        image_id_list = np.ones((len(class_list), 1)) * image_id
        scores_list = np.ones((len(class_list), 1))
        class_list = np.array(class_list).reshape((len(class_list), 1))
        # ["image_id", "bbox_y", "bbox_x", "w", "h", "score", "class_label"]
        if (len(bounding_box_list != 0)):
            gt_idx = np.hstack((image_id_list, bounding_box_list, scores_list, class_list))
            gt = np.vstack((gt, gt_idx))
    return gt


def main(cfg):
    dataset_root = cfg["data"]["root"]
    dataset = CocoDetection(root=os.path.join(dataset_root, "data"),
                            annFile=os.path.join(dataset_root, "labels.json"))
    clip_embedding = np.load(cfg["clip_embedding_path"])

    if (cfg["use_metric_data_path"]):
        print("Loading data from ", cfg["metric_data_path"])
        data = np.load(cfg["metric_data_path"])
        gt = data["gt"]
        prediction = data["prediction"]
        prediction_with_nms = data["prediction_with_nms"]
        prediction_with_nms_resized = data["prediction_with_nms_resized"]
    else:
        gt = get_groundtruths(dataset)
        prediction = np.load(cfg["prediction_path"])
        if cfg["perform_nms"]:
            prediction_with_nms = perform_nms(cfg, prediction)
        else:
            prediction_with_nms = prediction
        print("Resizing Predictions")
        prediction_with_nms_resized = resize_predictions_image_size(cfg, dataset, copy.deepcopy(prediction_with_nms))
        print("Metric Data saved at ", os.path.join(checkpoint_dir, "data.npz"))
        np.savez(os.path.join(checkpoint_dir, "data.npz"), gt=gt, prediction=prediction,
                 prediction_with_nms=prediction_with_nms,
                 prediction_with_nms_resized=prediction_with_nms_resized)

    print("GroundTruth Shape", gt.shape)
    print("Prediction Shape", prediction.shape)
    print("Prediction with NMS Shape", prediction_with_nms.shape)

    prediction_with_nms_resized = assign_classes(clip_embedding, prediction_with_nms_resized)
    # calculate_torchmetrics_mAP(gt, prediction_with_nms_resized)

    calculate_coco_result(gt=os.path.join(dataset_root, "labels.json"), prediction=prediction_with_nms_resized,
                          image_index_only=False, image_index=6)
    for id in range(1, 4):
        visualise_bbox(cfg=cfg, dataset=dataset, id=id, gt=gt, pred=prediction_with_nms_resized, draw_gt=True,
                       draw_pred=True,
                       resize_image_to_output_shape=False)
        visualise_bbox(cfg=cfg, dataset=dataset, id=id, gt=gt, pred=prediction_with_nms, draw_gt=False,
                       draw_pred=True,
                       resize_image_to_output_shape=True)


if __name__ == "__main__":
    args = get_args()
    cfg = load_config(args.c)
    log_file, checkpoint_dir = set_logging(cfg)
    sys.stdout = Logger(cfg, log_file)
    print("Log_directory : ", checkpoint_dir)
    shutil.copyfile(args.c, os.path.join(checkpoint_dir, "config.yaml"))

    sys.exit(main(cfg))
