import argparse
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
import torchvision
import yaml
from yaml.loader import SafeLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="metrics_config.yaml")
    args = parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    return config


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


def main(cfg):
    gt = np.load(cfg["gt"])
    predictions = np.load(cfg["predictions"])
    predictions_class = predictions["predictions_class"]
    predictions_bbox = predictions["predictions_bbox"]
    gt_class = gt["gt_class"]
    gt_bbox = gt["gt_bbox"]
    print("benc")
    score = torchvision.ops.complete_box_iou(torch.tensor(predictions_bbox), torch.tensor(gt_bbox))
    print(score)


if __name__ == "__main__":
    args = get_args()
    cfg = load_config(args.c)
    _, checkpoint_dir = set_logging(cfg)
    shutil.copyfile(args.c, os.path.join(checkpoint_dir, args.c))

    sys.exit(main(cfg))
