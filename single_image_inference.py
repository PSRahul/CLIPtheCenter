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
from torch.utils.data import DataLoader
from visualise.single_inference_module import SingleInferenceModel


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


def main(cfg):
    detection_model = DetectionModel(cfg)
    coco_dataset = DataModule(cfg)
    test_dataset = coco_dataset.load_val_dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    for batch_ndx, sample in enumerate(test_dataloader):
        batch = sample
        break
    print(batch)
    single_inference_model = SingleInferenceModel(cfg=cfg, model=detection_model)

    trainer.train()


if __name__ == "__main__":
    args = get_args()
    cfg = load_config(args.c)

    sys.exit(main(cfg))
