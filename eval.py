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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="eval_config.yaml")
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


def main(cfg):
    if (cfg["evaluation"]["run_inference"]):
        detection_model = DetectionModel(cfg)
        print(detection_model.print_details())

        coco_dataset = DataModule(cfg)
        eval_metrics = EvalMetrics(cfg=cfg, checkpoint_dir=checkpoint_dir, model=detection_model,
                                   test_dataloader=coco_dataset.load_val_dataloader())
        prediction_save_path = eval_metrics.eval()
    else:
        prediction_save_path = cfg["evaluation"]["prediction_save_path"]
        print("Predictions are loaded from", prediction_save_path)

    prediction = np.load(prediction_save_path)
    groundtruth = os.path.join(cfg["evaluation"]["test_data_root"], "labels.json")
    COCORunner(groundtruth=groundtruth, prediction=prediction)


if __name__ == "__main__":
    args = get_args()
    cfg = load_config(args.c)
    log_file, checkpoint_dir = set_logging(cfg)
    sys.stdout = Logger(cfg, log_file)
    print("Log_directory : ", checkpoint_dir)
    shutil.copyfile(args.c, os.path.join(checkpoint_dir, "configs/config.yaml"))

    sys.exit(main(cfg))
