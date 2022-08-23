import yaml
from yaml.loader import SafeLoader
import argparse
import logging
import os
from datetime import datetime
import sys
import pathlib
from network.model_builder import DetectionModel
from data.dataset_module import DataModule
from tqdm import tqdm
from trainer.trainer_module import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="config.yaml")
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


def main():
    args = get_args()
    cfg = load_config(args.c)
    log_file, checkpoint_dir = set_logging(cfg)
    sys.stdout = Logger(cfg, log_file)
    print("Log_directory : ", checkpoint_dir)
    detection_model = DetectionModel(cfg)
    coco_dataset = DataModule(cfg)
    trainer = Trainer(cfg=cfg, checkpoint_dir=checkpoint_dir, model=detection_model,
                      train_dataloader=coco_dataset.load_train_dataloader(),
                      val_dataloader=coco_dataset.load_val_dataloader())

    trainer.train()


if __name__ == "__main__":
    sys.exit(main())
