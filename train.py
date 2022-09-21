import argparse
import os
import shutil
import sys
from datetime import datetime

import yaml
from yaml.loader import SafeLoader

from data.dataset_module import DataModule
from network.model_builder.EffcientNet_ConvT import EfficientnetConv2DTModel
from network.model_builder.SMP import SMPModel
from network.model_builder.ResNet import ResNetModel
from trainer.EfficientnetConv2DT_trainer_module import EfficientnetConv2DTTrainer
from trainer.SMP_trainer_module import SMPTrainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="configs/train_smp.yaml")
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
        "/home/psrahul/MasterThesis/repo/Phase4/CLIPandDetect/",
        cfg["logging"]["checkpoint_dir"], cfg["smp"]["model"] + cfg["smp"]["encoder_name"],
        date_save_string,
    )
    print(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, "log.log")
    return log_file, checkpoint_dir


def main(cfg):
    detection_model = SMPModel(cfg)
    print(detection_model.print_details())
    coco_dataset = DataModule(cfg)
    trainer = SMPTrainer(cfg=cfg, checkpoint_dir=checkpoint_dir, model=detection_model,
                         train_dataloader=coco_dataset.load_train_dataloader(),
                         val_dataloader=coco_dataset.load_val_dataloader(),
                         test_dataloader=coco_dataset.load_val_dataloader())
    if (cfg["train"]):
        trainer.train()
    if (cfg["test"]):
        trainer.test()


if __name__ == "__main__":
    args = get_args()
    cfg = load_config(args.c)
    log_file, checkpoint_dir = set_logging(cfg)
    sys.stdout = Logger(cfg, log_file)
    print("Log_directory : ", checkpoint_dir)
    shutil.copyfile(args.c, os.path.join(checkpoint_dir, "config.yaml"))

    sys.exit(main(cfg))
