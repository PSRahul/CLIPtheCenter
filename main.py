import yaml
from yaml.loader import SafeLoader
import argparse
import logging
import os
from datetime import datetime
import sys
import pathlib


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
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def main():
    args = get_args()
    cfg = load_config(args.c)
    now = datetime.now()
    date_save_string = now.strftime("%d%m%Y_%H%M")
    checkpoint_dir = os.path.join(
        "/home/psrahul/MasterThesis/repo/Phase3/CLIPandDetect/",
        cfg["logging"]["checkpoint_dir"],
        date_save_string,
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not cfg["test_model"]:
        log_file = os.path.join(checkpoint_dir, "log.log")
        sys.stdout = Logger(cfg, log_file)

    pytorch_model_name = globals()[cfg["model"]["name"]]
    pytorch_model = pytorch_model_name(cfg)
    logger_pytorch = logging.getLogger("pytorch_lightning")


    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger_pytorch.addHandler(
        logging.FileHandler(os.path.join(checkpoint_dir, "trainer.log"))
    )
