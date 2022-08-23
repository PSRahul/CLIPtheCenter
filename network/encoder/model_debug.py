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


def main():
    args = get_args()
    cfg = load_config(args.c)
    detection_model = DetectionModel(cfg)
