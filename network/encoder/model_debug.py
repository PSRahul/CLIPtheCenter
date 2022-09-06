import argparse

import yaml
from yaml.loader import SafeLoader

from network.model_builder.arch1 import DetectionModel


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
