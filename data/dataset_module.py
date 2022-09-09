from data.dataset_class import CocoDetection
import os
import albumentations as A
import random
from torch.utils.data import DataLoader

import cv2
from matplotlib import pyplot as plt


class DataModule():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def load_train_dataset(self):
        root = self.cfg["data"]["train_data_root"]
        return CocoDetection(root=os.path.join(root, "data"),
                             annFile=os.path.join(root, "labels.json"), train=True,
                             cfg=self.cfg)

    def load_val_dataset(self):
        root = self.cfg["data"]["val_data_root"]
        return CocoDetection(root=os.path.join(root, "data"),
                             annFile=os.path.join(root, "labels.json"), train=False,
                             cfg=self.cfg)

    def load_test_dataset(self):
        root = self.cfg["data"]["test_data_root"]
        return CocoDetection(root=os.path.join(root, "data"),
                             annFile=os.path.join(root, "labels.json"), train=False,
                             cfg=self.cfg)

    def load_train_dataloader(self):
        return DataLoader(self.load_train_dataset(), batch_size=self.cfg["data"]["train_batch_size"], shuffle=True)

    def load_val_dataloader(self):
        return DataLoader(self.load_val_dataset(), batch_size=self.cfg["data"]["val_batch_size"], shuffle=True)

    def load_test_dataloader(self):
        return DataLoader(self.load_test_dataset(), batch_size=self.cfg["data"]["test_batch_size"], shuffle=True)
