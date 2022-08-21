from data.dataset_class import CocoDetection
import os


class DataModule():
    def __init__(self, cfg):
        super().__init__()
        self.train_dataset = self.load_train_dataset(cfg)
        self.val_dataset = self.load_val_dataset(cfg)

    def load_train_dataset(self, cfg):
        root = cfg["data"]["train_data_root"]
        return CocoDetection(root=os.path.join(root, "data"),
                             annFile=os.path.join(root, "labels.json"))

    def load_val_dataset(self, cfg):
        root = cfg["data"]["val_data_root"]
        return CocoDetection(root=os.path.join(root, "data"),
                             annFile=os.path.join(root, "labels.json"))
