# https://pytorch.org/vision/stable/_modules/torchvision/datasets/coco.html#CocoDetection

import os.path

import numpy as np
from pycocotools.coco import COCO

from PIL import Image

from torchvision.datasets.vision import VisionDataset
from data.augmentations import train_transform, test_transform, mask_transform


class CocoDetection(VisionDataset):
    def __init__(
            self,
            root,
            annFile,
            train,
            cfg,
            transform=None,
            target_transform=None,
            transforms=None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.train = train
        self.cfg = cfg

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        return anns

    def get_transformed_image(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        image = np.array(image)
        anns = self._load_target(id)
        bounding_box_list = []
        class_list = []
        for ann in anns:
            bounding_box_list.append(ann['bbox'])
            class_list.append(ann['category_id'])

        if self.train and self.cfg["data"]["train_aug"]:
            transform = train_transform
        else:
            transform = test_transform

        transformed = transform(image=image, bboxes=bounding_box_list, class_labels=class_list)
        image = transformed['image']
        bounding_box_list = transformed['bboxes']
        class_list = transformed['class_labels']
        return image, bounding_box_list, class_list

    def get_heatmap(self, image, bounding_box_list, class_list):
        transform = mask_transform
        transformed = transform(image=image, bboxes=bounding_box_list, class_labels=class_list)
        heatmap_image = transformed['image']
        heatmap_bounding_box_list = transformed['bboxes']
        heatmap_class_list = transformed['class_labels']
        return heatmap_image, heatmap_bounding_box_list, heatmap_class_list

    def __getitem__(self, index):
        image, bounding_box_list, class_list = self.get_transformed_image(index)
        heatmap_image, heatmap_bounding_box_list, heatmap_class_list = self.get_heatmap(image, bounding_box_list,
                                                                                        class_list)

    def __len__(self):
        return len(self.ids)
