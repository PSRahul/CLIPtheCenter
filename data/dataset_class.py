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

    def generate_gaussian_radius(self, height, width):
        radius_scale = self.cfg["heatmap"]["radius_scaling"]
        r = np.sqrt(height ** 2 + width ** 2)
        r = r / radius_scale
        return int(r)

    def generate_gaussian_peak(self, height, width):
        # This will only generate a matrix of size [diameter, diameter] that has gaussian distribution
        gaussian_radius = self.generate_gaussian_radius(height, width)
        gaussian_diameter = 2 * gaussian_radius + 1
        sigma = gaussian_diameter / 6
        m, n = [(ss - 1.) / 2. for ss in (gaussian_diameter, gaussian_diameter)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        gaussian_peak = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gaussian_peak[gaussian_peak < 1e-7 * gaussian_peak.max()] = 0
        return gaussian_radius, gaussian_peak

    def generate_gaussian_output_map(self, h, w, bbox_center_int):
        # This will generate a gaussian map in the output dimension size
        object_heatmap = np.zeros((self.cfg["heatmap"]["output_dimension"],
                                   self.cfg["heatmap"]["output_dimension"]))

        gaussian_radius, gaussian_peak = self.generate_gaussian_peak(h, w)

        output_height = output_width = self.cfg["heatmap"]["output_dimension"]

        left, right = min(bbox_center_int[0], gaussian_radius), min(output_width - bbox_center_int[0],
                                                                    gaussian_radius + 1)
        top, bottom = min(bbox_center_int[1], gaussian_radius), min(output_height - bbox_center_int[1],
                                                                    gaussian_radius + 1)

        masked_object_heatmap = object_heatmap[bbox_center_int[1] - top:bbox_center_int[1] + bottom,
                                bbox_center_int[0] - left:bbox_center_int[0] + right]
        masked_gaussian_peak = gaussian_peak[gaussian_radius - top:gaussian_radius + bottom,
                               gaussian_radius - left:gaussian_radius + right]

        np.maximum(masked_object_heatmap, masked_gaussian_peak, out=masked_object_heatmap)
        return object_heatmap

    def create_heatmap(self, heatmap_image, heatmap_bounding_box_list, heatmap_class_list):
        heatmap = np.zeros((1, self.cfg["heatmap"]["output_dimension"],
                            self.cfg["heatmap"]["output_dimension"]))

        for coco_bbox in heatmap_bounding_box_list:
            bbox = np.array([coco_bbox[0], coco_bbox[1],
                             coco_bbox[0] + coco_bbox[2], coco_bbox[1] + coco_bbox[3]],
                            dtype=np.float32)

            bbox_center = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            bbox_center_int = bbox_center.astype(np.int32)

            bbox_h, bbox_w = coco_bbox[3], coco_bbox[2]
            object_heatmap = self.generate_gaussian_output_map(bbox_h, bbox_w, bbox_center_int)
            print("breakpoint")

    def __getitem__(self, index):
        image, bounding_box_list, class_list = self.get_transformed_image(index)
        heatmap_image, heatmap_bounding_box_list, heatmap_class_list = self.get_heatmap(image, bounding_box_list,
                                                                                        class_list)
        image = image.transpose(2, 0, 1)
        heatmap_image = heatmap_image.transpose(2, 0, 1)

        self.create_heatmap(heatmap_image, heatmap_bounding_box_list, heatmap_class_list)

        print("Debug")

    def __len__(self):
        return len(self.ids)
