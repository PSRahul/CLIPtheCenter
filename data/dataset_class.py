# https://pytorch.org/vision/stable/_modules/torchvision/datasets/coco.html#CocoDetection

import os.path

import matplotlib.pyplot as plt
import torch
import numpy as np
from pycocotools.coco import COCO

from PIL import Image
from data.data_utils import create_heatmap_object

from torchvision.datasets.vision import VisionDataset
# from data.augmentations import train_transform, test_transform, mask_transform, tensor_image_transforms
from data.augmentations import GetAugementations


def get_class_dict():
    class_dict = {0: "aeroplane",
                  1: "bicycle",
                  2: "bird",
                  3: "boat",
                  4: "bottle",
                  5: "bus",
                  6: "car",
                  7: "cat",
                  8: "chair",
                  9: "cow",
                  10: "diningtable",
                  11: "dog",
                  12: "horse",
                  13: "motorbike",
                  14: "person",
                  15: "pottedplant",
                  16: "sheep",
                  17: "sofa",
                  18: "train",
                  19: "tvmonitor"}
    return class_dict


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
        self.class_dict = get_class_dict()
        get_augmentations = GetAugementations(cfg)
        self.train_transform, self.test_transform, self.mask_transform, self.tensor_image_model_transforms, self.tensor_image_clip_transforms = get_augmentations.transform

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return path, Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        return anns

    def get_transformed_image(self, index):
        id = self.ids[index]
        path, image = self._load_image(id)
        image = np.array(image)
        original_image_shape = np.array(image.shape[0:2])
        anns = self._load_target(id)
        bounding_box_list = []
        class_list = []
        for ann in anns:
            bounding_box_list.append(ann['bbox'])
            class_list.append(ann['category_id'])

        if self.train and self.cfg["data"]["train_aug"]:
            transform = self.train_transform
        else:
            transform = self.test_transform

        transformed = transform(image=image, bboxes=bounding_box_list, class_labels=class_list)
        image = transformed['image']
        bounding_box_list = transformed['bboxes']
        class_list = transformed['class_labels']
        return path, image, bounding_box_list, class_list, original_image_shape

    def get_heatmap_sized_image(self, image, bounding_box_list, class_list):
        transform = self.mask_transform
        transformed = transform(image=image, bboxes=bounding_box_list, class_labels=class_list)
        heatmap_sized_image = transformed['image']
        heatmap_sized_bounding_box_list = transformed['bboxes']
        heatmap_sized_class_list = transformed['class_labels']
        return heatmap_sized_image, heatmap_sized_bounding_box_list, heatmap_sized_class_list

    def __getitem__(self, index):
        image_id = self.ids[index]
        path, image, bounding_box_list, class_list, original_image_shape = self.get_transformed_image(index)
        heatmap_sized_image, heatmap_sized_bounding_box_list, heatmap_sized_class_list = self.get_heatmap_sized_image(
            image,
            bounding_box_list,
            class_list)

        # image = image.transpose(2, 0, 1)
        # heatmap_image = heatmap_image.transpose(2, 0, 1)

        center_heatmap = np.zeros((self.cfg["heatmap"]["output_dimension"],
                                   self.cfg["heatmap"]["output_dimension"]))
        bbox_heatmap = np.zeros((2, self.cfg["heatmap"]["output_dimension"],
                                 self.cfg["heatmap"]["output_dimension"]))

        bbox = np.zeros((self.cfg["max_objects_per_image"], 2))
        flattened_index = np.zeros(self.cfg["max_objects_per_image"])
        num_objects = 0
        for heatmap_sized_bounding_box in heatmap_sized_bounding_box_list:
            center_heatmap_i, bbox_heatmap_i, bbox_center = create_heatmap_object(self.cfg, heatmap_sized_bounding_box)
            np.maximum(center_heatmap, center_heatmap_i, out=center_heatmap)
            np.maximum(bbox_heatmap, bbox_heatmap_i, out=bbox_heatmap)
            bbox[num_objects] = int(heatmap_sized_bounding_box[2]), int(heatmap_sized_bounding_box[3])
            flattened_index[num_objects] = self.cfg["heatmap"]["output_dimension"] * bbox_center[1] + \
                                           bbox_center[0]
            num_objects += 1
            if (num_objects == self.cfg["max_objects_per_image"]):
                break

        center_heatmap = center_heatmap.astype(np.int)
        bbox_heatmap = bbox_heatmap.astype(np.int)
        if (self.cfg["debug"]):
            center_heatmap_np = center_heatmap
            plt.imsave(os.path.join("debug_outputs", str(index) + "_center_heatmap.png"), center_heatmap_np,
                       cmap="Greys")
            bbox_heatmap_np = bbox_heatmap[0, :, :]
            plt.imsave(os.path.join("debug_outputs", str(index) + "_bbox_heatmap_width.png"), bbox_heatmap_np,
                       cmap="Greys")
            bbox_heatmap_np = bbox_heatmap[1, :, :]
            plt.imsave(os.path.join("debug_outputs", str(index) + "_bbox_heatmap_height.png"), bbox_heatmap_np,
                       cmap="Greys")
            heatmap_sized_image_np = heatmap_sized_image
            plt.imsave(os.path.join("debug_outputs", str(index) + "_image.png"), heatmap_sized_image_np)

        batch_item = {}
        batch_item['image_id'] = torch.tensor(image_id)
        batch_item['image'] = self.tensor_image_model_transforms(image)
        batch_item['image_clip'] = self.tensor_image_clip_transforms(image)
        batch_item['image_path'] = path
        # batch_item['original_image_shape'] = torch.from_numpy(original_image_shape)
        batch_item['heatmap_sized_bounding_box_list'] = torch.from_numpy(np.hstack((image_id, (
            np.array(heatmap_sized_bounding_box)))))
        batch_item['center_heatmap'] = torch.from_numpy(center_heatmap)
        batch_item['bbox_heatmap'] = torch.from_numpy(bbox_heatmap)
        batch_item['bbox'] = torch.from_numpy(bbox)
        batch_item['flattened_index'] = torch.from_numpy(flattened_index)
        batch_item['num_objects'] = torch.tensor(num_objects)
        if (self.cfg["debug"]):
            groundtruth_bbox_np = batch_item["bbox_heatmap"].detach().cpu().numpy()
            groundtruth_bbox_np = groundtruth_bbox_np[0, :, :]
            print("breakpoint")

        return batch_item

    def __len__(self):
        return len(self.ids)
