# https://pytorch.org/vision/stable/_modules/torchvision/datasets/coco.html#CocoDetection

import os.path

import matplotlib.pyplot as plt
import torch
import numpy as np
from pycocotools.coco import COCO

from PIL import Image
from data.data_utils import get_gaussian_radius, get_gaussian_radius_centernet, generate_gaussian_peak

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

    def get_heatmap(self, image, bounding_box_list, class_list):
        transform = self.mask_transform
        transformed = transform(image=image, bboxes=bounding_box_list, class_labels=class_list)
        heatmap_image = transformed['image']
        heatmap_bounding_box_list = transformed['bboxes']
        heatmap_class_list = transformed['class_labels']
        return heatmap_image, heatmap_bounding_box_list, heatmap_class_list

    def generate_gaussian_output_map(self, h, w, bbox_center_int):
        # This will generate a gaussian map in the output dimension size
        object_heatmap = np.zeros((self.cfg["heatmap"]["output_dimension"],
                                   self.cfg["heatmap"]["output_dimension"]))

        gaussian_radius, gaussian_peak = generate_gaussian_peak(self.cfg, h, w)

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

    def create_heatmap_object(self, heatmap_bounding_box):
        # [x1,y1,w,h] -> [x1,y1,x1+w,y1+h]
        bbox = np.array([heatmap_bounding_box[0], heatmap_bounding_box[1],
                         heatmap_bounding_box[0] + heatmap_bounding_box[2],
                         heatmap_bounding_box[1] + heatmap_bounding_box[3]],
                        dtype=np.float32)
        # [x_center, y_center]
        bbox_center = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.int32)
        # [h,w]
        bbox_h, bbox_w = heatmap_bounding_box[3], heatmap_bounding_box[2]
        object_heatmap = self.generate_gaussian_output_map(bbox_h, bbox_w, bbox_center)
        # object_offset = bbox_center - bbox_center_int

        return object_heatmap, bbox_center

    def __getitem__(self, index):
        image_id = self.ids[index]
        path, image, bounding_box_list, class_list, original_image_shape = self.get_transformed_image(index)
        heatmap_image, heatmap_bounding_box_list, heatmap_class_list = self.get_heatmap(image, bounding_box_list,
                                                                                        class_list)
        if (self.cfg["debug"]):
            heatmap_image_np = heatmap_image
            plt.imsave(os.path.join("debug_outputs", str(index) + "_image.png"), heatmap_image_np)

        # image = image.transpose(2, 0, 1)
        # heatmap_image = heatmap_image.transpose(2, 0, 1)

        heatmap = np.zeros((self.cfg["heatmap"]["output_dimension"],
                            self.cfg["heatmap"]["output_dimension"]))

        bbox = np.zeros((self.cfg["max_objects_per_image"], 2))
        flattened_index = np.zeros(self.cfg["max_objects_per_image"])
        num_objects = 0
        for heatmap_bounding_box in heatmap_bounding_box_list:
            object_heatmap, bbox_center = self.create_heatmap_object(heatmap_bounding_box)
            np.maximum(heatmap, object_heatmap, out=heatmap)
            bbox[num_objects] = int(heatmap_bounding_box[2]), int(heatmap_bounding_box[3])
            flattened_index[num_objects] = self.cfg["heatmap"]["output_dimension"] * bbox_center[1] + \
                                           bbox_center[0]
            num_objects += 1
            if (num_objects == self.cfg["max_objects_per_image"]):
                break
        heatmap = np.clip(heatmap, 0, 1.0)
        if (self.cfg["debug"]):
            heatmap_np = heatmap
            plt.imsave(os.path.join("debug_outputs", str(index) + "_heatmap.png"), heatmap_np)  # cmap="Greys")

        assert heatmap.max() <= 1.0
        batch_item = {}
        batch_item['image_id'] = torch.tensor(image_id)
        batch_item['image'] = self.tensor_image_model_transforms(image)
        batch_item['image_clip'] = self.tensor_image_clip_transforms(image)
        batch_item['image_path'] = path
        # batch_item['original_image_shape'] = torch.from_numpy(original_image_shape)

        batch_item['heatmap'] = torch.from_numpy(heatmap)
        batch_item['bbox'] = torch.from_numpy(bbox)
        batch_item['flattened_index'] = torch.from_numpy(flattened_index)
        batch_item['num_objects'] = torch.tensor(num_objects)

        if self.cfg["dataset_class_debug"]:
            from PIL import Image
            for j, heatmap_class in enumerate(heatmap_class_list, 0):
                print(self.class_dict[int(heatmap_class)], heatmap_bounding_box_list[j])
            im = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
            im.save("debug/heatmap" + str(index) + ".jpeg")
            im = Image.fromarray(heatmap_image)
            im.save("debug/heatmap_image" + str(index) + ".jpeg")
            im = Image.fromarray(image)
            im.save("debug/image" + str(index) + ".jpeg")

        return batch_item

    def __len__(self):
        return len(self.ids)
