# https://pytorch.org/vision/stable/_modules/torchvision/datasets/coco.html#CocoDetection

import os.path
import torch
import numpy as np
from pycocotools.coco import COCO

from PIL import Image

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
        self.train_transform, self.test_transform, self.mask_transform, self.tensor_image_transforms = get_augmentations.transform

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
            transform = self.train_transform
        else:
            transform = self.test_transform

        transformed = transform(image=image, bboxes=bounding_box_list, class_labels=class_list)
        image = transformed['image']
        bounding_box_list = transformed['bboxes']
        class_list = transformed['class_labels']
        return image, bounding_box_list, class_list

    def get_heatmap(self, image, bounding_box_list, class_list):
        transform = self.mask_transform
        transformed = transform(image=image, bboxes=bounding_box_list, class_labels=class_list)
        heatmap_image = transformed['image']
        heatmap_bounding_box_list = transformed['bboxes']
        heatmap_class_list = transformed['class_labels']
        return heatmap_image, heatmap_bounding_box_list, heatmap_class_list

    def generate_gaussian_radius(self, height, width):
        if self.cfg["heatmap"]["fix_radius"]:
            r = self.cfg["heatmap"]["fix_radius_value"]
        else:
            radius_scale = self.cfg["heatmap"]["radius_scaling"]
            r = np.sqrt(height ** 2 + width ** 2)
            r = r / radius_scale
            r = 1
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

    def create_heatmap_object(self, heatmap_bounding_box):

        bbox = np.array([heatmap_bounding_box[0], heatmap_bounding_box[1],
                         heatmap_bounding_box[0] + heatmap_bounding_box[2],
                         heatmap_bounding_box[1] + heatmap_bounding_box[3]],
                        dtype=np.float32)

        bbox_center = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        bbox_center_int = bbox_center.astype(np.int32)

        bbox_h, bbox_w = heatmap_bounding_box[3], heatmap_bounding_box[2]
        object_heatmap = self.generate_gaussian_output_map(bbox_h, bbox_w, bbox_center_int)
        object_offset = bbox_center - bbox_center_int

        return object_heatmap, bbox_center_int, object_offset

    def __getitem__(self, index):
        image_id = self.ids[index]
        image, bounding_box_list, class_list = self.get_transformed_image(index)
        heatmap_image, heatmap_bounding_box_list, heatmap_class_list = self.get_heatmap(image, bounding_box_list,
                                                                                        class_list)
        # image = image.transpose(2, 0, 1)
        # heatmap_image = heatmap_image.transpose(2, 0, 1)

        heatmap = np.zeros((self.cfg["heatmap"]["output_dimension"],
                            self.cfg["heatmap"]["output_dimension"]))

        bbox = np.zeros((self.cfg["max_objects_per_image"], 2))
        offset = np.zeros((self.cfg["max_objects_per_image"], 2))
        flattened_index = np.zeros(self.cfg["max_objects_per_image"])
        num_objects = 0
        for heatmap_bounding_box in heatmap_bounding_box_list:
            object_heatmap, bbox_center_int, object_offset = self.create_heatmap_object(heatmap_bounding_box)
            heatmap += object_heatmap
            bbox[num_objects] = heatmap_bounding_box[3], heatmap_bounding_box[2]
            offset[num_objects] = object_offset
            flattened_index[num_objects] = self.cfg["heatmap"]["output_dimension"] * bbox_center_int[1] + \
                                           bbox_center_int[0]
            num_objects += 1
            if (num_objects == self.cfg["max_objects_per_image"]):
                break
        heatmap = np.clip(heatmap, 0, 1.0)
        assert heatmap.max() <= 1.0
        batch_item = {}
        batch_item['image_id'] = torch.from_numpy(image_id)
        batch_item['image'] = self.tensor_image_transforms(image)
        batch_item['heatmap'] = torch.from_numpy(heatmap)
        batch_item['bbox'] = torch.from_numpy(bbox)
        batch_item['offset'] = torch.from_numpy(offset)
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
