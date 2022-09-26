import albumentations as A
from torchvision import transforms
import cv2


class GetAugementations():
    def __init__(self, cfg):
        self.cfg = cfg
        if (cfg["model"]["encoder"]["encoder_name"] == "ResNet18Model"):
            self.transform = self.get_resnet_transforms()
        elif ("EfficientNet" in cfg["model"]["encoder"]["encoder_name"]):
            self.transform = self.get_efficientnet_transforms()
        else:
            self.transform = self.get_efficientnet_transforms()

    def get_resnet_transforms(self):
        train_transform = A.Compose([
            A.Resize(self.cfg["data"]["input_dimension"], self.cfg["data"]["input_dimension"],
                     interpolation=cv2.INTER_CUBIC),
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams
        (format='coco', label_fields=['class_labels']))

        test_transform = A.Compose([
            A.Resize(self.cfg["data"]["input_dimension"], self.cfg["data"]["input_dimension"],
                     interpolation=cv2.INTER_CUBIC),
        ], bbox_params=A.BboxParams
        (format='coco', label_fields=['class_labels']))

        mask_transform = A.Compose([
            A.Resize(self.cfg["heatmap"]["output_dimension"],
                     self.cfg["heatmap"]["output_dimension"], interpolation=cv2.INTER_CUBIC),
        ], bbox_params=A.BboxParams
        (format='coco', label_fields=['class_labels']))

        tensor_image_model_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )]
        )
        tensor_image_clip_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
                )]
        )
        return train_transform, test_transform, mask_transform, tensor_image_model_transforms, tensor_image_clip_transforms

    def get_efficientnet_transforms(self):
        train_transform = A.Compose([
            A.Resize(self.cfg["data"]["input_dimension"], self.cfg["data"]["input_dimension"],
                     interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams
        (format='coco', label_fields=['class_labels']))

        test_transform = A.Compose([
            A.Resize(self.cfg["data"]["input_dimension"], self.cfg["data"]["input_dimension"],
                     interpolation=cv2.INTER_CUBIC),
        ], bbox_params=A.BboxParams
        (format='coco', label_fields=['class_labels']))

        mask_transform = A.Compose([
            A.Resize(self.cfg["heatmap"]["output_dimension"],
                     self.cfg["heatmap"]["output_dimension"], interpolation=cv2.INTER_CUBIC),
        ], bbox_params=A.BboxParams
        (format='coco', label_fields=['class_labels']))

        tensor_image_model_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )]
        )
        tensor_image_clip_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
                )]
        )
        return train_transform, test_transform, mask_transform, tensor_image_model_transforms, tensor_image_clip_transforms
