import albumentations as A
from torchvision import transforms


class GetAugementations():
    def __init__(self, cfg):
        self.cfg = cfg
        if (cfg["model"]["encoder"]["encoder_name"] == "ResNet18Model"):
            self.transform = self.get_resnet_transforms()
        if ("EfficientNet" in cfg["model"]["encoder"]["encoder_name"]):
            self.transform = self.get_efficientnet_transforms()

    def get_resnet_transforms(self):
        train_transform = A.Compose([
            A.RandomSizedBBoxSafeCrop(width=384, height=384),
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams
        (format='coco', min_area=1600, min_visibility=0.1, label_fields=['class_labels']))

        test_transform = A.Compose([
            A.Resize(384, 384),
        ], bbox_params=A.BboxParams
        (format='coco', min_area=1600, min_visibility=0.1, label_fields=['class_labels']))

        mask_transform = A.Compose([
            A.Resize(self.cfg["heatmap"]["output_dimension"], self.cfg["heatmap"]["output_dimension"]),
        ], bbox_params=A.BboxParams
        (format='coco', label_fields=['class_labels']))

        tensor_image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )]
        )

        return train_transform, test_transform, mask_transform, tensor_image_transforms

    def get_efficientnet_transforms(self):
        train_transform = A.Compose([
            A.Resize(320, 320),
            A.RandomSizedBBoxSafeCrop(width=300, height=300),
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams
        (format='coco', min_area=1600, min_visibility=0.1, label_fields=['class_labels']))

        test_transform = A.Compose([
            A.Resize(300, 300),
        ], bbox_params=A.BboxParams
        (format='coco', min_area=1600, min_visibility=0.1, label_fields=['class_labels']))

        mask_transform = A.Compose([
            A.Resize(self.cfg["heatmap"]["output_dimension"],
                     self.cfg["heatmap"]["output_dimension"]),
        ], bbox_params=A.BboxParams
        (format='coco', label_fields=['class_labels']))

        tensor_image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )]
        )

        return train_transform, test_transform, mask_transform, tensor_image_transforms
