import albumentations as A

train_transform = A.Compose([
    A.RandomCrop(width=384, height=384),
    # A.HorizontalFlip(p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams
(format='coco', min_area=1600, min_visibility=0.1, label_fields=['class_labels']))

test_transform = A.Compose([
    A.Resize(384, 384),
], bbox_params=A.BboxParams
(format='coco', min_area=1600, min_visibility=0.1, label_fields=['class_labels']))

mask_transform = A.Compose([
    A.Resize(96, 96),
], bbox_params=A.BboxParams
(format='coco', min_area=1600, min_visibility=0.1, label_fields=['class_labels']))
