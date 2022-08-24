import torch.nn as nn
import torchvision
from torchinfo import summary
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
import sys
import torch
from network.model_utils import set_parameter_requires_grad


# from efficientnet_pytorch import EfficientNet


class EfficientNetB0Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        # self.model = nn.Sequential(*list(self.model.children())[:-3])

        self.model = set_parameter_requires_grad(
            self.model, cfg["model"]["encoder"]["freeze_params"]
        )

        # self.model.fc = nn.Linear(512, cfg["model"]["num_classes"])

    def forward(self, x):
        return self.model.forward(x)

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 3, 300, 300))

    def get_test_transforms(self):
        test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return test_transforms

    def get_train_transforms(self):
        train_transforms = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.3, hue=0.3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return train_transforms
