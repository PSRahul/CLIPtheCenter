import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from network.model_utils import set_parameter_requires_grad


class ResNet18Model(nn.Module):
    def __init__(self, cfg):

        # weights = ResNet18_Weights.DEFAULT
        pretrained = cfg["model"]["encoder"]["use_pretrained"]
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet18", weights=weights
        )
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model = set_parameter_requires_grad(
            self.model, cfg["model"]["encoder"]["freeze_params"]
        )

        # self.model.fc = nn.Linear(512, cfg["model"]["num_classes"])

    def forward(self, x):
        return self.model.forward(x)

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 3, 384, 384))

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
