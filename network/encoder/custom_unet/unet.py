import torch.nn as nn
import torch


class CustomUnetModel(nn.Module):

    def __init__(self):
        super().__init__()

        filter_size_max = 512

        self.drop = nn.Dropout(p=0)

        self.relu = nn.PReLU()
        self.max = nn.MaxPool2d(2, ceil_mode=True)
        self.bn1_1 = nn.BatchNorm2d(filter_size_max // 16)
        self.bn1_2 = nn.BatchNorm2d(filter_size_max // 16)
        self.bn2_2 = nn.BatchNorm2d(filter_size_max // 8)
        self.bn2_1 = nn.BatchNorm2d(filter_size_max // 8)
        self.bn3_1 = nn.BatchNorm2d(filter_size_max // 4)
        self.bn3_2 = nn.BatchNorm2d(filter_size_max // 4)
        self.bn3_3 = nn.BatchNorm2d(filter_size_max // 4)
        self.bn4_1 = nn.BatchNorm2d(filter_size_max // 8)
        self.bn4_2 = nn.BatchNorm2d(filter_size_max // 8)
        self.bn5_1 = nn.BatchNorm2d(filter_size_max // 16)
        self.bn5_2 = nn.BatchNorm2d(filter_size_max // 16)
        self.bn6 = nn.BatchNorm2d(filter_size_max // 32)

        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_1 = nn.Conv2d(3, filter_size_max // 16, 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(
            filter_size_max // 16, filter_size_max // 16, 3, padding=(1, 1))

        self.conv2_1 = nn.Conv2d(
            filter_size_max // 16, filter_size_max // 8, 3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(
            filter_size_max // 8, filter_size_max // 8, 3, padding=(1, 1))

        self.conv3_1 = nn.Conv2d(
            filter_size_max // 8, filter_size_max // 4, 3, padding=(1, 1))
        self.conv3_2 = nn.Conv2d(
            filter_size_max // 4, filter_size_max // 4, 3, padding=(1, 1))

        self.conv4_1 = nn.Conv2d(
            filter_size_max // 4, filter_size_max // 2, 3, padding=(1, 1))
        self.conv4_2 = nn.Conv2d(
            filter_size_max // 2, filter_size_max // 2, 3, padding=(1, 1))

        self.conv4_3 = nn.Conv2d(
            filter_size_max, filter_size_max // 2, 3, padding=(1, 1))
        self.conv4_4 = nn.Conv2d(filter_size_max // 2, filter_size_max // 4, 3)

        self.conv3_3 = nn.Conv2d(
            filter_size_max // 2, filter_size_max // 4, 3, padding=(1, 1))
        self.conv3_4 = nn.Conv2d(
            filter_size_max // 4, filter_size_max // 8, 3, padding=(1, 1))

        self.conv2_3 = nn.Conv2d(
            filter_size_max // 4, filter_size_max // 8, 3, padding=(1, 1))
        self.conv2_4 = nn.Conv2d(
            filter_size_max // 8, filter_size_max // 16, 3, padding=(1, 1))

        self.conv1_3 = nn.Conv2d(
            filter_size_max // 8, filter_size_max // 16, 3, padding=(1, 1))
        self.conv1_4 = nn.Conv2d(
            filter_size_max // 16, filter_size_max // 32, 3, padding=(1, 1))

        self.conv1_5 = nn.Conv2d(filter_size_max // 32, 16, 3, padding=(1, 1))

    def forward(self, x):
        x = self.bn1_1((self.relu(self.conv1_1(x))))
        x1 = self.bn1_2(self.relu(self.conv1_2(x)))
        x = self.drop(x)
        x = self.max(x1)

        x = self.bn2_1(self.relu(self.conv2_1(x)))
        x2 = self.bn2_2(self.relu(self.conv2_2(x)))
        x = self.drop(x)
        x = self.max(x2)

        x = self.bn3_1(self.relu(self.conv3_1(x)))
        x3 = self.bn3_2(self.relu(self.conv3_2(x)))
        x = self.drop(x)
        x = self.max(x3)

        x = self.up(x)
        x = torch.cat((x3, x), dim=1)
        x = self.bn3_3(self.relu(self.conv3_3(x)))
        x = self.bn4_1(self.relu(self.conv3_4(x)))

        x = self.drop(x)
        x = self.up(x)

        x = torch.cat((x2, x), dim=1)
        x = self.bn4_2(self.relu(self.conv2_3(x)))
        x = self.bn5_1(self.relu(self.conv2_4(x)))

        x = self.drop(x)
        x = self.up(x)

        x = torch.cat((x1, x), dim=1)
        x = self.bn5_2(self.relu(self.conv1_3(x)))
        x = self.bn6(self.relu(self.conv1_4(x)))

        x = self.conv1_5(x)

        return x
