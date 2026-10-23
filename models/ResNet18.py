import torch
from torch import nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 1),
            kernel_size=(3, 3),
            padding=1,
        )

        self.norm_1 = nn.BatchNorm2d(
            num_features=out_channels,
        )

        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=(1, 1),
            kernel_size=(3, 3),
            padding=1,
        )

        self.norm_2 = nn.BatchNorm2d(
            num_features=out_channels,
        )

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
            )

        else:
            self.shortcut = nn.Identity()

    def forward(self, input: torch.Tensor):
        out = self.relu(self.norm_1(self.conv_1(input)))
        out = self.norm_2(self.conv_2(out))
        out = out + self.shortcut(input)

        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            stride=(2, 2),
            kernel_size=(7, 7),
            padding=3,
        )

        self.norm_1 = nn.BatchNorm2d(
            num_features=64,
        )

        self.pooling_1 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            ceil_mode=True,
        )

        self.relu = nn.ReLU(inplace=True)

        self.block_1a = ResnetBlock(64, 64)
        self.block_1b = ResnetBlock(64, 64)

        self.pooling_2 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=2,
            ceil_mode=True,
        )

        self.block_2a = ResnetBlock(64, 128)
        self.block_2b = ResnetBlock(128, 128)

        self.pooling_3 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=2,
            ceil_mode=True,
        )

        self.block_3a = ResnetBlock(128, 256)
        self.block_3b = ResnetBlock(256, 256)

        self.pooling_4 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=2,
            ceil_mode=True,
        )

        self.block_4a = ResnetBlock(256, 512)
        self.block_4b = ResnetBlock(512, 512)

        self.pooling_5 = nn.AvgPool2d(
            kernel_size=(7, 7),
            stride=(1, 1),
        )

        self.fc = nn.Linear(
            in_features=512,
            out_features=21,
        )

    def forward(self, images: torch.Tensor):
        features = self.conv_1(images)
        features = self.norm_1(features)
        features = self.pooling_1(features)
        features = self.relu(features)

        features = self.block_1a(features)
        features = self.block_1b(features)
        features = self.pooling_2(features)

        features = self.block_2a(features)
        features = self.block_2b(features)
        features = self.pooling_3(features)

        features = self.block_3a(features)
        features = self.block_3b(features)
        features = self.pooling_4(features)

        features = self.block_4a(features)
        features = self.block_4b(features)
        features = self.pooling_5(features)

        features = features.squeeze(-1).squeeze(-1)
        
        output = self.fc(features)

        return output