import torch
from torch import nn
import torch.nn.functional as F


class inception(nn.Module):
    def __init__(self, in_channels, oc_1, rc_2, oc_2, rc_3, oc_3, oc_4):
        super().__init__()

        self.branch1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=oc_1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=rc_2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
            ),
            nn.Conv2d(
                in_channels=rc_2,
                out_channels=oc_2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=rc_3,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
            ),
            nn.Conv2d(
                in_channels=rc_3,
                out_channels=oc_3,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=2,
            ),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                ceil_mode=True,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=oc_4,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
            ),
        )

    def forward(self, input: torch.Tensor):
        out1 = self.branch1(input)
        out2 = self.branch2(input)
        out3 = self.branch3(input)
        out4 = self.branch4(input)

        return torch.cat([out1, out2, out3, out4], dim=1)



class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=3,
        )

        self.pooling_1 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            ceil_mode=True,
        )

        self.conv_2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

        self.conv_3 = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1
        )

        self.pooling_2 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            ceil_mode=True,
        )

        self.incep_3a = inception(
            in_channels=192,
            oc_1=64,
            rc_2=96,
            oc_2=128,
            rc_3=16,
            oc_3=32,
            oc_4=32,
        )

        self.incep_3b = inception(
            in_channels=256,
            oc_1=128,
            rc_2=128,
            oc_2=192,
            rc_3=32,
            oc_3=96,
            oc_4=64,
        )

        self.pooling_3 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            ceil_mode=True,
        )

        self.incep_4a = inception(
            in_channels=480,
            oc_1=192,
            rc_2=96,
            oc_2=208,
            rc_3=16,
            oc_3=48,
            oc_4=64,
        )

        self.incep_4b = inception(
            in_channels=512,
            oc_1=160,
            rc_2=112,
            oc_2=224,
            rc_3=24,
            oc_3=64,
            oc_4=64,
        )

        self.incep_4c = inception(
            in_channels=512,
            oc_1=128,
            rc_2=128,
            oc_2=256,
            rc_3=24,
            oc_3=64,
            oc_4=64,
        )

        self.incep_4d = inception(
            in_channels=512,
            oc_1=112,
            rc_2=144,
            oc_2=288,
            rc_3=32,
            oc_3=64,
            oc_4=64,
        )

        self.incep_4e = inception(
            in_channels=528,
            oc_1=256,
            rc_2=160,
            oc_2=320,
            rc_3=32,
            oc_3=128,
            oc_4=128,
        )

        self.pooling_4 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            ceil_mode=True,
        )

        self.incep_5a = inception(
            in_channels=832,
            oc_1=256,
            rc_2=160,
            oc_2=320,
            rc_3=32,
            oc_3=128,
            oc_4=128,
        )

        self.incep_5b = inception(
            in_channels=832,
            oc_1=384,
            rc_2=192,
            oc_2=384,
            rc_3=48,
            oc_3=128,
            oc_4=128,
        )

        self.pooling_5 = nn.AvgPool2d(
            kernel_size=(7, 7),
            stride=(1, 1),
        )

        self.drop_out = nn.Dropout(p=0.4)

        self.fc = nn.Linear(
            in_features=1024,
            out_features=1000,
        )

        self.fc2 = nn.Linear(
            in_features=1000,
            out_features=21,
        )

        self.output = nn.Softmax(dim=1)


    def forward(self, images: torch.Tensor):
        images = images.unsqueeze(1)

        features = self.conv_1(images)
        features = self.pooling_1(features)

        features = self.conv_2(features)
        features = self.conv_3(features)
        features = self.pooling_2(features)

        features = self.incep_3a(features)
        features = self.incep_3b(features)
        features = self.pooling_3(features)

        features = self.incep_4a(features)
        features = self.incep_4b(features)
        features = self.incep_4c(features)
        features = self.incep_4d(features)
        features = self.incep_4e(features)
        features = self.pooling_4(features)

        features = self.incep_5a(features)
        features = self.incep_5b(features)

        features = self.pooling_5(features)

        features = features.squeeze(-1).squeeze(-1)

        features = self.drop_out(features)
        features = self.fc(features)
        logits = self.fc2(features)

        return logits