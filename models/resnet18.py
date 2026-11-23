import torch
import torch.nn as nn
from .residual import ConvResidualBlock, DownsampleResidualBlock
from .importance_pool import (
    ImportanceGuidedAdaptivePooling,
    LearnedWeightedMeanPooling,
)


class ResNCIF18(nn.Module):
    def __init__(self, channels=3, num_classes=10, pooling="importance"):
        super().__init__()
        self.conv1=nn.Conv2d(channels,16,kernel_size=3,stride=1,padding=1)
        self.conv1_block=ConvResidualBlock(16)
        self.conv2_block=ConvResidualBlock(16)
        self.conv3_block=ConvResidualBlock(16)
        self.downsample1=DownsampleResidualBlock(16,32)
        self.conv4_block=ConvResidualBlock(32)
        self.conv5_block=ConvResidualBlock(32)
        self.downsample2=DownsampleResidualBlock(32,64)
        self.conv6_block=ConvResidualBlock(64)
        self.conv7_block=ConvResidualBlock(64)
        self.avgpool=nn.AdaptiveAvgPool2d((1, 1))
        self.fc=nn.Linear(64,num_classes)
        if pooling == "importance":
            self.maxpool = ImportanceGuidedAdaptivePooling(output_size=None)
        elif pooling == "max":
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif pooling == "learned":
            self.maxpool = LearnedWeightedMeanPooling(16)
        else:
            raise ValueError("pooling must be 'max', 'importance', or 'learned'")
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        # Stage 1: 32x32, 16 channels
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.conv3_block(x)
        # Stage 2: 16x16, 32 channels
        x = self.downsample1(x)
        x = self.conv4_block(x)
        x = self.conv5_block(x)
        # Stage 3: 8x8, 64 channels
        x = self.downsample2(x)
        x = self.conv6_block(x)
        x = self.conv7_block(x)
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18(ResNCIF18):
    """Classical max-pooling version of the project's CIFAR-style ResNet."""

    def __init__(self, channels=3, num_classes=10):
        super().__init__(channels=channels, num_classes=num_classes, pooling="max")


class ImportanceGuidedResNet18(ResNCIF18):
    """Importance-partitioned max-pooling version of the shared backbone."""

    def __init__(self, channels=3, num_classes=10):
        super().__init__(channels=channels, num_classes=num_classes, pooling="importance")


class LearnedWeightedResNet18(ResNCIF18):
    """Feature-dependent learned weighted-mean pooling version."""

    def __init__(self, channels=3, num_classes=10):
        super().__init__(channels=channels, num_classes=num_classes, pooling="learned")
