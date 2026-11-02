import torch
import torch.nn as nn
from .residual import ConvResidualBlock, DownsampleResidualBlock
class ResNet18(nn.Module):
    def __init__(self, channels=3,num_classes=10):
        super().__init__()
        self.conv1=nn.Conv2d(channels,64,kernel_size=7,stride=2,padding=3)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv1_block=ConvResidualBlock(64)
        self.downsample1=DownsampleResidualBlock(64,128)
        self.conv2_block=ConvResidualBlock(128)
        self.downsample2=DownsampleResidualBlock(128,256)
        self.conv3_block=ConvResidualBlock(256)
        self.downsample3=DownsampleResidualBlock(256,512)
        self.conv4_block=ConvResidualBlock(512)
        self.avgpool=nn.AdaptiveAvgPool2d((1, 1))
        self.fc=nn.Linear(512,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv1_block(x)
        x = self.downsample1(x)
        x = self.conv2_block(x)
        x = self.downsample2(x)
        x = self.conv3_block(x)
        x = self.downsample3(x)
        x = self.conv4_block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
