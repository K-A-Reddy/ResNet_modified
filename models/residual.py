import torch
import torch.nn as nn
class ConvResidualBlock(nn.Module):
  def __init__(self, channels:int):
    super().__init__()
    self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    self.bn1= nn.BatchNorm2d(channels)
    self.relu = nn.ReLU()
    self.conv2= nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn2= nn.BatchNorm2d(channels)
  def forward(self,x):
    identity=x
    out = self.conv1(x)
    out= self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out= self.bn2(out)
    out = out+identity
    y=self.relu(out)
    return y
class DownsampleResidualBlock(nn.Module):
  def __init__(self,in_channels,out_channels:int):
    super().__init__()
    self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1)
    self.bn1=nn.BatchNorm2d(out_channels)
    self.relu=nn.ReLU()
    self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
    self.bn2=nn.BatchNorm2d(out_channels)
    self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(out_channels))
  def forward(self,x):
    identity = self.shortcut(x)
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + identity
    out = self.relu(out)
    return out