import torch 
import torch.nn as nn
import torch.nn.functional as F

class Convolution_Block(nn.Module):
    def __init__(self, in_channels, stride=1):
        super().__init__()
        padding = 'same' if stride == 1 else 'valid'
        self.conv = nn.Conv2d(in_channels, 2*in_channels, 3, padding=padding, stride=stride)
        self.batch_norm  = nn.BatchNorm2d(2*in_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x

class Depth_Convolution_Block(nn.Module):
    def __init__(self, in_channels, stride=2):
        super().__init__()
        padding = 'same' if stride == 1 else 'valid'
        self.depth_conv = nn.Conv2d(in_channels, in_channels, 3, groups=in_channels, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x

class Convolution_Block_Same(nn.Module):
    def __init__(self, in_channels, stride=1):
        super().__init__()
        padding = 'same' if stride==1 else 'valid'
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=padding, stride=stride)
        self.batch_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)

        return x

class MobileNetV2(nn.Module):
    def __init__(self, in_channels, classes=10):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Depth_Convolution_Block(32, stride=1),
            Convolution_Block(32),
            Depth_Convolution_Block(64),
            Convolution_Block(64),
            Depth_Convolution_Block(128, stride=1),
            Convolution_Block_Same(128),
            Depth_Convolution_Block(128),
            Convolution_Block(128),
            Depth_Convolution_Block(256, stride=1),
            Convolution_Block_Same(256),
            Depth_Convolution_Block(256, stride=2),
            Convolution_Block(256)
        )

        self.list = []
        self.module_list_depth = nn.ModuleList([Depth_Convolution_Block(512, stride=1) for i in range(5)])
        self.module_list_conv = nn.ModuleList([Convolution_Block_Same(512) for i in range(5)])

        self.layers2 = nn.Sequential(
            Depth_Convolution_Block(512, stride=2),
            Convolution_Block(512),
            Depth_Convolution_Block(1024, stride=2),
            Convolution_Block_Same(1024),
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(7*7*1024, classes),
        )

    def forward(self, x):
        x = self.layers1(x)
        for layer1, layer2 in zip(self.module_list_depth, self.module_list_conv):
            x = layer1(x)
            x = layer2(x)
        x = self.layers2(x)
        return x



            

