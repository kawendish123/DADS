import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import abc


# 1. 实现核心的条件通道加权模块 (CCW)
class ConditionalChannelWeighting(nn.Module):
    def __init__(self, in_channels, stride=1):
        super(ConditionalChannelWeighting, self).__init__()
        self.stride = stride
        # 深度卷积提取空间特征
        self.branch_dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
        )
        # 权重生成分支 (简化版)
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 类似 ShuffleNet 的方式，这里简化为加权逻辑
        weight = self.weight_gen(x)
        out = self.branch_dw(x)
        return out * weight


# 2. 定义 LiteHRNet 主体结构
class LiteHRNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1000):
        super(LiteHRNet, self).__init__()
        # DADS 框架必需的属性标识
        self.has_dag_topology = False  # 如果是简单的序列结构设为 False

        # 参照 AlexNet 的 layers 结构定义
        self.layers = nn.Sequential(
            # Stem 阶段
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 阶段 1: CCW Blocks
            ConditionalChannelWeighting(32),
            nn.ReLU(inplace=True),

            # 阶段 2: 降采样与 CCW
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 下采样
            ConditionalChannelWeighting(64),
            nn.ReLU(inplace=True),

            # 分类头
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        self.len = len(self.layers)

    def forward(self, x):
        return self.layers(x)

    # 适配 DADS 的迭代器接口
    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.layers[index]