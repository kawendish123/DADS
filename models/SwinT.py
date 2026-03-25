import torch
import torch.nn as nn
from torchvision.models import swin_t


# ==========================================
# 1. 保持原样的原子层 Wrapper
# ==========================================
class PatchEmbedWrapper(nn.Module):
    def __init__(self, patch_embed_module):
        super().__init__()
        self.module = patch_embed_module

    def forward(self, x):
        return self.module(x)


class SwinBlockWrapper(nn.Module):
    def __init__(self, block_module):
        super().__init__()
        self.module = block_module

    def forward(self, x):
        return self.module(x)


class PatchMergingWrapper(nn.Module):
    def __init__(self, merge_module):
        super().__init__()
        self.module = merge_module

    def forward(self, x):
        return self.module(x)


class HeadWrapper(nn.Module):
    def __init__(self, norm, avgpool, flatten, head):
        super().__init__()
        self.norm = norm
        self.avgpool = avgpool
        self.flatten = flatten
        self.head = head

    def forward(self, x):
        x = self.norm(x)
        # 【关键修复】：手动进行维度置换 (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.head(x)


class SwinDADSModel(nn.Module):


    def __init__(self, in_channels: int = 3):
        super().__init__()

        # 1. 在内部自动实例化原生的 torchvision swin 模型
        raw_model = swin_t(weights=None)

        # 我们需要动态修改 Swin Transformer 第一层 Patch Embedding 的输入通道数
        if in_channels != 3:
            # swin_t 的 PatchEmbed 默认第一层是一个 kernel=4, stride=4 的 Conv2d
            original_conv = raw_model.features[0][0]
            raw_model.features[0][0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride
            )

        self.layers = nn.ModuleList()

        # 2. 提取 Patch Embedding
        self.layers.append(PatchEmbedWrapper(raw_model.features[0]))

        # 3. 提取所有的 SwinBlock 和 PatchMerging
        for i in range(1, len(raw_model.features)):
            stage_module = raw_model.features[i]
            if stage_module.__class__.__name__ == 'PatchMerging':
                self.layers.append(PatchMergingWrapper(stage_module))
            else:
                for block in stage_module:
                    self.layers.append(SwinBlockWrapper(block))

        # 4. 提取最终的分类头
        self.layers.append(HeadWrapper(
            norm=raw_model.norm,
            avgpool=raw_model.avgpool,
            flatten=raw_model.flatten,
            head=raw_model.head
        ))


        self.has_dag_topology = False
        self.dag_dict = {}
        self.record_output_list = []

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, item):
        if item >= len(self.layers):
            raise StopIteration()
        return self.layers[item]

    def __iter__(self):
        return iter(self.layers)