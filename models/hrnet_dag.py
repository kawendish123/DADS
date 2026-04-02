import torch
import torch.nn as nn
from collections import abc

BN_MOMENTUM = 0.1
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Operation_Fuse(nn.Module):
    """
    用于显式替换 HRNet 中的融合操作，作为 DAG 的合并节点。
    """

    def __init__(self, fuse_layers_list):
        super().__init__()
        # fuse_layers_list 是一个 nn.ModuleList，包含了对多个分支的变换
        self.fuse_layers = fuse_layers_list
        self.relu = nn.ReLU(inplace=True)

    def forward(self, outputs):
        # outputs 是前面多个分支输出的列表 [out_branch1, out_branch2, ...]
        res = sum([self.fuse_layers[j](outputs[j]) for j in range(len(outputs))])
        return self.relu(res)


class HRNet_DAG(nn.Module):
    def __init__(self, base_channel: int = 32, num_joints: int = 17):
        super().__init__()
        self.branch_list = nn.ModuleList()
        self.merge_ops = nn.ModuleDict()  # 存放融合操作

        # -------------- 1. 拆解网络结构构建分支 --------------
        # 分支 0: Stem + Layer1 (主干)
        stem_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            Bottleneck(64, 64, downsample=nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
            )),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )
        self.branch_list.append(stem_layer1)    #branch 0 (主干)

        # 分支 1: Transition 1 -> branch 1 (高分辨率)
        trans1_b1 = nn.Sequential(nn.Conv2d(256, base_channel, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                                  nn.ReLU(inplace=True)
                                  )
        self.branch_list.append(trans1_b1)    #branch 1 (高分辨率)

        # 分支 2: Transition 1 -> branch 2 (低分辨率)
        trans1_b2 = nn.Sequential(
                        nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                )
        self.branch_list.append(trans1_b2)    #branch 2 (低分辨率)

        # stage2
        # 分支 1 (高分辨率)
        self.branch_list.append(nn.Sequential(
            BasicBlock(base_channel, base_channel),
            BasicBlock(base_channel, base_channel),
            BasicBlock(base_channel, base_channel),
            BasicBlock(base_channel, base_channel)
        ))  # 假设这是 branch_list[3]

        # 分支 2 (低分辨率)
        self.branch_list.append(nn.Sequential(
            BasicBlock(base_channel * 2, base_channel * 2),
            BasicBlock(base_channel * 2, base_channel * 2),
            BasicBlock(base_channel * 2, base_channel * 2),
            BasicBlock(base_channel * 2, base_channel * 2)
        ))  # 假设这是 branch_list[4]

        # --- Stage 2 融合点定义 ---
        # 提取 StageModule 中的融合逻辑
        self.merge_ops["s2_f1"] = Operation_Fuse(nn.ModuleList([
            nn.Identity(),
            nn.Sequential(nn.Conv2d(base_channel * 2, base_channel, 1), nn.BatchNorm2d(base_channel),
                          nn.Upsample(scale_factor=2))
        ])) #merge 0

        self.merge_ops["s2_f2"] = Operation_Fuse(nn.ModuleList([
            nn.Sequential(nn.Conv2d(base_channel, base_channel * 2, 3, 2, 1), nn.BatchNorm2d(base_channel * 2)),
            nn.Identity()
        ])) #merge 1

        # 分支 1 和 2 保持不变
        self.trans2_b1 = nn.Identity()
        self.trans2_b2 = nn.Identity()
        self.branch_list.append(self.trans2_b1)    #branch 5 (高分辨率)
        self.branch_list.append(self.trans2_b2)    #branch 6 (低分辨率)

        # 分支 3：基于分支 2 (1/8 尺度) 生成 1/16 尺度
        self.trans2_b3 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )
        )
        self.branch_list.append(self.trans2_b3)    #branch 7 (低分辨率)

        # stage3
        for m in range(4):
            # 1. 添加该模块的 3 个并行计算分支 (每个分支 4 个 BasicBlock)
            self.branch_list.append(nn.Sequential(*[BasicBlock(base_channel, base_channel) for _ in range(4)]))
            self.branch_list.append(nn.Sequential(*[BasicBlock(base_channel * 2, base_channel * 2) for _ in range(4)]))
            self.branch_list.append(nn.Sequential(*[BasicBlock(base_channel * 4, base_channel * 4) for _ in range(4)]))
            #19
        # --- Stage 3 融合点定义 ---
            # 提取 StageModule 中的融合逻辑
            self.merge_ops[f"s3_m{m}_f1"] = Operation_Fuse(nn.ModuleList([
                nn.Identity(),
                #两倍上采样
                nn.Sequential(nn.Conv2d(base_channel * 2, base_channel, 1),
                              nn.BatchNorm2d(base_channel),
                              nn.Upsample(scale_factor=2,mode='nearest')),
                #4倍上采样
                nn.Sequential(nn.Conv2d(base_channel * 4, base_channel, 1),
                              nn.BatchNorm2d(base_channel),
                              nn.Upsample(scale_factor=4,mode='nearest')),
            ]))
            self.merge_ops[f"s3_m{m}_f2"] = Operation_Fuse(nn.ModuleList([
                #2倍下采样
                nn.Sequential(nn.Conv2d(base_channel, base_channel * 2, 3, 2, 1),
                              nn.BatchNorm2d(base_channel * 2)),
                nn.Identity(),
                #2倍上采样
                nn.Sequential(nn.Conv2d(base_channel * 4, base_channel * 2, 1),
                              nn.BatchNorm2d(base_channel*2),
                              nn.Upsample(scale_factor=2,mode='nearest')),
            ]))
            self.merge_ops[f"s3_m{m}_f3"] = Operation_Fuse(nn.ModuleList([
                # 4倍下采样
                nn.Sequential(
                    # 第一次下采样：尺寸减半，通道通常保持不变（遵循原版逻辑）
                    nn.Sequential(
                        nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)  # 中间这层激活必不可少
                    ),
                    # 第二次下采样：尺寸再减半，并调整到目标通道数
                    nn.Sequential(
                        nn.Conv2d(base_channel, base_channel*4, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(base_channel*4, momentum=BN_MOMENTUM)
                    )
                ),
                # 2倍下采样
                nn.Sequential(nn.Conv2d(base_channel*2, base_channel * 4, 3, 2, 1),
                              nn.BatchNorm2d(base_channel * 4)),
                nn.Identity()
            ]))


        # 前三个分支透传
        self.trans3_b1 = nn.Identity()
        self.branch_list.append(self.trans3_b1)    #branch 20 (高分辨率)
        self.trans3_b2 = nn.Identity()
        self.branch_list.append(self.trans3_b2)    #branch 21 (中分辨率)
        self.trans3_b3 = nn.Identity()
        self.branch_list.append(self.trans3_b3)    #branch 22 (低分辨率)
        # 分支 4：基于分支 3 (1/16 尺度) 生成 1/32 尺度
        self.trans3_b4 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )
        )
        self.branch_list.append(self.trans3_b4)    #branch 23 (拉分辨率)


        # stage4
        # --- Stage 4 重构：循环 3 次 ---
        for m in range(3):
            # 1. 添加该模块的 4 个并行计算分支
            self.branch_list.append(nn.Sequential(*[BasicBlock(base_channel, base_channel) for _ in range(4)]))
            self.branch_list.append(nn.Sequential(*[BasicBlock(base_channel * 2, base_channel * 2) for _ in range(4)]))
            self.branch_list.append(nn.Sequential(*[BasicBlock(base_channel * 4, base_channel * 4) for _ in range(4)]))
            self.branch_list.append(nn.Sequential(*[BasicBlock(base_channel * 8, base_channel * 8) for _ in range(4)]))
            # 35
            # 2. 定义该模块的融合操作
            if m < 2:  # 前两个模块：正常的 4 进 4 出融合
                self.merge_ops[f"s4_m{m}_f1"] = Operation_Fuse(nn.ModuleList([
            # 直接复制
            nn.Identity(),
            # 2倍上采样
            nn.Sequential(nn.Conv2d(base_channel * 2, base_channel, 1),
                          nn.BatchNorm2d(base_channel),
                          nn.Upsample(scale_factor=2, mode='nearest')),
            # 4倍上采样
            nn.Sequential(nn.Conv2d(base_channel * 4, base_channel, 1),
                          nn.BatchNorm2d(base_channel),
                          nn.Upsample(scale_factor=4, mode='nearest')),
            # 8倍上采样
            nn.Sequential(nn.Conv2d(base_channel * 8, base_channel, 1),
                          nn.BatchNorm2d(base_channel),
                          nn.Upsample(scale_factor=8, mode='nearest')),
        ]))
                self.merge_ops[f"s4_m{m}_f2"] = Operation_Fuse(nn.ModuleList([
            # 2倍下采样
            nn.Sequential(nn.Conv2d(base_channel, base_channel * 2, 3, 2, 1),
                          nn.BatchNorm2d(base_channel * 2)),
            # 直接复制
            nn.Identity(),
            # 2倍上采样
            nn.Sequential(nn.Conv2d(base_channel * 4, base_channel * 2, 1),
                          nn.BatchNorm2d(base_channel * 2),
                          nn.Upsample(scale_factor=2, mode='nearest')),
            # 4倍上采样
            nn.Sequential(nn.Conv2d(base_channel * 8, base_channel * 2, 1),
                          nn.BatchNorm2d(base_channel * 2),
                          nn.Upsample(scale_factor=4, mode='nearest')),
        ]))

                self.merge_ops[f"s4_m{m}_f3"] = Operation_Fuse(nn.ModuleList([
            # 4倍下采样
            nn.Sequential(
                # 第一次下采样：尺寸减半，通道通常保持不变（遵循原版逻辑）
                nn.Sequential(
                    nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)  # 中间这层激活必不可少
                ),
                # 第二次下采样：尺寸再减半，并调整到目标通道数
                nn.Sequential(
                    nn.Conv2d(base_channel, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM)
                )
            ),
            # 2倍下采样
            nn.Sequential(nn.Conv2d(base_channel * 2, base_channel * 4, 3, 2, 1),
                          nn.BatchNorm2d(base_channel * 4)),
            # 直接复制
            nn.Identity(),
        #     2倍上采样
            nn.Sequential(nn.Conv2d(base_channel * 8, base_channel * 4, 1),
                          nn.BatchNorm2d(base_channel * 4),
                          nn.Upsample(scale_factor=2, mode='nearest')),
        ]))
                self.merge_ops[f"s4_m{m}_f4"] = Operation_Fuse(nn.ModuleList([
            # 1. 8倍下采样 (来自分支 1: 1/4 -> 1/32)
            # 依据：原版逻辑需堆叠 3 个 stride=2 的卷积，前两个加 ReLU
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel, base_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(base_channel, base_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(base_channel, base_channel * 8, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM)
                )
            ),

            # 2. 4倍下采样 (来分分支 2: 1/8 -> 1/32)
            # 依据：堆叠 2 个 stride=2 的卷积，中间加 ReLU
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 2, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 8, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM)
                )
            ),

            # 3. 2倍下采样 (来自分支 3: 1/16 -> 1/32)
            nn.Sequential(
                nn.Conv2d(base_channel * 4, base_channel * 8, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM)
            ),

            # 4. 直接复制 (来自分支 4)
            nn.Identity()
        ]))
            else:  # 最后一个模块：4 进 1 出融合 (只输出最高分辨率)
                self.merge_ops[f"s4_m{m}_f1"] = Operation_Fuse(nn.ModuleList([
            # 直接复制
            nn.Identity(),
            # 2倍上采样
            nn.Sequential(nn.Conv2d(base_channel * 2, base_channel, 1),
                          nn.BatchNorm2d(base_channel),
                          nn.Upsample(scale_factor=2, mode='nearest')),
            # 4倍上采样
            nn.Sequential(nn.Conv2d(base_channel * 4, base_channel, 1),
                          nn.BatchNorm2d(base_channel),
                          nn.Upsample(scale_factor=4, mode='nearest')),
            # 8倍上采样
            nn.Sequential(nn.Conv2d(base_channel * 8, base_channel, 1),
                          nn.BatchNorm2d(base_channel),
                          nn.Upsample(scale_factor=8, mode='nearest')),
        ]))



        self.final_layer = nn.Sequential(
            nn.Conv2d(base_channel, num_joints, kernel_size=1, stride=1)
        )
        self.branch_list.append(self.final_layer)  # 假设这是 branch_list[36]

        # -------------- 2. 计算 Accumulate Length --------------
        self.accumulate_len = []
        for i, branch in enumerate(self.branch_list):
            # 核心修正：判断分支是否为 nn.Sequential
            # 如果是 Sequential，取其内部层数；如果是 Identity 或单层卷积，长度计为 1
            if isinstance(branch, nn.Sequential):
                current_branch_len = len(branch)
            else:
                current_branch_len = 1

            if i == 0:
                self.accumulate_len.append(current_branch_len)
            else:
                self.accumulate_len.append(self.accumulate_len[i - 1] + current_branch_len)

        # -------------- 3. 构建 DAG 拓扑词典 (完整修正版) --------------
        self.has_dag_topology = True

        # 辅助函数：获取第 n 个分支“最后一层”的逻辑索引 (用于作为前置依赖)
        def end_of(n):
            return self.accumulate_len[n]

        # 辅助函数：获取第 n 个分支“第一层”的逻辑索引 (用于作为字典的 Key)
        def start_of(n):
            return self.accumulate_len[n - 1] + 1 if n > 0 else 1

        # 1. 记录所有需要缓存输出的层索引 (所有作为输入被引用的分支末尾)
        self.record_output_list = [end_of(i) for i in range(len(self.branch_list))]

        # 2. 核心拓扑依赖字典
        self.dag_dict = {
            # ================= Stage 1 & Transition 1 =================
            start_of(1): end_of(0),  # trans1_b1 (Branch 1) 依赖 Stem (Branch 0) 的末尾
            start_of(2): end_of(0),  # trans1_b2 (Branch 2) 依赖 Stem (Branch 0) 的末尾

            # ================= Stage 2 =================
            start_of(3): end_of(1),  # s2_b1 (Branch 3) 依赖 trans1_b1
            start_of(4): end_of(2),  # s2_b2 (Branch 4) 依赖 trans1_b2

            # Stage 2 Fusion 节点依赖
            "s2_f1": [end_of(3), end_of(4)],
            "s2_f2": [end_of(3), end_of(4)],

            # ================= Transition 2 =================
            start_of(5): "s2_f1",  # trans2_b1 (Branch 5) 依赖 s2_f1
            start_of(6): "s2_f2",  # trans2_b2 (Branch 6) 依赖 s2_f2
            start_of(7): "s2_f2",  # trans2_b3 (Branch 7) 依赖 s2_f2 (下采样起点)
        }

        # ================= Stage 3 (4 组 Module，索引 8 ~ 19) =================
        for m in range(4):
            b1_idx = 8 + m * 3
            b2_idx = b1_idx + 1
            b3_idx = b1_idx + 2

            # A. 确定每个 Module 计算分支的输入依赖
            if m == 0:
                self.dag_dict[start_of(b1_idx)] = end_of(5)  # 依赖 trans2_b1
                self.dag_dict[start_of(b2_idx)] = end_of(6)  # 依赖 trans2_b2
                self.dag_dict[start_of(b3_idx)] = end_of(7)  # 依赖 trans2_b3
            else:
                self.dag_dict[start_of(b1_idx)] = f"s3_m{m - 1}_f1"
                self.dag_dict[start_of(b2_idx)] = f"s3_m{m - 1}_f2"
                self.dag_dict[start_of(b3_idx)] = f"s3_m{m - 1}_f3"

            # B. 记录当前 Module 融合节点的输入依赖
            fusion_inputs = [end_of(b1_idx), end_of(b2_idx), end_of(b3_idx)]
            self.dag_dict[f"s3_m{m}_f1"] = fusion_inputs
            self.dag_dict[f"s3_m{m}_f2"] = fusion_inputs
            self.dag_dict[f"s3_m{m}_f3"] = fusion_inputs

        # ================= Transition 3 (索引 20 ~ 23) =================
        self.dag_dict[start_of(20)] = "s3_m3_f1"  # trans3_b1 依赖 Stage3 最后融合输出 1
        self.dag_dict[start_of(21)] = "s3_m3_f2"  # trans3_b2 依赖 Stage3 最后融合输出 2
        self.dag_dict[start_of(22)] = "s3_m3_f3"  # trans3_b3 依赖 Stage3 最后融合输出 3
        self.dag_dict[start_of(23)] = "s3_m3_f3"  # trans3_b4 依赖 Stage3 最后融合输出 3 (生成1/32)

        # ================= Stage 4 (3 组 Module，索引 24 ~ 35) =================
        for m in range(3):
            b1_idx = 24 + m * 4
            b2_idx = b1_idx + 1
            b3_idx = b1_idx + 2
            b4_idx = b1_idx + 3

            # A. 确定每个 Module 计算分支的输入依赖
            if m == 0:
                self.dag_dict[start_of(b1_idx)] = end_of(20)  # 依赖 trans3_b1
                self.dag_dict[start_of(b2_idx)] = end_of(21)  # 依赖 trans3_b2
                self.dag_dict[start_of(b3_idx)] = end_of(22)  # 依赖 trans3_b3
                self.dag_dict[start_of(b4_idx)] = end_of(23)  # 依赖 trans3_b4
            else:
                self.dag_dict[start_of(b1_idx)] = f"s4_m{m - 1}_f1"
                self.dag_dict[start_of(b2_idx)] = f"s4_m{m - 1}_f2"
                self.dag_dict[start_of(b3_idx)] = f"s4_m{m - 1}_f3"
                self.dag_dict[start_of(b4_idx)] = f"s4_m{m - 1}_f4"

            # B. 记录当前 Module 融合节点的输入依赖
            fusion_inputs = [end_of(b1_idx), end_of(b2_idx), end_of(b3_idx), end_of(b4_idx)]
            if m < 2:
                self.dag_dict[f"s4_m{m}_f1"] = fusion_inputs
                self.dag_dict[f"s4_m{m}_f2"] = fusion_inputs
                self.dag_dict[f"s4_m{m}_f3"] = fusion_inputs
                self.dag_dict[f"s4_m{m}_f4"] = fusion_inputs
            else:
                # 最后一个模块仅输出 1 路最高分辨率融合
                self.dag_dict[f"s4_m{m}_f1"] = fusion_inputs

        # ================= Final Layer (索引 36) =================
        self.dag_dict[start_of(36)] = "s4_m2_f1"  # 依赖 Stage 4 最后一次单路融合输出

    def forward(self, x):
        # 1. Stem + Stage 1 (索引 0)
        # 输入: [B, 3, H, W] -> 输出: [B, 256, H/4, W/4]
        x = self.branch_list[0](x)

        # 2. Transition 1 (索引 1, 2)
        # 生成两个分辨率的分支
        x_s2_b1 = self.branch_list[1](x)  # 1/4 尺度, c 通道
        x_s2_b2 = self.branch_list[2](x)  # 1/8 尺度, 2c 通道

        # 3. Stage 2 (索引 3, 4)
        # Blocks 计算
        x_s2_b1 = self.branch_list[3](x_s2_b1)
        x_s2_b2 = self.branch_list[4](x_s2_b2)
        # Fusion 融合 (s2_f1, s2_f2)
        f2_1 = self.merge_ops["s2_f1"]([x_s2_b1, x_s2_b2])
        f2_2 = self.merge_ops["s2_f2"]([x_s2_b1, x_s2_b2])

        # 4. Transition 2 (索引 5, 6, 7)
        # 从 2 分支扩展到 3 分支
        x_s3_b1 = self.branch_list[5](f2_1)  # Identity
        x_s3_b2 = self.branch_list[6](f2_2)  # Identity
        x_s3_b3 = self.branch_list[7](f2_2)  # 下采样生成 1/16 尺度

        current_s3_inputs = [x_s3_b1, x_s3_b2, x_s3_b3]

        # 5. Stage 3 (4 个 Module 重复: 索引 8-19)
        for m in range(4):
            offset = 8 + m * 3
            # Parallel Blocks
            b1 = self.branch_list[offset](current_s3_inputs[0])
            b2 = self.branch_list[offset + 1](current_s3_inputs[1])
            b3 = self.branch_list[offset + 2](current_s3_inputs[2])
            # Multi-scale Fusion
            current_s3_inputs = [
                self.merge_ops[f"s3_m{m}_f1"]([b1, b2, b3]),
                self.merge_ops[f"s3_m{m}_f2"]([b1, b2, b3]),
                self.merge_ops[f"s3_m{m}_f3"]([b1, b2, b3])
            ]

        # 6. Transition 3 (索引 20, 21, 22, 23)
        # 从 3 分支扩展到 4 分支
        x_s4_b1 = self.branch_list[20](current_s3_inputs[0])
        x_s4_b2 = self.branch_list[21](current_s3_inputs[1])
        x_s4_b3 = self.branch_list[22](current_s3_inputs[2])
        x_s4_b4 = self.branch_list[23](current_s3_inputs[2])  # 基于分支 3 生成 1/32

        current_s4_inputs = [x_s4_b1, x_s4_b2, x_s4_b3, x_s4_b4]

        # 7. Stage 4 (3 个 Module 重复: 索引 24-35)
        for m in range(3):
            offset = 24 + m * 4
            # Parallel Blocks
            b1 = self.branch_list[offset](current_s4_inputs[0])
            b2 = self.branch_list[offset + 1](current_s4_inputs[1])
            b3 = self.branch_list[offset + 2](current_s4_inputs[2])
            b4 = self.branch_list[offset + 3](current_s4_inputs[3])

            if m < 2:  # 前两个模块输出 4 个分支
                current_s4_inputs = [
                    self.merge_ops[f"s4_m{m}_f1"]([b1, b2, b3, b4]),
                    self.merge_ops[f"s4_m{m}_f2"]([b1, b2, b3, b4]),
                    self.merge_ops[f"s4_m{m}_f3"]([b1, b2, b3, b4]),
                    self.merge_ops[f"s4_m{m}_f4"]([b1, b2, b3, b4])
                ]
            else:  # 最后一个模块只融合输出最高分辨率分支
                x_final_fused = self.merge_ops[f"s4_m{m}_f1"]([b1, b2, b3, b4])

        # 8. Final Layer (索引 36)
        return self.branch_list[36](x_final_fused)

    def __len__(self):
        # 返回整个 branch_list 中所有层的总和
        return self.accumulate_len[-1]

    def __getitem__(self, item):
        # 兼容逻辑：将全局索引映射回具体的 branch 和内部 layer
        if isinstance(item, int):
            # 获取层所在的分支下标
            part_index = self.getBlockIndex(item, self.accumulate_len)
            # 计算在该分支内部的相对偏移
            start_index = self.accumulate_len[part_index - 1] if part_index > 0 else 0
            return self.branch_list[part_index][item - start_index]
        else:
            # 这里的切片逻辑通常由 EasyModel 框架的 EdgeModel 类来处理
            return super().__getitem__(item)

    def getBlockIndex(self, item, accumulate_len):
        # 你的导航仪函数
        for part_index in range(len(accumulate_len)):
            if item < accumulate_len[part_index]:
                return part_index
        return len(accumulate_len)

    def __iter__(self):
        # 迭代器：依次返回 branch_list 中的每一层
        for branch in self.branch_list:
            # 核心修正：判断分支是否为可迭代对象（如 nn.Sequential）
            if isinstance(branch, nn.Sequential):
                for layer in branch:
                    yield layer
            else:
                # 如果是 Identity 或单层卷积，直接将其作为“层”返回
                yield branch