"""
BiHSFPN 消融变体
- BiHSFPN_noCHFP: CHFP 替换为 nn.Identity, 保留 BiHSFPN 所有其他结构 (含 AERF? 否, AERF 内嵌在 CHFP 里, 一起跳过)
- BiHSFPN_noP2:    3 路输入版本 (P3/P4/P5), 关闭 P2 浅层

设计原则: 继承 BiHSFPN, 最小改动, 不重写 forward 逻辑.
"""
import torch.nn as nn
from typing import List

from .bi_hsfpn import BiHSFPN, ConvBnAct


class BiHSFPN_noCHFP(BiHSFPN):
    """
    消融 D: 去掉 CHFP (含内嵌的 AERF).
    把 self.chfp 4 个模块都换成 nn.Identity, 这样 forward 时通道注意力+高通+AERF 全部跳过,
    但 BiHSFPN 的 LightFusion + DynamicFusion + 4 路双向融合保留.
    """
    def __init__(self,
                 in_channels: List[int] = [64, 64, 128, 256],
                 out_channels: List[int] = [64, 64, 128, 256],
                 alpha: float = 0.3):
        super().__init__(in_channels=in_channels, out_channels=out_channels, alpha=alpha)
        # 用 nn.Identity 覆盖 self.chfp (4 路全部跳过)
        self.chfp = nn.ModuleList([nn.Identity() for _ in range(4)])
