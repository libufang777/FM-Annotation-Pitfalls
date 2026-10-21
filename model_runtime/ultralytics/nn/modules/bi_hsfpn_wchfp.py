"""
BiHSFPN_WCHFP -- BiHSFPN with WCHFP replacing CHFP
仅替换 CHFP -> WCHFP, 其他逻辑 100% 继承原 BiHSFPN.
"""
import torch.nn as nn
from typing import List

from .bi_hsfpn import BiHSFPN, ConvBnAct
from .wchfp import WCHFP


class BiHSFPN_WCHFP(BiHSFPN):
    """继承 BiHSFPN, 只把 self.chfp 列表里 4 个 CHFP 实例替换成 WCHFP."""
    def __init__(self,
                 in_channels: List[int] = [64, 64, 128, 256],
                 out_channels: List[int] = [64, 64, 128, 256],
                 alpha: float = 0.3,
                 wavelet: str = 'db4'):
        super().__init__(in_channels=in_channels, out_channels=out_channels, alpha=alpha)
        # 用 WCHFP 覆盖父类的 self.chfp (4 路)
        c1o, c2o, c3o, c4o = out_channels
        self.chfp = nn.ModuleList([
            WCHFP(c1o, alpha=alpha, wavelet=wavelet),
            WCHFP(c2o, alpha=alpha, wavelet=wavelet),
            WCHFP(c3o, alpha=alpha, wavelet=wavelet),
            WCHFP(c4o, alpha=alpha, wavelet=wavelet),
        ])
