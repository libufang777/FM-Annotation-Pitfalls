"""
BiHSFPN_WCHFP wavelet 基子类
都继承 BiHSFPN_WCHFP, 只改 wavelet 参数, 用于消融对比.
"""
from typing import List
from .bi_hsfpn_wchfp import BiHSFPN_WCHFP


class BiHSFPN_WCHFP_HAAR(BiHSFPN_WCHFP):
    """haar 小波 (支撑长度 2, 最锐边缘但有块状伪影). 用于参数量最小对照."""
    def __init__(self,
                 in_channels: List[int] = [64, 64, 128, 256],
                 out_channels: List[int] = [64, 64, 128, 256],
                 alpha: float = 0.3):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         alpha=alpha, wavelet='haar')


class BiHSFPN_WCHFP_SYM4(BiHSFPN_WCHFP):
    """sym4 小波 (支撑长度 8, db4 的近对称版, 减少相位失真)."""
    def __init__(self,
                 in_channels: List[int] = [64, 64, 128, 256],
                 out_channels: List[int] = [64, 64, 128, 256],
                 alpha: float = 0.3):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         alpha=alpha, wavelet='sym4')
