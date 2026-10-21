"""
WCHFP -- Wavelet-based Channel High-pass Filtering Path (with embedded AERF)
对齐原 CHFP 接口和 AERF 嵌入位置, 是 BiHSFPN_WCHFP 的子模块.
LL 保留不动 (对照 Retinex 失败教训), 三个高频子带可学习增益 + IDWT + AERF.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward, DWTInverse
from .bi_hsfpn import AERF, ConvBnAct


class SubBandGainPredictor(nn.Module):
    """输入 (B,C,H,W) 输出 g_LH, g_HL, g_HH each (B,C) ∈ [0,+inf)."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels * 3, bias=True)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc2.bias, math.log(math.e - 1))  # softplus(.) ≈ 1

    def forward(self, x):
        s = self.gap(x).flatten(1)
        s = F.softplus(self.fc1(s))
        g = F.softplus(self.fc2(s))
        return g.chunk(3, dim=1)


class WCHFP(nn.Module):
    """接口与 CHFP 完全对齐: WCHFP(channels, alpha=0.3) -> (B,C,H,W)."""
    def __init__(self, channels: int, alpha: float = 0.3, wavelet: str = 'db4'):
        super().__init__()
        self.channels = channels
        self.alpha = alpha
        self.dwt  = DWTForward(J=1, wave=wavelet, mode='zero')
        self.idwt = DWTInverse(wave=wavelet, mode='zero')
        self.gain = SubBandGainPredictor(channels)

        mid = max(channels // 4, 8)
        self.cp_gap = nn.AdaptiveAvgPool2d(1)
        self.cp_gmp = nn.AdaptiveMaxPool2d(1)
        self.cp_fc = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sp_fuse = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False), nn.Sigmoid())
        self.aerf = AERF(channels)
        self.out_conv = ConvBnAct(channels, channels, k=3)

    def _wavelet_highpass(self, x):
        ll, yh = self.dwt(x)
        coeffs = yh[0]
        lh, hl, hh = coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2]
        g_lh, g_hl, g_hh = self.gain(x)
        lh = lh * g_lh.view(-1, self.channels, 1, 1)
        hl = hl * g_hl.view(-1, self.channels, 1, 1)
        hh = hh * g_hh.view(-1, self.channels, 1, 1)
        yh_new = [torch.stack([lh, hl, hh], dim=2)]
        x_high = self.idwt((ll, yh_new))
        # 奇数尺寸 fallback
        if x_high.shape[-2:] != x.shape[-2:]:
            x_high = F.interpolate(x_high, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x_high

    def forward(self, x):
        fi   = self._wavelet_highpass(x)
        u_cp = self.cp_fc(torch.cat([self.cp_gap(fi), self.cp_gmp(fi)], dim=1))
        u_sp = self.sp_fuse(fi)
        feat = u_cp * x + u_sp * x
        feat = self.aerf(feat)
        return self.out_conv(feat)
