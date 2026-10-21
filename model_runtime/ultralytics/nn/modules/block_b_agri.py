"""
Phase 8 Block B — agricultural SOTA modules (paper-faithful, runnable).

B1 YOLOv8-DML: C2f-DWR, MEFP (simplified), LSCD, WIoUv3 (loss flag).
B2 ESM-YOLOv11: EMA + Slim-neck (GSConv / VoVGSCSP).
B3 RLDD-YOLOv11n: SCSABlock + CARAFE upsample.

No hyperparameter tuning beyond paper defaults where stated.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import Bottleneck, C2f, DFL
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = (
    "DWR",
    "C2fDWR",
    "SPDConv",
    "GSConv",
    "GSBottleneck",
    "VoVGSCSP",
    "EMA",
    "SCSA",
    "SCSABlock",
    "CARAFE",
    "DetectLSCD",
)


# ---------------------------------------------------------------------------
# B1 — YOLOv8-DML
# ---------------------------------------------------------------------------


class DWR(nn.Module):
    """Dilated-wise residual block (YOLOv8-DML Fig.4a)."""

    def __init__(self, c: int):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1)
        self.dw1 = DWConv(c, c, 3, 1, d=1)
        self.dw3 = DWConv(c, c, 3, 1, d=3)
        self.dw5 = DWConv(c, c, 3, 1, d=5)
        self.conv2 = Conv(c, c, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.dw1(y) + self.dw3(y) + self.dw5(y)
        return self.conv2(y)


class C2fDWR(C2f):
    """C2f with DWR bottlenecks instead of standard Bottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DWR(self.c) for _ in range(n))


class SPDConv(nn.Module):
    """Space-to-depth conv for P2 branch in MEFP (scale=2)."""

    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.s = s
        self.conv = Conv(c1 * (s * s), c2, k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.s
        b, c, h, w = x.shape
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * s * s, h // s, w // s)
        return self.conv(x)


class DetectLSCD(Detect):
    """
    Lightweight shared convolutional detection head (YOLOv8-DML LSCD).
    Shared reg/cls stacks across P3/P4/P5 after per-level 1x1 + GN.
    Subclasses Detect so DetectionModel builds strides and runs bias_init.
    """

    def __init__(self, nc=80, ch=()):
        nn.Module.__init__(self)
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c_hidden = max(16, min(ch) // 4)
        self.stem = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, c_hidden, 1, 1, bias=False),
                nn.GroupNorm(16 if c_hidden >= 16 else 1, c_hidden),
                nn.SiLU(inplace=True),
            )
            for x in ch
        )
        self.cv2 = nn.Sequential(Conv(c_hidden, c_hidden, 3, 1), Conv(c_hidden, c_hidden, 3, 1), nn.Conv2d(c_hidden, 4 * self.reg_max, 1))
        self.cv3 = nn.Sequential(
            DWConv(c_hidden, c_hidden, 3),
            Conv(c_hidden, c_hidden, 1),
            DWConv(c_hidden, c_hidden, 3),
            Conv(c_hidden, c_hidden, 1),
            nn.Conv2d(c_hidden, self.nc, 1),
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            xi = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2(xi), self.cv3(xi)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x):
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Shared LSCD head: one cls/reg bias using smallest stride (P3)."""
        s = float(self.stride.min())
        self.cv2[-1].bias.data[:] = 1.0
        self.cv3[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


# ---------------------------------------------------------------------------
# B2 — ESM-YOLOv11 (GSConv / VoVGSCSP / EMA)
# ---------------------------------------------------------------------------


class GSConv(nn.Module):
    """Slim-neck GSConv (AlanLi1997/slim-neck-by-gsconv)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, act)
        self.cv2 = Conv(c_, c_, 5, 1, 2, c_, d, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class GSBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv_lighting = nn.Sequential(GSConv(c1, c_, 1, 1), GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class VoVGSCSP(nn.Module):
    """VoV-GSCSP neck block (ESM-YOLOv11)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.gc2 = GSConv(c1, c_, 3, 1, 1)
        self.m = C2f(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.m(self.cv1(x))
        y = self.gc2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


class EMA(nn.Module):
    """Efficient multi-scale attention (ESM-YOLOv11 Sec.2.2.3, grouped spatial-channel)."""

    def __init__(self, channels: int, factor: int = 32):
        super().__init__()
        self.groups = max(1, channels // factor)
        assert channels % self.groups == 0
        self.c_g = channels // self.groups
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Conv2d(self.c_g, self.c_g, 1, bias=False)
        self.gn = nn.GroupNorm(self.c_g, self.c_g)
        self.conv3x3 = nn.Conv2d(self.c_g, self.c_g, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups
        gx = x.reshape(b * g, self.c_g, h, w)
        x_h = self.pool_h(gx)
        x_w = self.pool_w(gx).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        attn = gx * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        y = self.gn(attn) + self.conv3x3(gx)
        return y.reshape(b, c, h, w)


# ---------------------------------------------------------------------------
# B3 — RLDD-YOLOv11n (SCSA + CARAFE)
# ---------------------------------------------------------------------------


class _SMSA(nn.Module):
    """Shareable multi-semantic spatial attention (SCSA paper §3.1, K=4)."""

    def __init__(self, dim: int, kernels=(3, 5, 7, 9)):
        super().__init__()
        self.k = len(kernels)
        assert dim % self.k == 0
        self.c_g = dim // self.k
        self.convs_h = nn.ModuleList(nn.Conv1d(self.c_g, self.c_g, k, padding=k // 2, groups=self.c_g) for k in kernels)
        self.convs_w = nn.ModuleList(nn.Conv1d(self.c_g, self.c_g, k, padding=k // 2, groups=self.c_g) for k in kernels)
        self.gn_h = nn.GroupNorm(self.k, dim)
        self.gn_w = nn.GroupNorm(self.k, dim)

    def _spatial_attn(self, seq: torch.Tensor, convs: nn.ModuleList, gn: nn.GroupNorm) -> torch.Tensor:
        parts = []
        for i, conv in enumerate(convs):
            parts.append(conv(seq[:, i * self.c_g : (i + 1) * self.c_g]))
        out = gn(torch.cat(parts, dim=1))
        return out.sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_h = x.mean(2)
        x_w = x.mean(3)
        attn_h = self._spatial_attn(x_h, self.convs_h, self.gn_h).unsqueeze(2)
        attn_w = self._spatial_attn(x_w, self.convs_w, self.gn_w).unsqueeze(3)
        return x * attn_h * attn_w


class _PCSA(nn.Module):
    """Progressive channel-wise self-attention (SCSA paper §3.2)."""

    def __init__(self, dim: int, pool_k: int = 7):
        super().__init__()
        self.pool_k = pool_k
        self.q = nn.Conv1d(dim, dim, 1, bias=False)
        self.k = nn.Conv1d(dim, dim, 1, bias=False)
        self.v = nn.Conv1d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pad = self.pool_k // 2
        xp = F.avg_pool2d(x, self.pool_k, stride=1, padding=pad)
        _, _, hp, wp = xp.shape
        n = hp * wp
        flat = xp.flatten(2)
        q = self.q(flat)
        k = self.k(flat)
        v = self.v(flat)
        scale = math.sqrt(c)
        attn = torch.softmax(q @ k.transpose(1, 2) / scale, dim=-1)
        xc = (attn @ v).reshape(b, c, hp, wp)
        gate = torch.sigmoid(F.adaptive_avg_pool2d(xc, 1))
        return x * gate


class SCSA(nn.Module):
    """Spatial-channel synergistic attention: PCSA(SMSA(X)) (Si et al. / RLDD)."""

    def __init__(self, dim: int):
        super().__init__()
        self.smsa = _SMSA(dim)
        self.pcsa = _PCSA(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pcsa(self.smsa(x))


class SCSABlock(nn.Module):
    """SCSA + residual bottleneck (RLDD neck, after C3k2 outputs)."""

    def __init__(self, c: int):
        super().__init__()
        self.attn = SCSA(c)
        self.ffn = nn.Sequential(Conv(c, c, 3, 1), Conv(c, c, 1, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.attn(x))


class CARAFE(nn.Module):
    """CARAFE-style 2x upsample (RLDD): learned content mask on upsampled features."""

    def __init__(self, c: int, scale: int = 2, **_kwargs):
        super().__init__()
        self.scale = scale
        self.mask = nn.Sequential(Conv(c, max(c // 4, 8), 3, 1), nn.Conv2d(max(c // 4, 8), c, 1, 1, 0), nn.Sigmoid())
        self.refine = Conv(c, c, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.scale
        up = F.interpolate(x, scale_factor=s, mode="nearest")
        m = F.interpolate(self.mask(x), scale_factor=s, mode="bilinear", align_corners=False)
        return self.refine(up * m)
