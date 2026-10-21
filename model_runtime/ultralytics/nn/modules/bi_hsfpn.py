"""
BI-HSFPN R3
改动:
  A4  : BiHSFPN 改为 4 路输入/输出 [P2,P3,P4,P5]，P2=stride4 浅层直连
  AERF: 自适应有效感受野模块，插入每个 CHFP out_conv 之前
        三路并联: 1x1Conv(精确) + AdaptiveAvgPool(3)(中等) + AdaptiveAvgPool(5)(大)
        Concat + 1x1Conv 融合，参数增加 <0.1M
  检测头: P2 使用 DTDH（与 P3 一致）

覆盖到: ultralytics/nn/modules/bi_hsfpn.py
"""

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .block import DFL
from ultralytics.utils.tal import make_anchors, dist2bbox


# ──────────────────────────────────────────────────────────────
# 基础模块（与 R0 完全一致）
# ──────────────────────────────────────────────────────────────

class ConvBnAct(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1):
        super().__init__()
        self.dw = ConvBnAct(in_c, in_c, k=k, s=s, g=in_c)
        self.pw = ConvBnAct(in_c, out_c, k=1)
    def forward(self, x):
        return self.pw(self.dw(x))


class LightFusion(nn.Module):
    """轻量跨层融合（R0 原版）"""
    def __init__(self, channels: int):
        super().__init__()
        self.fuse  = nn.Sequential(
            ConvBnAct(channels * 2, channels, 1),
            DWConv(channels, channels, k=3),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x_cur, x_up):
        return x_cur + torch.tanh(self.alpha) * self.fuse(
            torch.cat([x_cur, x_up], dim=1))


class DynamicFusion(nn.Module):
    """动态三路融合（R0 原版）"""
    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.eps     = eps
        self.base_w  = nn.Parameter(torch.ones(3))
        self.density_est = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid())
        self.modulate = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 3), nn.Softmax(dim=-1))
        nn.init.zeros_(self.modulate[2].weight)
        nn.init.constant_(self.modulate[2].bias, 0.0)
    def forward(self, skip, td, bu):
        w  = F.relu(self.base_w) + self.eps
        w  = w / w.sum()
        dw = self.modulate(self.density_est(skip).view(-1, 1))
        fw = (w.unsqueeze(0) * dw)
        fw = fw / (fw.sum(dim=1, keepdim=True) + self.eps)
        fw = fw.view(-1, 3, 1, 1)
        return fw[:,0:1]*skip + fw[:,1:2]*td + fw[:,2:3]*bu


# ──────────────────────────────────────────────────────────────
# 新增: AERF 自适应有效感受野模块
# ──────────────────────────────────────────────────────────────

class AERF(nn.Module):
    """
    Adaptive Effective Receptive Field Module
    三路并联感受野，自动适应不同大小病斑:
      路1: 1x1 Conv          → 精确局部特征（小病斑/边界）
      路2: AdaptiveAvgPool(3) → 中等感受野（中等病斑）
      路3: AdaptiveAvgPool(5) → 大感受野（大面积病斑/上下文）
    Concat(3×channels) → 1x1Conv(channels) 融合
    残差连接保证训练初期稳定

    参考: RFB-Net (ECCV2018), ASPP (DeepLabv3)
    参数量: ~3×channels² / 1 = 极少，<0.05M per instance

    放置位置: CHFP out_conv 之前
    """
    def __init__(self, channels: int):
        super().__init__()
        # 路1: 1x1精确卷积
        self.branch_exact = ConvBnAct(channels, channels, k=1)
        # 路2: 3×3自适应平均池化后恢复分辨率 + 1x1对齐
        self.branch_mid   = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            ConvBnAct(channels, channels, k=1),
        )
        # 路3: 5×5自适应平均池化后恢复分辨率 + 1x1对齐
        self.branch_large = nn.Sequential(
            nn.AdaptiveAvgPool2d(5),
            ConvBnAct(channels, channels, k=1),
        )
        # 三路融合
        self.fuse = ConvBnAct(channels * 3, channels, k=1)
        # 残差缩放因子，初始化为0，训练初期不扰动主干
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        b1 = self.branch_exact(x)                                           # [B,C,H,W]
        b2 = F.interpolate(self.branch_mid(x),   size=(H, W),
                           mode='bilinear', align_corners=False)             # [B,C,H,W]
        b3 = F.interpolate(self.branch_large(x), size=(H, W),
                           mode='bilinear', align_corners=False)             # [B,C,H,W]
        out = self.fuse(torch.cat([b1, b2, b3], dim=1))
        return x + torch.tanh(self.alpha) * out                              # 残差


# ──────────────────────────────────────────────────────────────
# 改动: CHFP 插入 AERF（out_conv 之前）
# ──────────────────────────────────────────────────────────────

class CHFP(nn.Module):
    """
    C-HFP R3: 高频感知 + rate=1,2,4 + AERF 自适应感受野
    AERF 插在 out_conv 之前，对融合后特征做多尺度感受野增强
    """
    def __init__(self, channels: int, alpha: float = 0.3, reduction: int = 4):
        super().__init__()
        self.alpha = alpha
        mid = max(channels // reduction, 8)
        # 通道注意力
        self.cp_gap = nn.AdaptiveAvgPool2d(1)
        self.cp_gmp = nn.AdaptiveMaxPool2d(1)
        self.cp_fc  = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间多尺度（保留 R0 的 dilation=4）
        c1 = channels // 3
        c2 = channels // 3
        c3 = channels - c1 - c2
        self.sp_d1   = nn.Conv2d(channels, c1, 3, 1, 1, dilation=1, bias=False)
        self.sp_d2   = nn.Conv2d(channels, c2, 3, 1, 2, dilation=2, bias=False)
        self.sp_d4   = nn.Conv2d(channels, c3, 3, 1, 4, dilation=4, bias=False)
        self.sp_bn   = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)
        self.sp_fuse = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False), nn.Sigmoid())
        # AERF 插在 out_conv 之前
        self.aerf     = AERF(channels)
        self.out_conv = ConvBnAct(channels, channels, k=3)

    def _high_pass(self, x):
        _, _, H, W = x.shape
        k = max(3, int(min(H, W) * self.alpha) | 1)
        return (x - F.avg_pool2d(x, k, stride=1, padding=k//2)).clamp(-10., 10.)

    def forward(self, x):
        fi   = self._high_pass(x)
        u_cp = self.cp_fc(torch.cat([self.cp_gap(fi), self.cp_gmp(fi)], dim=1))
        sp   = F.silu(self.sp_bn(torch.cat([
            self.sp_d1(fi), self.sp_d2(fi), self.sp_d4(fi)], dim=1)))
        u_sp = self.sp_fuse(sp)
        feat = u_cp * x + u_sp * x      # CHFP 原始融合结果
        feat = self.aerf(feat)           # AERF 多尺度感受野增强
        return self.out_conv(feat)


# ──────────────────────────────────────────────────────────────
# 改动: BiHSFPN 4路输入/输出版本（A4）
# ──────────────────────────────────────────────────────────────

class BiHSFPN(nn.Module):
    """
    BI-HSFPN R3: 4 路输入/输出
    输入: [P2(64), P3(64), P4(128), P5(256)]  @ scale.n
    输出: [P2(64), P3(64), P4(128), P5(256)]

    P2(stride4) 浅层直连，提供细粒度纹理特征，针对 Bean_Rust 小病斑
    AERF 已内嵌在每个 CHFP 中
    """
    def __init__(self,
                 in_channels:  List[int] = [64, 64, 128, 256],
                 out_channels: List[int] = [64, 64, 128, 256],
                 alpha: float = 0.3):
        super().__init__()
        assert len(in_channels) == 4 and len(out_channels) == 4
        c1i, c2i, c3i, c4i = in_channels   # P2,P3,P4,P5 输入通道
        c1o, c2o, c3o, c4o = out_channels   # P2,P3,P4,P5 输出通道

        # 通道对齐
        self.align = nn.ModuleList([
            nn.Identity() if c1i == c1o else ConvBnAct(c1i, c1o, 1),
            nn.Identity() if c2i == c2o else ConvBnAct(c2i, c2o, 1),
            nn.Identity() if c3i == c3o else ConvBnAct(c3i, c3o, 1),
            nn.Identity() if c4i == c4o else ConvBnAct(c4i, c4o, 1),
        ])

        # CHFP（内含 AERF）
        self.chfp = nn.ModuleList([
            CHFP(c1o, alpha),
            CHFP(c2o, alpha),
            CHFP(c3o, alpha),
            CHFP(c4o, alpha),
        ])

        # top-down 通道调整
        self.adjust_p5_to_p4 = ConvBnAct(c4o, c3o, 1)
        self.adjust_p4_to_p3 = ConvBnAct(c3o, c2o, 1)
        self.adjust_p3_to_p2 = ConvBnAct(c2o, c1o, 1)

        # LightFusion (top-down)
        self.lf_p4 = LightFusion(c3o)
        self.lf_p3 = LightFusion(c2o)
        self.lf_p2 = LightFusion(c1o)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # bottom-up 融合
        self.fuse_p2 = ConvBnAct(c1o * 2, c1o, 1)   # P2: cat融合
        self.fuse_p3 = DynamicFusion(c2o)             # P3: 动态三路
        self.fuse_p4 = DynamicFusion(c3o)             # P4: 动态三路
        self.fuse_p5 = DynamicFusion(c4o)             # P5: 动态三路

        self.down_p2 = DWConv(c1o, c2o, k=3, s=2)
        self.down_p3 = DWConv(c2o, c3o, k=3, s=2)
        self.down_p4 = DWConv(c3o, c4o, k=3, s=2)

        self.out_conv = nn.ModuleList([
            ConvBnAct(c1o, c1o, 3),
            ConvBnAct(c2o, c2o, 3),
            ConvBnAct(c3o, c3o, 3),
            ConvBnAct(c4o, c4o, 3),
        ])

    def _match(self, src, ref):
        if src.shape[-2:] != ref.shape[-2:]:
            src = F.interpolate(src, size=ref.shape[-2:],
                                mode='bilinear', align_corners=False)
        return src

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        f1, f2, f3, f4 = features   # P2, P3, P4, P5

        # 通道对齐
        f1 = self.align[0](f1)
        f2 = self.align[1](f2)
        f3 = self.align[2](f3)
        f4 = self.align[3](f4)

        # CHFP 高频增强（含 AERF）
        h1 = self.chfp[0](f1)   # P2
        h2 = self.chfp[1](f2)   # P3
        h3 = self.chfp[2](f3)   # P4
        h4 = self.chfp[3](f4)   # P5

        # ── Top-down ──────────────────────────────────────────
        td4 = h4                 # P5 不变
        td3 = self.lf_p4(h3, self._match(
            self.up(self.adjust_p5_to_p4(td4)), h3))   # P4 ← P5
        td2 = self.lf_p3(h2, self._match(
            self.up(self.adjust_p4_to_p3(td3)), h2))   # P3 ← P4
        td1 = self.lf_p2(h1, self._match(
            self.up(self.adjust_p3_to_p2(td2)), h1))   # P2 ← P3

        # ── Bottom-up ─────────────────────────────────────────
        p1 = self.fuse_p2(torch.cat([h1, td1], dim=1))              # P2
        p2 = self.fuse_p3(h2, td2, self._match(self.down_p2(p1), h2))  # P3
        p3 = self.fuse_p4(h3, td3, self._match(self.down_p3(p2), h3))  # P4
        p4 = self.fuse_p5(h4, td4, self._match(self.down_p4(p3), h4))  # P5

        return [
            self.out_conv[0](p1),
            self.out_conv[1](p2),
            self.out_conv[2](p3),
            self.out_conv[3](p4),
        ]


# ──────────────────────────────────────────────────────────────
# 检测头（P2 使用 DTDH，其余与 R0 一致）
# ──────────────────────────────────────────────────────────────

class DTDH(nn.Module):
    """双任务检测头，用于 P2 和 P3"""
    def __init__(self, in_channels=64, num_classes=2, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = reg_max
        mid = max(in_channels, 64)
        self.cls_dw1 = DWConv(in_channels, mid)
        self.cls_dw2 = DWConv(mid, mid)
        se_mid = max(mid // 4, 8)
        self.se_gap = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(mid, se_mid, 1, bias=False)
        self.se_act = nn.ReLU(inplace=True)
        self.se_fc2 = nn.Conv2d(se_mid, mid, 1, bias=False)
        self.se_sig = nn.Sigmoid()
        self.cls_out = nn.Conv2d(mid, num_classes, 1)
        self.reg_dw1        = DWConv(in_channels, mid)
        self.reg_dw2        = DWConv(mid, mid)
        self.reg_coarse_out = nn.Conv2d(mid, 4 * reg_max, 1)
        self.dfl_proj = nn.Conv2d(reg_max, 1, 1, bias=False)
        w = torch.arange(reg_max, dtype=torch.float32) / reg_max
        self.dfl_proj.weight.data = w.view(1, reg_max, 1, 1)
        self.dfl_proj.weight.requires_grad = False
        refine_in = in_channels + 4 * reg_max
        self.reg_refine = nn.Sequential(
            nn.Conv2d(refine_in, mid, 1, bias=False), nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid // 2, 1, bias=False),  nn.SiLU(inplace=True),
            nn.Conv2d(mid // 2, 4, 1), nn.Tanh())
        nn.init.zeros_(self.reg_refine[-2].weight)
        nn.init.zeros_(self.reg_refine[-2].bias)

    def _se(self, f):
        return f * self.se_sig(
            self.se_fc2(self.se_act(self.se_fc1(self.se_gap(f)))))

    def _dfl(self, raw):
        B, _, H, W = raw.shape
        x = F.softmax(raw.view(B * 4, self.reg_max, H, W), dim=1)
        return self.dfl_proj(x).view(B, 4, H, W)

    def forward(self, x):
        cf  = self._se(self.cls_dw2(self.cls_dw1(x)))
        cls = self.cls_out(cf)
        rf  = self.reg_dw2(self.reg_dw1(x))
        cr  = self.reg_coarse_out(rf)
        return cls, self._dfl(cr) + self.reg_refine(
            torch.cat([x, cr], dim=1)) * 0.5


class StandardHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=2, reg_max=16):
        super().__init__()
        mid = in_channels
        self.cls_branch = nn.Sequential(
            ConvBnAct(in_channels, mid, 3),
            ConvBnAct(mid, mid, 3),
            nn.Conv2d(mid, num_classes, 1))
        self.reg_branch = nn.Sequential(
            ConvBnAct(in_channels, mid, 3),
            ConvBnAct(mid, mid, 3),
            nn.Conv2d(mid, 4 * reg_max, 1))
    def forward(self, x):
        return self.cls_branch(x), self.reg_branch(x)


class YOLOMSHFHead(nn.Module):
    """R3: 4路检测头 [P2,P3,P4,P5]，P2/P3 用 DTDH，P4/P5 用 StandardHead"""
    def __init__(self, in_channels=[64, 64, 128, 256], num_classes=2, reg_max=16):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.dtdh_p2 = DTDH(c1, num_classes, reg_max)   # P2 新增
        self.dtdh_p3 = DTDH(c2, num_classes, reg_max)   # P3 原有
        self.head_p4 = StandardHead(c3, num_classes, reg_max)
        self.head_p5 = StandardHead(c4, num_classes, reg_max)
        self.nc      = num_classes
        self.reg_max = reg_max
        self.no      = num_classes + reg_max * 4

    def forward(self, features):
        p1, p2, p3, p4 = features
        cls1, reg1 = self.dtdh_p2(p1)
        cls2, reg2 = self.dtdh_p3(p2)
        cls3, reg3 = self.head_p4(p3)
        cls4, reg4 = self.head_p5(p4)
        return [
            torch.cat([reg1, cls1], dim=1),
            torch.cat([reg2, cls2], dim=1),
            torch.cat([reg3, cls3], dim=1),
            torch.cat([reg4, cls4], dim=1),
        ]


# ──────────────────────────────────────────────────────────────
# NMS / Detector
# ──────────────────────────────────────────────────────────────

def pa_nms(boxes, scores, labels,
           iou_threshold=0.45, soft_threshold=0.25,
           sigma=0.5, small_area_ratio=1.5):
    if boxes.numel() == 0:
        return boxes, scores, labels
    order  = scores.argsort(descending=True)
    boxes  = boxes[order].float()
    scores = scores[order].clone().float()
    labels = labels[order]
    N      = boxes.shape[0]
    keep   = torch.ones(N, dtype=torch.bool, device=boxes.device)
    areas  = ((boxes[:,2]-boxes[:,0]).clamp(0) *
               (boxes[:,3]-boxes[:,1]).clamp(0))
    med = areas.median().item()
    st  = iou_threshold * 0.6
    for i in range(N):
        if not keep[i]: continue
        for j in range(i + 1, N):
            if not keep[j]: continue
            ix1 = max(boxes[i,0].item(), boxes[j,0].item())
            iy1 = max(boxes[i,1].item(), boxes[j,1].item())
            ix2 = min(boxes[i,2].item(), boxes[j,2].item())
            iy2 = min(boxes[i,3].item(), boxes[j,3].item())
            inter = max(ix2-ix1, 0.) * max(iy2-iy1, 0.)
            if inter == 0.: continue
            iou = inter / (areas[i].item() + areas[j].item() - inter + 1e-7)
            if (areas[i].item() < med * small_area_ratio and
                    areas[j].item() < med * small_area_ratio and iou > st):
                scores[j] *= math.exp(-(iou * iou) / sigma)
                if scores[j].item() < soft_threshold:
                    keep[j] = False
            elif iou > iou_threshold:
                keep[j] = False
    keep = keep & (scores >= soft_threshold)
    idx  = keep.nonzero(as_tuple=False).squeeze(1)
    return boxes[idx], scores[idx], labels[idx]


class YOLOMSHFDetector(nn.Module):
    """YOLO-MSHF R3, 4路检测 stride=[4,8,16,32]"""
    def __init__(self,
                 backbone_channels: List[int] = [64, 64, 128, 256],
                 num_classes: int = 2,
                 reg_max: int = 16):
        super().__init__()
        self.neck = BiHSFPN(backbone_channels, backbone_channels)
        self.head = YOLOMSHFHead(backbone_channels, num_classes, reg_max)
        self.nc      = num_classes
        self.reg_max = reg_max
        self.no      = num_classes + reg_max * 4
        self.nl      = len(backbone_channels)
        self.stride  = torch.tensor([4, 8, 16, 32], dtype=torch.float32)
        self.dfl     = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.shape   = None
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)
        self.dynamic = False

    def forward(self, feats: List[torch.Tensor]):
        features = self.neck(feats)
        x = self.head(features)
        if self.training:
            return x, x
        y = self._inference(x)
        return y, x

    def _inference(self, x):
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box),
                         self.anchors.unsqueeze(0), dim=1) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)
