"""
BLPA: Beer-Lambert Polarized Attention (Phase 2)
================================================
针对小目标病斑检测 (median 14x14, 90% < 32x32) 的物理启发注意力。

物理对应:
    A(x,y) = sum_c [ epsilon(c) * concentration(x,y) * light_path(x,y) ]
    其中:
        epsilon(c)        : 通道吸收系数 (channel-wise absorption)
        concentration(x,y): 病害浓度图 (per-pixel disease concentration)
        light_path(x,y)   : 光程修正图 (per-pixel light path correction)

针对小目标的 3 项改造 (相比原 PSA/BLPA):
    1. 去全局池化 -> 改用 3x3 局部聚合, 保留小目标定位
    2. epsilon 维度减半 (eps_dim = C // 2), 病斑色素谱简单不需太多通道
    3. light_path 用 1x1 conv 像素级估计, 不在空间维度做 attention

参数量估算 (C=64, eps_dim=32):
    eps_branch:   C -> eps_dim -> C  ~= 64*32 + 32*64 = 4K
    conc_branch:  3x3 conv C -> C    ~= 64*64*9      = 37K
    light_branch: 1x1 conv C -> 1    ~= 64           = 64
    fuse:         1x1 conv C -> C    ~= 64*64        = 4K
    总计 ~= 45K (比 PathoAgent 单层 ~80K 小一半)

C=64 单层 ~45K, P2+P3 双层 ~90K, 远低于 PathoAgent 全加 ~330K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BLPA(nn.Module):
    """Beer-Lambert Polarized Attention - 小目标优化版.

    Args:
        c (int):       输入/输出通道数.
        eps_dim (int): epsilon (吸收系数) 的中间维度. 默认 c // 2.
        local_k (int): concentration 局部聚合核大小. 默认 3.
    """

    def __init__(self, c: int, eps_dim: int = None, local_k: int = 3):
        super().__init__()
        self.c = c
        self.eps_dim = eps_dim if eps_dim is not None else max(c // 2, 8)
        pad = local_k // 2

        # epsilon(c): 通道吸收系数 - 通道级 attention, 不动空间
        # 输入: [B,C,H,W] -> GAP -> [B,C,1,1] -> 降维 -> 升维 -> sigmoid
        # 注意: 这里只对通道维度做 GAP, 不影响空间分辨率, 对小目标无害
        self.eps_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, self.eps_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.eps_dim, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # concentration(x,y): 病害浓度图 - 局部 3x3 聚合, 保留小目标
        # 输入: [B,C,H,W] -> 3x3 DepthwiseConv -> BN -> GELU -> 1x1 -> Sigmoid
        # 关键: 不用 global pooling, 14x14 的病斑信号不会被稀释
        self.conc_branch = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=local_k, padding=pad, groups=c, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # light_path(x,y): 光程修正图 - 像素级标量
        # 输入: [B,C,H,W] -> 1x1 -> [B,1,H,W] -> Sigmoid
        # 物理意义: 每个像素自己的光程系数, 越大病害"看上去越深"
        self.light_branch = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # 融合后的 1x1 conv (类似 transformer 里的 output projection)
        self.fuse = nn.Conv2d(c, c, kernel_size=1, bias=True)

        # 残差 gate (可学习的标量, 初始 0, 让模型决定要不要用)
        # 这是 CrossFormer / DiT 风格的 zero-init gate, 训练初期等价于 identity
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        # Beer-Lambert 三因子分解
        eps = self.eps_branch(x)       # [B, C, 1, 1]   通道吸收谱
        conc = self.conc_branch(x)     # [B, C, H, W]   每像素每通道浓度
        light = self.light_branch(x)   # [B, 1, H, W]   每像素光程系数

        # A = epsilon * c * l (三者外积)
        # 广播: [B,C,1,1] * [B,C,H,W] * [B,1,H,W] -> [B,C,H,W]
        absorption = eps * conc * light

        # 注意力权重作用于原特征
        out = x * absorption
        out = self.fuse(out)

        # zero-init residual: 训练初期 = x, 避免一上来就破坏预训练特征
        return x + self.gamma * out


# ============================================================
# Self-test
# ============================================================
def _self_test():
    print("=" * 60)
    print("BLPA self-test")
    print("=" * 60)

    test_cases = [
        # (B, C, H, W, name)
        (2, 64, 160, 160, "P2 (small targets)"),
        (2, 64,  80,  80, "P3 (medium targets)"),
        (2, 128, 40,  40, "P4 (large targets)"),
        (2, 256, 20,  20, "P5 (global)"),
    ]

    for B, C, H, W, name in test_cases:
        m = BLPA(c=C)
        x = torch.randn(B, C, H, W)
        y = m(x)

        # check shape
        assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"

        # check no NaN
        assert not torch.isnan(y).any(), "NaN in output!"
        assert not torch.isinf(y).any(), "Inf in output!"

        # 因为 gamma init = 0, 一开始 y 应该 == x
        assert torch.allclose(y, x, atol=1e-6), \
            f"zero-init failed: max diff = {(y - x).abs().max().item()}"

        # param count
        n_params = sum(p.numel() for p in m.parameters())

        print(f"  [{name:20s}] in={tuple(x.shape)} out={tuple(y.shape)} "
              f"params={n_params:6d} ({n_params/1e3:.1f}K)  OK")

    # gradient test
    print("\n  gradient flow test...")
    m = BLPA(c=64)
    x = torch.randn(2, 64, 80, 80, requires_grad=True)
    y = m(x)
    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    # 因为 gamma=0, 梯度只通过残差路径 (x.grad = 1/N), 这是正确的
    print(f"  x.grad mean = {x.grad.mean().item():.6f}  OK")

    # gamma 梯度: 应该非零 (说明 BLPA 这条路径也在学)
    assert m.gamma.grad is not None
    assert not torch.isnan(m.gamma.grad).any()
    print(f"  gamma.grad = {m.gamma.grad.item():.6f}  OK")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()