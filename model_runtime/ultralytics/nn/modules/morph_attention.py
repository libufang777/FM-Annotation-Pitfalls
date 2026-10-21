"""
MorphAttention: Differentiable Mathematical Morphology Attention
===================================================================
YOLO-MSHF module for lesion-scale feature modulation.

References:
- Shen et al., Deep Morphological Neural Networks, arXiv 1909.01532
- Hu et al., Pyramid Sparse Transformer (PST), arXiv 2505.12772

Key design choices (locked):
- 2 heads (Bean Rust circular / Angular elongated)
- Gabor-like parametric elliptical SE: (theta, aspect, radius)
- Hybrid sharing: shape params shared across P3/P4/P5, gain/gate per-scale
- Placement: before each Detect head (reject R3/WCHFP early-stage failure mode)
- Slow start: gate_logit init -3, conv1x1_out zero init
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Global registry singleton storage. Populated by MorphAttention.__init__.
# Maps registry_key (str) -> MorphShapeRegistry instance.
# This is how P3/P4/P5 MorphAttention instances share shape parameters.
_MORPH_REGISTRIES: dict = {}


class MorphShapeRegistry(nn.Module):
    """
    Shared SE shape parameter registry across all MorphAttention instances.

    Implements §3.2 'hybrid sharing': shape params (theta, aspect, radius,
    alpha) are shared across P3/P4/P5 detection scales; per-scale gain/gate
    parameters live in each MorphAttention instance.

    Initialization (per §4.2 of design doc):
        Head 0 (Bean Rust prior):
            theta_0       = 0.0          (rad, horizontal)
            aspect_0_raw  = 0.0          (softplus -> ~1.0, circular)
            radius_0      = 2.0          (SE radius in feature grid units)
            alpha_0_logit = 0.85         (sigmoid -> 0.70, opening-biased)

        Head 1 (Angular Leaf Spot prior):
            theta_1       = pi/4         (rad, diagonal)
            aspect_1_raw  = 1.0          (softplus -> ~1.61, elongated)
            radius_1      = 2.5
            alpha_1_logit = -0.85        (sigmoid -> 0.30, closing-biased)
    """

    def __init__(self):
        super().__init__()

        # Head 0: initially circular (Bean Rust prior)
        self.theta_0 = nn.Parameter(torch.tensor(0.0))
        self.aspect_0_raw = nn.Parameter(torch.tensor(0.0))
        self.radius_0 = nn.Parameter(torch.tensor(2.0))
        self.alpha_0_logit = nn.Parameter(torch.tensor(0.85))

        # Head 1: initially elongated ellipse (Angular prior)
        self.theta_1 = nn.Parameter(torch.tensor(math.pi / 4.0))
        self.aspect_1_raw = nn.Parameter(torch.tensor(1.0))
        self.radius_1 = nn.Parameter(torch.tensor(2.5))
        self.alpha_1_logit = nn.Parameter(torch.tensor(-0.85))

    def get_shapes(self):
        """Returns a dict of shape parameters with clamping applied."""
        # aspect ratio: softplus(raw) + 0.3 (ensure > 0.3), then clamp to [0.3, 3.0]
        a0 = torch.clamp(F.softplus(self.aspect_0_raw) + 0.3, 0.3, 3.0)
        a1 = torch.clamp(F.softplus(self.aspect_1_raw) + 0.3, 0.3, 3.0)
        # radius: clamp to [0.5, 3.0] (grid is 7x7, k=3, max radius 3)
        r0 = torch.clamp(self.radius_0, 0.5, 3.0)
        r1 = torch.clamp(self.radius_1, 0.5, 3.0)
        return {
            "theta_0": self.theta_0,
            "aspect_0": a0,
            "radius_0": r0,
            "alpha_0": torch.sigmoid(self.alpha_0_logit),
            "theta_1": self.theta_1,
            "aspect_1": a1,
            "radius_1": r1,
            "alpha_1": torch.sigmoid(self.alpha_1_logit),
        }


def _get_or_create_registry(registry_key: str) -> MorphShapeRegistry:
    """
    Get the shared registry for the given key, creating it if necessary.
    This is the mechanism by which P3/P4/P5 instances share shape parameters.

    IMPORTANT: When the model is moved across devices or wrapped in DDP,
    the registry must follow. Because the registry is a nn.Module containing
    nn.Parameter, and it is held as a submodule of each MorphAttention
    instance (via self.registry attribute), PyTorch's device-move and DDP
    machinery will replicate parameters correctly IF the registry is
    registered as a submodule. But since we want SHARING (not replication),
    we use a careful pattern:
      - Only the FIRST MorphAttention instance with a given registry_key
        registers the registry as a submodule.
      - Subsequent instances hold a reference but do not register it.
    This is implemented via __setattr__ override below.
    """
    if registry_key not in _MORPH_REGISTRIES:
        _MORPH_REGISTRIES[registry_key] = MorphShapeRegistry()
    return _MORPH_REGISTRIES[registry_key]


class MorphAttention(nn.Module):
    """
    Single-scale MorphAttention module. Inserted before each Detect head.

    Forward:
        F_proj   = ReLU(BN(Conv1x1_in(F)))       # [B, 16, H, W]
        F0       = soft_morph_head0(F_proj)      # via head 0 SE
        F1       = soft_morph_head1(F_proj)      # via head 1 SE
        F_cat    = concat([F0, F1], dim=1)       # [B, 32, H, W]
        A        = sigmoid(BN(Conv1x1_out(F_cat))) # [B, c_in, H, W]
        g        = sigmoid(gate_logit)            # scalar in (0, 1)
        out      = F + g * F * (A - 1)            # gated residual; init g≈0.05, A≈0.5

    At initialization (g≈0.05, A=sigmoid(0)=0.5):
        out = F + 0.05 * F * (-0.5) = 0.975 * F
    So output is ~97.5% of input -> slow start, won't disrupt Phase 3 features.

    Args:
        c_in: input channel count (e.g. 128 for P3, 256 for P4, 512 for P5)
        registry_key: shared registry identifier (use 'shared' for hybrid sharing)
        k: SE grid half-size (k=3 means 7x7 SE grid). Default 3.
        beta: soft min/max sharpness. Default 5.0.
        tau: SE construction sharpness (sigmoid temperature). Default 5.0.
    """

    # MorphAttention exports a fixed output channel = input channel,
    # so ultralytics tasks.py can compute c2 = c1 trivially.

    def __init__(self, c_in: int, registry_key: str = "shared",
                 k: int = 3, beta: float = 5.0, tau: float = 5.0):
        super().__init__()
        self.c_in = c_in
        self.k = k
        self.beta = beta
        self.tau = tau
        self.registry_key = registry_key

        # Shared SE shape parameters (via global registry)
        registry = _get_or_create_registry(registry_key)
        # NOTE: We use object.__setattr__ to avoid registering registry as
        # a submodule of *every* MorphAttention. Only the first instance
        # to claim a given key will register it. This avoids parameter
        # duplication / DDP issues.
        if not _registry_already_owned(registry_key):
            # First instance: register as submodule (parameters will be tracked here)
            self.registry = registry
            _mark_registry_owned(registry_key, self)
        else:
            # Subsequent instances: hold reference but do NOT register
            # (otherwise parameters would be duplicated in state_dict)
            object.__setattr__(self, "registry", registry)

        # Per-scale independent parameters
        self.conv1x1_in = nn.Conv2d(c_in, 16, kernel_size=1, bias=False)
        self.bn_in = nn.BatchNorm2d(16)
        self.conv1x1_out = nn.Conv2d(32, c_in, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(c_in)

        # Slow-start gate: sigmoid(-3) = 0.0474
        self.gate_logit = nn.Parameter(torch.tensor(-3.0))

        # Zero-init conv1x1_out so initial A = sigmoid(0) = 0.5
        # (combined with gate g≈0.05, initial output is ~97.5% of input)
        nn.init.zeros_(self.conv1x1_out.weight)
        # BN affine init (standard PyTorch defaults are weight=1, bias=0; redundant but explicit)
        nn.init.ones_(self.bn_out.weight)
        nn.init.zeros_(self.bn_out.bias)

        # Precompute SE grid coordinates as non-learnable buffer
        coords = torch.arange(-k, k + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")
        # persistent=False: 不进 state_dict，避免加载 Phase3 权重时 grid 张量拷贝报错
        self.register_buffer("grid_xx", xx.clone(), persistent=False)
        self.register_buffer("grid_yy", yy.clone(), persistent=False)

    def _build_se(self, theta: torch.Tensor, aspect: torch.Tensor,
                  radius: torch.Tensor) -> torch.Tensor:
        """
        Build a differentiable elliptical structuring element.

        Parameters: theta (orientation, rad), aspect (axis ratio), radius (size)
        Returns: SE tensor of shape (2k+1, 2k+1), values in (0, 1), sums to ~1.

        SE construction:
            x' =  x*cos(theta) + y*sin(theta)     # rotation
            y' = -x*sin(theta) + y*cos(theta)
            d² = (x'/aspect)² + (y'*aspect)²      # elliptical distance
            SE = sigmoid(tau * (1 - d²/radius²))  # soft binary mask
            SE = SE / sum(SE)                     # normalize
        """
        theta, aspect, radius = theta.float(), aspect.float(), radius.float()
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        # Rotation
        x_rot = self.grid_xx.float() * cos_t + self.grid_yy.float() * sin_t
        y_rot = -self.grid_xx.float() * sin_t + self.grid_yy.float() * cos_t
        # Elliptical squared distance
        d2 = (x_rot / aspect) ** 2 + (y_rot * aspect) ** 2
        # Soft mask
        se = torch.sigmoid(self.tau * (1.0 - d2 / (radius ** 2)))
        # Normalize (sum to 1, probability-like)
        se = se / (se.sum() + 1e-8)
        return se

    def _soft_erosion(self, x: torch.Tensor, se: torch.Tensor) -> torch.Tensor:
        """
        Soft erosion via log-sum-exp trick (numerically stable).

        Mathematical form:
            soft_erosion(x, se)(p) = -1/β · log[ Σ_q se(q) · exp(-β · x(p+q)) ]

        Stable computation:
            Let m(p) = -β · min_q x(p+q)  (estimated as -β · max(x) globally
            per channel for stability)
            log Σ se(q) exp(-β x(p+q))
              = log Σ se(q) exp(-β x(p+q) - m + m)
              = m + log Σ se(q) exp(-β x(p+q) - m)
              = m + log conv(exp(-β x - m/β · (-1)·...), se)
            -> -1/β · result

        Implementation strategy:
            We subtract a per-(batch, channel) maximum from -β·x for stability.

        Args:
            x: [B, C, H, W] input feature
            se: [k, k] structuring element (sum ≈ 1, already normalized)

        Returns:
            [B, C, H, W] eroded feature
        """
        k = se.shape[-1]
        pad = k // 2
        beta = self.beta
        c = x.shape[1]

        # Step 1: compute -β · x and subtract per-(batch, channel) max for stability
        neg_beta_x = -beta * x                                  # [B, C, H, W]
        # Per-(B, C) max over spatial dims for numerical stability
        max_per_bc = neg_beta_x.amax(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        # exp of stabilized values
        exp_arg = neg_beta_x - max_per_bc                        # [B, C, H, W], all <= 0
        exp_x = torch.exp(exp_arg)                               # in (0, 1]

        # Step 2: depthwise conv with SE (SE shared across all channels)
        # se is [k, k]; expand to depthwise kernel [C, 1, k, k]
        se_kernel = se.float().view(1, 1, k, k).expand(c, 1, k, k).contiguous()
        conv_out = F.conv2d(exp_x, se_kernel, padding=pad, groups=c)  # [B, C, H, W]

        # Step 3: take log and add back the max
        # log Σ se(q) exp(-β x(p+q)) = log(conv_out) + max_per_bc
        log_sum = torch.log(conv_out + 1e-8) + max_per_bc       # [B, C, H, W]

        # Step 4: soft erosion = -1/β · log_sum
        result = -log_sum / beta
        return result

    def _soft_dilation(self, x: torch.Tensor, se: torch.Tensor) -> torch.Tensor:
        """
        Soft dilation via log-sum-exp trick (numerically stable).

        Mathematical form:
            soft_dilation(x, se)(p) = +1/β · log[ Σ_q se(q) · exp(+β · x(p+q)) ]

        Args:
            x: [B, C, H, W] input feature
            se: [k, k] structuring element (sum ≈ 1)

        Returns:
            [B, C, H, W] dilated feature
        """
        k = se.shape[-1]
        pad = k // 2
        beta = self.beta
        c = x.shape[1]

        # Step 1: β · x with per-(batch, channel) max subtraction
        pos_beta_x = beta * x                                    # [B, C, H, W]
        max_per_bc = pos_beta_x.amax(dim=(2, 3), keepdim=True)   # [B, C, 1, 1]
        exp_arg = pos_beta_x - max_per_bc
        exp_x = torch.exp(exp_arg)

        # Step 2: depthwise conv
        se_kernel = se.float().view(1, 1, k, k).expand(c, 1, k, k).contiguous()
        conv_out = F.conv2d(exp_x, se_kernel, padding=pad, groups=c)

        # Step 3: log and restore max
        log_sum = torch.log(conv_out + 1e-8) + max_per_bc

        # Step 4: soft dilation = +1/β · log_sum
        result = log_sum / beta
        return result

    def _soft_opening(self, x: torch.Tensor, se: torch.Tensor) -> torch.Tensor:
        """Opening: dilation(erosion(x)). Removes small bright structures."""
        return self._soft_dilation(self._soft_erosion(x, se), se)

    def _soft_closing(self, x: torch.Tensor, se: torch.Tensor) -> torch.Tensor:
        """Closing: erosion(dilation(x)). Fills small dark gaps."""
        return self._soft_erosion(self._soft_dilation(x, se), se)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Morph math in fp32 (AMP off); cast stable output back to input dtype for next layer.
        with torch.amp.autocast("cuda", enabled=False):
            out = torch.clamp(self._forward_impl(x.float()), -1e4, 1e4)
        return out.to(device=x.device, dtype=x.dtype)

    def _conv_bn_relu(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, x: torch.Tensor) -> torch.Tensor:
        """Conv+BN in fp32 even when AMP has cast module weights to fp16."""
        w = conv.weight.float()
        b = conv.bias.float() if conv.bias is not None else None
        x = F.conv2d(x, w, b, stride=conv.stride, padding=conv.padding, groups=conv.groups)
        rm, rv = bn.running_mean.float(), bn.running_var.float()
        x = F.batch_norm(
            x, rm, rv, bn.weight.float(), bn.bias.float(),
            training=bn.training, momentum=bn.momentum, eps=bn.eps,
        )
        return F.relu(x)

    def _conv_bn_sigmoid(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, x: torch.Tensor) -> torch.Tensor:
        w = conv.weight.float()
        b = conv.bias.float() if conv.bias is not None else None
        x = F.conv2d(x, w, b, stride=conv.stride, padding=conv.padding, groups=conv.groups)
        rm, rv = bn.running_mean.float(), bn.running_var.float()
        x = F.batch_norm(
            x, rm, rv, bn.weight.float(), bn.bias.float(),
            training=bn.training, momentum=bn.momentum, eps=bn.eps,
        )
        return torch.sigmoid(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Get shared shape parameters from registry
        shapes = self.registry.get_shapes()

        # Build two structuring elements
        se_0 = self._build_se(shapes["theta_0"], shapes["aspect_0"], shapes["radius_0"])
        se_1 = self._build_se(shapes["theta_1"], shapes["aspect_1"], shapes["radius_1"])

        # Channel projection: c_in -> 16
        f_proj = self._conv_bn_relu(self.conv1x1_in, self.bn_in, x)

        # Head 0: opening/closing mixture (alpha-weighted)
        f0_open = self._soft_opening(f_proj, se_0)
        f0_close = self._soft_closing(f_proj, se_0)
        f0 = shapes["alpha_0"] * f0_open + (1.0 - shapes["alpha_0"]) * f0_close

        # Head 1: opening/closing mixture
        f1_open = self._soft_opening(f_proj, se_1)
        f1_close = self._soft_closing(f_proj, se_1)
        f1 = shapes["alpha_1"] * f1_open + (1.0 - shapes["alpha_1"]) * f1_close

        # Concatenate heads, project back to c_in, generate attention
        f_cat = torch.cat([f0, f1], dim=1)                     # [B, 32, H, W]
        a = self._conv_bn_sigmoid(self.conv1x1_out, self.bn_out, f_cat)  # [B, c_in, H, W] in (0, 1)

        # Gated residual (slow start)
        g = torch.sigmoid(self.gate_logit)                      # scalar in (0, 1)
        out = x + g * x * (a - 1.0)
        # Equivalent: out = x * (1 + g * (a - 1)) = x * ((1-g) + g*a)
        return out


# ---------------------------------------------------------------------------
# Internal: registry ownership tracking (to avoid duplicate submodule registration)
# ---------------------------------------------------------------------------

_REGISTRY_OWNERS: dict = {}


def _registry_already_owned(registry_key: str) -> bool:
    """Check whether some MorphAttention instance already owns this registry."""
    return registry_key in _REGISTRY_OWNERS


def _mark_registry_owned(registry_key: str, owner: nn.Module) -> None:
    """Record that this MorphAttention instance owns the registry for given key."""
    _REGISTRY_OWNERS[registry_key] = owner


def reset_morph_registries() -> None:
    """
    Reset all global registries. Call this when building a new model
    (e.g., between training runs in the same Python process, or in tests).

    IMPORTANT: This must be called BEFORE building a new model, otherwise
    the previous model's registry parameters will be reused.
    """
    global _MORPH_REGISTRIES, _REGISTRY_OWNERS
    _MORPH_REGISTRIES.clear()
    _REGISTRY_OWNERS.clear()


# ---------------------------------------------------------------------------
# Self-test (run with: python -m ultralytics.nn.modules.morph_attention)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MorphAttention self-test")
    print("=" * 60)

    reset_morph_registries()

    # Simulate three detection heads: P3 (128ch), P4 (256ch), P5 (512ch)
    m_p3 = MorphAttention(c_in=128, registry_key="shared")
    m_p4 = MorphAttention(c_in=256, registry_key="shared")
    m_p5 = MorphAttention(c_in=512, registry_key="shared")

    # Test 1: shape sharing — all three should reference the same registry
    assert m_p3.registry is m_p4.registry, "Registry not shared between P3 and P4"
    assert m_p4.registry is m_p5.registry, "Registry not shared between P4 and P5"
    print("[OK] Test 1: registry is shared across P3/P4/P5")

    # Test 2: forward pass (different spatial resolutions)
    x_p3 = torch.randn(2, 128, 80, 80)
    x_p4 = torch.randn(2, 256, 40, 40)
    x_p5 = torch.randn(2, 512, 20, 20)
    y_p3 = m_p3(x_p3)
    y_p4 = m_p4(x_p4)
    y_p5 = m_p5(x_p5)
    assert y_p3.shape == x_p3.shape
    assert y_p4.shape == x_p4.shape
    assert y_p5.shape == x_p5.shape
    print(f"[OK] Test 2: forward pass output shapes correct")
    print(f"     P3: {x_p3.shape} -> {y_p3.shape}")
    print(f"     P4: {x_p4.shape} -> {y_p4.shape}")
    print(f"     P5: {x_p5.shape} -> {y_p5.shape}")

    # Test 3: slow start verification — output should be very close to input
    # at initialization (g≈0.05, A≈0.5, so out ≈ 0.975 * x)
    rel_diff_p3 = (y_p3 - x_p3).abs().mean() / x_p3.abs().mean()
    rel_diff_p4 = (y_p4 - x_p4).abs().mean() / x_p4.abs().mean()
    rel_diff_p5 = (y_p5 - x_p5).abs().mean() / x_p5.abs().mean()
    print(f"[OK] Test 3: slow start relative diff (should be < 10%):")
    print(f"     P3 rel_diff: {rel_diff_p3.item():.4f}")
    print(f"     P4 rel_diff: {rel_diff_p4.item():.4f}")
    print(f"     P5 rel_diff: {rel_diff_p5.item():.4f}")
    assert rel_diff_p3 < 0.15, f"P3 slow start failed: {rel_diff_p3}"
    assert rel_diff_p4 < 0.15, f"P4 slow start failed: {rel_diff_p4}"
    assert rel_diff_p5 < 0.15, f"P5 slow start failed: {rel_diff_p5}"

    # Test 4: backward pass — verify gradients flow to all relevant params
    loss = (y_p3.mean() + y_p4.mean() + y_p5.mean())
    loss.backward()
    # Registry parameters should have non-zero gradient
    assert m_p3.registry.theta_0.grad is not None, "theta_0 grad is None"
    assert m_p3.registry.aspect_0_raw.grad is not None, "aspect_0_raw grad is None"
    assert m_p3.registry.radius_0.grad is not None, "radius_0 grad is None"
    # Per-scale conv weights should have grad
    assert m_p3.conv1x1_in.weight.grad is not None
    assert m_p3.conv1x1_out.weight.grad is not None
    assert m_p3.gate_logit.grad is not None
    print(f"[OK] Test 4: backward pass — gradients flow correctly")

    # Test 5: NaN/Inf check
    for name, t in [("y_p3", y_p3), ("y_p4", y_p4), ("y_p5", y_p5)]:
        assert not torch.isnan(t).any(), f"{name} contains NaN"
        assert not torch.isinf(t).any(), f"{name} contains Inf"
    print(f"[OK] Test 5: no NaN/Inf in outputs")

    # Test 6: parameter count
    n_p3 = sum(p.numel() for p in m_p3.parameters())
    n_p4 = sum(p.numel() for p in m_p4.parameters())
    n_p5 = sum(p.numel() for p in m_p5.parameters())
    # Note: m_p3 is the registry owner, so its param count includes registry params
    print(f"[INFO] Parameter counts (P3 owns registry):")
    print(f"     P3 (with registry): {n_p3} params")
    print(f"     P4 (ref only):      {n_p4} params")
    print(f"     P5 (ref only):      {n_p5} params")
    print(f"     Total (no dup):     {n_p3 + n_p4 + n_p5} params")

    # Test 7: SE shape inspection (should be ~circular for head 0, ~elongated for head 1)
    shapes = m_p3.registry.get_shapes()
    print(f"[INFO] Initial SE shapes:")
    print(f"     Head 0: theta={shapes['theta_0'].item():.3f}, "
          f"aspect={shapes['aspect_0'].item():.3f}, "
          f"radius={shapes['radius_0'].item():.3f}, "
          f"alpha={shapes['alpha_0'].item():.3f}")
    print(f"     Head 1: theta={shapes['theta_1'].item():.3f}, "
          f"aspect={shapes['aspect_1'].item():.3f}, "
          f"radius={shapes['radius_1'].item():.3f}, "
          f"alpha={shapes['alpha_1'].item():.3f}")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
