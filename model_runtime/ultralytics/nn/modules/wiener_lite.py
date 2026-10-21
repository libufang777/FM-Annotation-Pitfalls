"""
WienerLite: Learnable frequency-domain Wiener deblurring on a single feature map.
Phase 6 of YOLO-MSHF — inserted before Detect (parallel to Phase 5 MorphAttention placement).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WienerLite(nn.Module):
    """
    Single-scale Wiener deblurring with slow-start gated residual.

    Physics: learnable isotropic Gaussian blur kernel G (7x7, sigma learnable) and
    learnable 1/SNR (large initial => weak deblurring). Frequency domain:
        Ŝ = conj(G) / (|G|² + 1/SNR) · Y  →  IFFT → real part.

    Output: out = x + sigmoid(gate_logit) * (deblurred - x), gate_logit init -3 (~0.05).

    Args:
        c_in: input/output channel count (unchanged).
        kernel_size: blur kernel spatial size (default 7).
    """

    def __init__(self, c_in: int, kernel_size: int = 7):
        super().__init__()
        self.c_in = c_in
        self.kernel_size = kernel_size

        # Learnable Gaussian sigma (softplus keeps positive)
        self.log_sigma = nn.Parameter(torch.tensor(0.5))

        # 1/SNR: large => weak deblur (denominator dominated by regularization)
        self.inv_snr_log = nn.Parameter(torch.tensor(math.log(math.exp(10.0) - 1.0)))

        # Slow start: sigmoid(-3) ≈ 0.047
        self.gate_logit = nn.Parameter(torch.tensor(-3.0))

        half = kernel_size // 2
        coords = torch.arange(-half, half + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("grid_xx", xx.clone(), persistent=False)
        self.register_buffer("grid_yy", yy.clone(), persistent=False)

    def _gaussian_kernel(self) -> torch.Tensor:
        """Normalized 2D isotropic Gaussian, shape [k, k]."""
        sigma = F.softplus(self.log_sigma) + 0.1
        g = torch.exp(-(self.grid_xx.float() ** 2 + self.grid_yy.float() ** 2) / (2.0 * sigma ** 2))
        return g / (g.sum() + 1e-8)

    def _pad_kernel_fft(self, g: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Embed k×k kernel at center of H×W grid and return fft2(G)."""
        kh, kw = g.shape
        g_pad = torch.zeros(h, w, device=g.device, dtype=g.dtype)
        h0, w0 = h // 2 - kh // 2, w // 2 - kw // 2
        g_pad[h0 : h0 + kh, w0 : w0 + kw] = g
        return torch.fft.fft2(g_pad)

    def _wiener_deblur(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] fp32 → deblurred same shape."""
        b, c, h, w = x.shape
        g = self._gaussian_kernel()
        inv_snr = F.softplus(self.inv_snr_log) + 1e-3

        g_fft = self._pad_kernel_fft(g, h, w)  # [H, W] complex
        denom = g_fft.abs().square() + inv_snr
        wiener = torch.conj(g_fft) / denom

        y_fft = torch.fft.fft2(x, dim=(-2, -1))
        s_hat = wiener * y_fft
        return torch.fft.ifft2(s_hat, dim=(-2, -1)).real

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFT / division must run in fp32 under AMP (same pattern as MorphAttention).
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()
            deblurred = torch.clamp(self._wiener_deblur(xf), -1e4, 1e4)
            g = torch.sigmoid(self.gate_logit.float())
            out = xf + g * (deblurred - xf)
        return out.to(device=x.device, dtype=x.dtype)


if __name__ == "__main__":
    print("=" * 60)
    print("WienerLite self-test")
    print("=" * 60)

    scales = [
        ("P2", 64, 160, 160),
        ("P3", 64, 80, 80),
        ("P4", 128, 40, 40),
        ("P5", 256, 20, 20),
    ]
    modules = []
    for name, ch, hh, ww in scales:
        mod = WienerLite(c_in=ch)
        modules.append((name, mod, ch, hh, ww))

    # Forward + slow start
    for name, mod, ch, hh, ww in modules:
        x = torch.randn(2, ch, hh, ww)
        y = mod(x)
        assert y.shape == x.shape, f"{name} shape mismatch"
        rel = (y - x).abs().mean() / (x.abs().mean() + 1e-8)
        print(f"[OK] {name} forward {tuple(x.shape)} rel_diff={rel.item():.4f}")
        assert rel < 0.15, f"{name} slow start failed: rel_diff={rel}"
        assert not torch.isnan(y).any() and not torch.isinf(y).any(), f"{name} nan/inf"

    # Backward
    loss = sum(mod(torch.randn(1, ch, hh, ww, requires_grad=True)).mean()
               for _, mod, ch, hh, ww in modules)
    loss.backward()
    for name, mod, *_ in modules:
        assert mod.log_sigma.grad is not None, f"{name} log_sigma grad"
        assert mod.inv_snr_log.grad is not None, f"{name} inv_snr_log grad"
        assert mod.gate_logit.grad is not None, f"{name} gate grad"
    print("[OK] backward — gradients on learnable params")

    # AMP fp16 input smoke (module should return fp16, no nan)
    if torch.cuda.is_available():
        mod = WienerLite(64).cuda()
        x = torch.randn(1, 64, 80, 80, device="cuda", dtype=torch.float16)
        with torch.cuda.amp.autocast():
            y = mod(x)
        assert y.dtype == torch.float16
        assert not y.isnan().any()
        print("[OK] AMP fp16 input → fp16 output, no nan")
    else:
        print("[SKIP] CUDA not available for AMP test")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
