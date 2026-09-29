"""Image quality metrics. Images are (H, W, 3) float tensors in [0, 1]."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = ((pred.clamp(0, 1) - gt) ** 2).mean()
    return -10.0 * torch.log10(mse.clamp(min=1e-10))


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    return w.expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, gt: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Standard SSIM (Wang et al. 2004) with a Gaussian window, averaged over the image."""
    x = pred.permute(2, 0, 1)[None]
    y = gt.permute(2, 0, 1)[None]
    c = x.shape[1]
    w = _gaussian_window(window, sigma, c, x.device, x.dtype)
    pad = window // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    sxx = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x**2
    syy = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y**2
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return s.mean()


class LPIPS:
    """Lazy wrapper around the ``lpips`` package (optional dependency)."""

    def __init__(self, net: str = "alex"):
        self._net = net
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import lpips  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            import lpips

            self._model = lpips.LPIPS(net=self._net, verbose=False).to(pred.device).eval()
        x = pred.clamp(0, 1).permute(2, 0, 1)[None] * 2 - 1
        y = gt.permute(2, 0, 1)[None] * 2 - 1
        with torch.no_grad():
            return self._model(x, y).mean()
