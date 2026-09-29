"""A set of 3D Gaussians with spherical-harmonic colour.

Parameters are stored in their unconstrained form:
  means (N,3) | quats (N,4) wxyz | log_scales (N,3) | opacity logits (N,) | sh0 (N,1,3) | shN (N,K-1,3)
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from nr.utils.sh import num_sh_bases, rgb_to_sh

PARAM_NAMES = ("means", "quats", "scales", "opacities", "sh0", "shN")


def knn_mean_dist(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Mean distance to the k nearest neighbours (excluding self)."""
    n = points.shape[0]
    if n <= 1:
        return torch.full((n,), 0.1)
    k = min(k, n - 1)
    try:
        from scipy.spatial import cKDTree

        pts = points.detach().cpu().double().numpy()
        dist, _ = cKDTree(pts).query(pts, k=k + 1)
        return torch.from_numpy(dist[:, 1:].mean(1)).float()
    except ImportError:  # pragma: no cover - chunked brute force fallback
        out = []
        for s in range(0, n, 4096):
            d = torch.cdist(points[s : s + 4096], points)
            out.append(d.topk(k + 1, largest=False).values[:, 1:].mean(1))
        return torch.cat(out)


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


class GaussianModel(nn.Module):
    def __init__(self, num: int, sh_degree: int = 3):
        super().__init__()
        self.sh_degree = sh_degree
        k = num_sh_bases(sh_degree)
        self.means = nn.Parameter(torch.zeros(num, 3))
        self.quats = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(num, 1))
        self.scales = nn.Parameter(torch.zeros(num, 3))
        self.opacities = nn.Parameter(torch.zeros(num))
        self.sh0 = nn.Parameter(torch.zeros(num, 1, 3))
        self.shN = nn.Parameter(torch.zeros(num, k - 1, 3))

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_points(
        cls,
        points: torch.Tensor,
        colors: torch.Tensor | None = None,
        sh_degree: int = 3,
        init_opacity: float = 0.1,
        knn: int = 3,
        max_init_scale: float | None = None,
    ) -> GaussianModel:
        points = points.float()
        n = points.shape[0]
        model = cls(n, sh_degree)
        if colors is None:
            colors = torch.full((n, 3), 0.5)
        dist = knn_mean_dist(points, knn).clamp(min=1e-4)
        if max_init_scale is not None:
            dist = dist.clamp(max=max_init_scale)
        with torch.no_grad():
            model.means.copy_(points)
            model.scales.copy_(torch.log(dist)[:, None].repeat(1, 3))
            model.opacities.fill_(float(inverse_sigmoid(torch.tensor(init_opacity))))
            model.sh0.copy_(rgb_to_sh(colors.float().clamp(0, 1))[:, None, :])
        return model

    @classmethod
    def empty_like_state(cls, state: dict, prefix: str = "") -> GaussianModel:
        n = state[prefix + "means"].shape[0]
        k = state[prefix + "shN"].shape[1] + 1
        return cls(n, int(round(math.sqrt(k))) - 1)

    # ------------------------------------------------------------------ activations
    def __len__(self) -> int:
        return self.means.shape[0]

    @property
    def num(self) -> int:
        return self.means.shape[0]

    def activated(self) -> dict[str, torch.Tensor]:
        return {
            "means": self.means,
            "quats": self.quats,
            "scales": torch.exp(self.scales),
            "opacities": torch.sigmoid(self.opacities),
            "sh": torch.cat([self.sh0, self.shN], dim=1),
        }

    def params(self) -> dict[str, nn.Parameter]:
        return {name: getattr(self, name) for name in PARAM_NAMES}

    # ------------------------------------------------------------------ PLY I/O (standard 3DGS layout)
    def save_ply(self, path: str, means=None, quats=None) -> None:
        """Write in the layout used by the original 3DGS code so external viewers can read it.

        ``means`` / ``quats`` may override the stored values (e.g. world-space object Gaussians).
        """
        save_ply(
            path,
            means if means is not None else self.means,
            quats if quats is not None else self.quats,
            self.scales,
            self.opacities,
            self.sh0,
            self.shN,
        )

    @classmethod
    def load_ply(cls, path: str) -> GaussianModel:
        with open(path, "rb") as f:
            header = []
            while True:
                line = f.readline().decode().strip()
                header.append(line)
                if line == "end_header":
                    break
            names = [ln.split()[-1] for ln in header if ln.startswith("property")]
            n = int(next(ln for ln in header if ln.startswith("element vertex")).split()[-1])
            data = np.frombuffer(f.read(), dtype=np.float32).reshape(n, len(names))
        col = {name: i for i, name in enumerate(names)}
        n_rest = sum(1 for nm in names if nm.startswith("f_rest_"))
        k = n_rest // 3 + 1
        model = cls(n, int(round(math.sqrt(k))) - 1)

        def g(prefix, count):
            return torch.from_numpy(np.stack([data[:, col[f"{prefix}{i}"]] for i in range(count)], 1).copy())

        with torch.no_grad():
            model.means.copy_(torch.from_numpy(data[:, [col["x"], col["y"], col["z"]]].copy()))
            model.sh0.copy_(g("f_dc_", 3)[:, None, :])
            if k > 1:
                model.shN.copy_(g("f_rest_", n_rest).reshape(n, 3, k - 1).transpose(1, 2))
            model.opacities.copy_(torch.from_numpy(data[:, col["opacity"]].copy()))
            model.scales.copy_(g("scale_", 3))
            model.quats.copy_(g("rot_", 4))
        return model


def save_ply(path, means, quats, log_scales, opacity_logits, sh0, shN) -> None:
    n = means.shape[0]
    k_rest = shN.shape[1]
    cols = [
        means.detach().cpu(),
        torch.zeros(n, 3),
        sh0.detach().cpu().reshape(n, 3),
        shN.detach().cpu().transpose(1, 2).reshape(n, 3 * k_rest),  # channel-major as in 3DGS
        opacity_logits.detach().cpu().reshape(n, 1),
        log_scales.detach().cpu(),
        quats.detach().cpu(),
    ]
    data = torch.cat(cols, dim=1).float().numpy()
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * k_rest)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
    header += "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.astype("<f4").tobytes())
