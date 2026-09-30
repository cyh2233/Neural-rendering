"""Pure-PyTorch reference rasterizer for 3D Gaussian Splatting.

Readable, differentiable, device-agnostic and *slow*. It mirrors the math of the
original 3DGS / gsplat "classic" mode so it can stand in for gsplat on CPU:

1. transform means into the camera frame and project them (pinhole);
2. EWA splatting: Sigma_2D = J W Sigma_3D W^T J^T + eps * I;
3. sort by depth and alpha-composite front to back, per 16x16 pixel tile, using only the
   Gaussians whose 3-sigma footprint overlaps the tile (like gsplat).

``composite_dense`` evaluates every Gaussian at every pixel and is kept as a reference.
"""

from __future__ import annotations

import math

import torch

from nr.utils.geometry import quat_to_rotmat

ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99


def project_gaussians(means, quats, scales, viewmat, K, width, height, near=0.01, far=1e10, eps2d=0.3):
    """Project 3D Gaussians to 2D.

    Returns means2d (N, 2), conics (N, 3) [a, b, c] of the inverse 2D covariance,
    depths (N,), radii (N,) integer pixel radius (0 = culled).
    """
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    pc = means @ R.T + t
    z = pc[:, 2]
    zc = z.clamp(min=near)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    rot = quat_to_rotmat(quats)
    M = rot * scales[:, None, :]
    cov3d = M @ M.transpose(1, 2)
    cov_c = R @ cov3d @ R.T

    # Clamp the point used for the Jacobian to slightly outside the frustum (as in 3DGS).
    lim_x = 1.3 * 0.5 * width / fx
    lim_y = 1.3 * 0.5 * height / fy
    tx = torch.minimum(torch.maximum(pc[:, 0] / zc, -lim_x), lim_x) * zc
    ty = torch.minimum(torch.maximum(pc[:, 1] / zc, -lim_y), lim_y) * zc
    zero = torch.zeros_like(zc)
    J = torch.stack(
        [
            torch.stack([fx / zc, zero, -fx * tx / zc**2], -1),
            torch.stack([zero, fy / zc, -fy * ty / zc**2], -1),
        ],
        dim=1,
    )  # (N, 2, 3)
    cov2d = J @ cov_c @ J.transpose(1, 2)
    a = cov2d[:, 0, 0] + eps2d
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + eps2d
    det = a * c - b * b
    det_safe = torch.where(det > 0, det, torch.ones_like(det))
    conics = torch.stack([c / det_safe, -b / det_safe, a / det_safe], -1)

    means2d = torch.stack([fx * pc[:, 0] / zc + cx, fy * pc[:, 1] / zc + cy], -1)

    with torch.no_grad():
        mid = 0.5 * (a + c)
        lam = mid + torch.sqrt((mid * mid - det).clamp(min=0.1))
        radius = torch.ceil(3.0 * torch.sqrt(lam))
        u, v = means2d[:, 0], means2d[:, 1]
        on_screen = (u + radius > 0) & (u - radius < width) & (v + radius > 0) & (v - radius < height)
        visible = (z > near) & (z < far) & (det > 0) & on_screen
        radii = torch.where(visible, radius, torch.zeros_like(radius)).long()
    return means2d, conics, z, radii


def _empty(height, width, n_ch, device, dtype):
    return (
        torch.zeros(height, width, n_ch, device=device, dtype=dtype),
        torch.zeros(height, width, 1, device=device, dtype=dtype),
        torch.zeros(height, width, 1, device=device, dtype=dtype),
    )


def _blend(p, m2d, con, opa, feats):
    """Alpha-composite depth-sorted Gaussians over pixel centres ``p`` (P, 2).
    Returns (P, C+1) accumulated features and (P,) accumulated alpha."""
    d = p[None, :, :] - m2d[:, None, :]  # (G, P, 2)
    dx, dy = d[..., 0], d[..., 1]
    power = -0.5 * (con[:, 0:1] * dx * dx + con[:, 2:3] * dy * dy) - con[:, 1:2] * dx * dy
    alpha = opa[:, None] * torch.exp(power.clamp(max=0.0))
    alpha = alpha.clamp(max=ALPHA_MAX)
    alpha = torch.where(alpha < ALPHA_MIN, torch.zeros_like(alpha), alpha)
    trans = torch.cumprod(1.0 - alpha, dim=0)
    trans = torch.cat([torch.ones_like(trans[:1]), trans[:-1]], dim=0)
    w = alpha * trans  # (G, P)
    return w.T @ feats, w.sum(0)


def _sorted_visible(means2d, conics, depths, opacities, colors, radii):
    vis = (radii > 0).nonzero().squeeze(1)
    if vis.numel() == 0:
        return None
    idx = vis[torch.argsort(depths[vis])]
    feats = torch.cat([colors[idx], depths[idx, None]], dim=-1)  # (G, C+1)
    return idx, means2d[idx], conics[idx], opacities[idx], feats


def composite(means2d, conics, depths, opacities, colors, radii, width, height, tile: int = 16):
    """Tile-based front-to-back alpha compositing (the scheme used by gsplat / 3DGS).

    The image is cut into ``tile`` x ``tile`` blocks. Each block only blends the Gaussians whose
    3-sigma bounding box overlaps it, so cost scales with the Gaussians' screen footprint instead
    of N_gaussians * N_pixels. Returns premultiplied rgb (H,W,C), alpha (H,W,1), depth (H,W,1).
    """
    device, dtype = means2d.device, means2d.dtype
    n_ch = colors.shape[-1]
    sv = _sorted_visible(means2d, conics, depths, opacities, colors, radii)
    if sv is None:
        return _empty(height, width, n_ch, device, dtype)
    idx, m2d, con, opa, feats = sv
    r = radii[idx].to(dtype)
    u, v = m2d[:, 0].detach(), m2d[:, 1].detach()
    ntx, nty = (width + tile - 1) // tile, (height + tile - 1) // tile
    x0 = torch.arange(ntx, device=device, dtype=dtype) * tile
    y0 = torch.arange(nty, device=device, dtype=dtype) * tile
    hit_x = (u[:, None] + r[:, None] >= x0) & (u[:, None] - r[:, None] < x0 + tile)  # (G, ntx)
    hit_y = (v[:, None] + r[:, None] >= y0) & (v[:, None] - r[:, None] < y0 + tile)  # (G, nty)

    out_feat, out_alpha, out_pix = [], [], []
    for ty in range(nty):
        row = hit_y[:, ty].nonzero().squeeze(1)
        if row.numel() == 0:
            continue
        ys = torch.arange(ty * tile, min((ty + 1) * tile, height), device=device)
        hx = hit_x[row]
        for tx in range(ntx):
            sel = row[hx[:, tx]]  # still depth-sorted: nonzero() keeps order
            if sel.numel() == 0:
                continue
            xs = torch.arange(tx * tile, min((tx + 1) * tile, width), device=device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            p = torch.stack([gx.reshape(-1).to(dtype) + 0.5, gy.reshape(-1).to(dtype) + 0.5], -1)
            f, a = _blend(p, m2d[sel], con[sel], opa[sel], feats[sel])
            out_feat.append(f)
            out_alpha.append(a)
            out_pix.append((gy * width + gx).reshape(-1))
    if not out_pix:
        return _empty(height, width, n_ch, device, dtype)
    pix = torch.cat(out_pix)
    feat = torch.zeros(height * width, n_ch + 1, device=device, dtype=dtype).index_copy(0, pix, torch.cat(out_feat))
    acc = torch.zeros(height * width, device=device, dtype=dtype).index_copy(0, pix, torch.cat(out_alpha))
    feat = feat.reshape(height, width, n_ch + 1)
    acc = acc.reshape(height, width, 1)
    depth = feat[..., n_ch:] / acc.clamp(min=1e-10)  # expected depth, like gsplat "ED"
    return feat[..., :n_ch], acc, depth


def composite_dense(means2d, conics, depths, opacities, colors, radii, width, height, max_elems=1 << 22):
    """Reference compositing without tiling: every visible Gaussian against every pixel.
    Slow; kept to validate :func:`composite`. Returns the same outputs."""
    device, dtype = means2d.device, means2d.dtype
    n_ch = colors.shape[-1]
    sv = _sorted_visible(means2d, conics, depths, opacities, colors, radii)
    if sv is None:
        return _empty(height, width, n_ch, device, dtype)
    _, m2d, con, opa, feats = sv
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pix = torch.stack([xs + 0.5, ys + 0.5], -1).reshape(-1, 2)  # pixel centres
    chunk = max(1, max_elems // m2d.shape[0])
    out_feat, out_alpha = [], []
    for s in range(0, pix.shape[0], chunk):
        f, a = _blend(pix[s : s + chunk], m2d, con, opa, feats)
        out_feat.append(f)
        out_alpha.append(a)
    feat = torch.cat(out_feat, 0).reshape(height, width, n_ch + 1)
    acc = torch.cat(out_alpha, 0).reshape(height, width, 1)
    depth = feat[..., n_ch:] / acc.clamp(min=1e-10)
    return feat[..., :n_ch], acc, depth


def rasterize_torch(
    means, quats, scales, opacities, colors, viewmat, K, width, height, near=0.01, far=1e10, dense=False
):
    means2d, conics, depths, radii = project_gaussians(means, quats, scales, viewmat, K, width, height, near, far)
    if means2d.requires_grad:
        means2d.retain_grad()
    fn = composite_dense if dense else composite
    rgb, alpha, depth = fn(means2d, conics, depths, opacities, colors, radii, width, height)
    info = {"means2d": means2d, "radii": radii, "depths": depths, "width": width, "height": height}
    return rgb, alpha, depth, info


def gaussian_2d_value(sigma_px: float, dx: float, dy: float) -> float:
    """Helper for tests: isotropic 2D Gaussian falloff exp(-r^2 / (2 sigma^2))."""
    return math.exp(-0.5 * (dx * dx + dy * dy) / (sigma_px * sigma_px))
