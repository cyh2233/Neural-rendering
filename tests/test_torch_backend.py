import math

import pytest
import torch

from nr.render.backend import rasterize
from nr.render.gsplat_backend import gsplat_available
from nr.render.torch_backend import project_gaussians, rasterize_torch

K = torch.tensor([[50.0, 0, 16], [0, 50, 16], [0, 0, 1]])


def _one(z=5.0, s=0.2, o=0.8, color=(1.0, 0.5, 0.25)):
    return (
        torch.tensor([[0.0, 0.0, z]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.full((1, 3), s),
        torch.tensor([o]),
        torch.tensor([color]),
    )


def test_single_gaussian_matches_analytic():
    s, o, z = 0.2, 0.8, 5.0
    means, quats, scales, opac, cols = _one(z, s, o)
    rgb, alpha, depth, info = rasterize_torch(means, quats, scales, opac, cols, torch.eye(4), K, 32, 32)
    sig2 = (50 * s / z) ** 2 + 0.3  # EWA footprint + 0.3 px low-pass
    for i, j in [(15, 15), (15, 18), (20, 16), (10, 12)]:
        dx, dy = j + 0.5 - 16, i + 0.5 - 16
        expected = o * math.exp(-0.5 * (dx * dx + dy * dy) / sig2)
        if expected < 1 / 255:
            expected = 0.0
        assert alpha[i, j, 0].item() == pytest.approx(expected, abs=1e-5)
        assert rgb[i, j, 1].item() == pytest.approx(0.5 * expected, abs=1e-5)
    assert depth[15, 15, 0].item() == pytest.approx(z, abs=1e-4)
    assert info["radii"][0] > 0


def test_front_gaussian_occludes_back():
    means = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 6.0]])
    quats = torch.tensor([[1.0, 0, 0, 0]] * 2)
    scales = torch.full((2, 3), 0.5)
    opac = torch.tensor([0.99, 0.99])
    cols = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Order of inputs must not matter: the rasterizer sorts by depth.
    for perm in ([0, 1], [1, 0]):
        rgb, alpha, depth, _ = rasterize_torch(means[perm], quats, scales, opac, cols[perm], torch.eye(4), K, 32, 32)
        assert rgb[16, 16, 0] > 0.95 and rgb[16, 16, 1] < 0.05
        assert depth[16, 16, 0].item() == pytest.approx(3.0, abs=0.1)


def test_behind_camera_is_culled():
    means, quats, scales, opac, cols = _one(z=-5.0)
    rgb, alpha, _, info = rasterize_torch(means, quats, scales, opac, cols, torch.eye(4), K, 32, 32)
    assert alpha.abs().max() == 0 and info["radii"][0] == 0


def test_anisotropic_projection_orientation():
    # Gaussian elongated along camera x must produce a wider footprint in u than v.
    means, quats, _, _, _ = _one()
    scales = torch.tensor([[0.5, 0.05, 0.05]])
    _, conics, _, _ = project_gaussians(means, quats, scales, torch.eye(4), K, 32, 32)
    a, b, c = conics[0]
    assert a < c and abs(b) < 1e-6


def test_gradients_flow_to_all_parameters():
    torch.manual_seed(0)
    n = 6
    means = (torch.randn(n, 3) * 0.3 + torch.tensor([0, 0, 4.0])).requires_grad_()
    quats = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1).requires_grad_()
    scales = torch.full((n, 3), 0.2).requires_grad_()
    opac = torch.full((n,), 0.5).requires_grad_()
    cols = torch.rand(n, 3).requires_grad_()
    rgb, alpha, depth, info = rasterize_torch(means, quats, scales, opac, cols, torch.eye(4), K, 32, 32)
    (rgb.sum() + depth.sum() * 0.01).backward()
    for t in (means, quats, scales, opac, cols, info["means2d"]):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0


def test_gradcheck_small():
    torch.manual_seed(1)
    n = 3
    dt = torch.float64
    means = (torch.randn(n, 3, dtype=dt) * 0.2 + torch.tensor([0, 0, 4.0], dtype=dt)).requires_grad_()
    quats = torch.nn.functional.normalize(torch.randn(n, 4, dtype=dt), dim=-1).requires_grad_()
    scales = torch.full((n, 3), 0.4, dtype=dt).requires_grad_()
    opac = torch.tensor([0.3, 0.5, 0.6], dtype=dt).requires_grad_()
    cols = torch.rand(n, 3, dtype=dt).requires_grad_()
    Kd = K.to(dt) * torch.tensor([[0.25], [0.25], [1.0]], dtype=dt)  # 8x8 image

    def f(m, q, s, o, c):
        rgb, _, _, _ = rasterize_torch(m, q, s, o, c, torch.eye(4, dtype=dt), Kd, 8, 8)
        return rgb

    assert torch.autograd.gradcheck(f, (means, quats, scales, opac, cols), eps=1e-6, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not gsplat_available(), reason="needs CUDA + gsplat")
def test_gsplat_matches_torch_backend():
    torch.manual_seed(0)
    n = 200
    dev = "cuda"
    means = torch.randn(n, 3, device=dev) + torch.tensor([0, 0, 6.0], device=dev)
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1)
    scales = torch.rand(n, 3, device=dev) * 0.2 + 0.05
    opac = torch.rand(n, device=dev) * 0.8 + 0.1
    cols = torch.rand(n, 3, device=dev)
    Kc = torch.tensor([[60.0, 0, 32], [0, 60, 24], [0, 0, 1]], device=dev)
    view = torch.eye(4, device=dev)
    a = rasterize(means, quats, scales, opac, cols, view, Kc, 64, 48, backend="torch")
    b = rasterize(means, quats, scales, opac, cols, view, Kc, 64, 48, backend="gsplat")
    assert (a[0] - b[0]).abs().mean() < 1e-2
    assert (a[1] - b[1]).abs().mean() < 1e-2
