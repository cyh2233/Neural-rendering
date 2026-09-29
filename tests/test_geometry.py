import math

import torch

from nr.utils.config import load_config
from nr.utils.geometry import (
    invert_transform,
    lateral_shift,
    make_transform,
    normalize_quat,
    project_points,
    quat_multiply,
    quat_to_rotmat,
    rotmat_to_quat,
    yaw_to_quat,
)
from nr.utils.sh import eval_sh, rgb_to_sh, sh_to_rgb


def test_quat_roundtrip_and_orthonormal():
    q = normalize_quat(torch.randn(100, 4, dtype=torch.float64))
    q = torch.where(q[:, :1] < 0, -q, q)
    r = quat_to_rotmat(q)
    eye = torch.eye(3, dtype=torch.float64).expand(100, 3, 3)
    assert torch.allclose(r @ r.transpose(1, 2), eye, atol=1e-10)
    assert torch.allclose(torch.linalg.det(r), torch.ones(100, dtype=torch.float64))
    assert torch.allclose(rotmat_to_quat(r), q, atol=1e-10)


def test_quat_multiply_matches_matrix_product():
    a = normalize_quat(torch.randn(10, 4, dtype=torch.float64))
    b = normalize_quat(torch.randn(10, 4, dtype=torch.float64))
    assert torch.allclose(quat_to_rotmat(quat_multiply(a, b)), quat_to_rotmat(a) @ quat_to_rotmat(b), atol=1e-10)


def test_yaw_quat():
    r = quat_to_rotmat(yaw_to_quat(torch.tensor(math.pi / 2)))
    assert torch.allclose(r @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1, 0]), atol=1e-6)


def test_invert_and_project():
    r = quat_to_rotmat(normalize_quat(torch.randn(4)))
    tf = make_transform(r, torch.randn(3))
    assert torch.allclose(invert_transform(tf) @ tf, torch.eye(4), atol=1e-5)
    K = torch.tensor([[100.0, 0, 50], [0, 100, 40], [0, 0, 1]])
    uv, z = project_points(torch.tensor([[0.0, 0.0, 2.0], [1.0, -1.0, 2.0]]), torch.eye(4), K)
    assert torch.allclose(uv, torch.tensor([[50.0, 40.0], [100.0, -10.0]]))
    assert torch.allclose(z, torch.tensor([2.0, 2.0]))


def test_lateral_shift_moves_along_camera_x():
    c2w = torch.eye(4)
    out = lateral_shift(c2w, 1.5, up=0.5)
    assert torch.allclose(out[:3, 3], torch.tensor([1.5, -0.5, 0.0]))


def test_sh_dc_roundtrip():
    rgb = torch.rand(20, 3)
    sh = torch.zeros(20, 16, 3)
    sh[:, 0] = rgb_to_sh(rgb)
    out = eval_sh(3, sh, torch.nn.functional.normalize(torch.randn(20, 3), dim=-1)) + 0.5
    assert torch.allclose(out, rgb, atol=1e-6)
    assert torch.allclose(sh_to_rgb(sh[:, 0]), rgb, atol=1e-6)


def test_config_inheritance_and_overrides():
    cfg = load_config("configs/synthetic.yaml", ["train.max_steps=7", "model.sh_degree=2", "new.key=[1, 2]"])
    assert cfg.train.max_steps == 7
    assert cfg.model.sh_degree == 2
    assert cfg.new.key == [1, 2]
    assert cfg.train.lr.scales == 5.0e-3  # inherited from base.yaml
    assert cfg.train.lr.means == 1.0e-3   # overridden in synthetic.yaml
