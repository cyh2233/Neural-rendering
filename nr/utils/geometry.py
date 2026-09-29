"""Rigid-body and camera geometry helpers.

Conventions used throughout the project:
  * Quaternions are (w, x, y, z), Hamilton convention.
  * Cameras follow OpenCV: +x right, +y down, +z forward.
  * ``c2w`` is a 4x4 camera-to-world matrix, ``viewmat`` its inverse (world-to-camera).
All functions work on both numpy arrays and torch tensors where noted.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------- torch


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return F.normalize(q, dim=-1)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz quaternion -> (..., 3, 3) rotation matrix."""
    q = normalize_quat(q)
    w, x, y, z = q.unbind(-1)
    r = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return r.reshape(q.shape[:-1] + (3, 3))


def rotmat_to_quat(r: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix -> (..., 4) wxyz quaternion (w >= 0)."""
    m = r.reshape(-1, 3, 3)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cands = torch.stack(
        [
            torch.stack([1 + trace, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], -1),
            torch.stack([m[:, 2, 1] - m[:, 1, 2], 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0]], -1),
            torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 0, 1] + m[:, 1, 0], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], m[:, 1, 2] + m[:, 2, 1]], -1),
            torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1], 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], -1),
        ],
        dim=1,
    )  # (N, 4 candidates, 4)
    diag = torch.stack([1 + trace, 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
                        1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], -1)
    best = diag.argmax(-1)
    q = cands[torch.arange(m.shape[0]), best]
    q = F.normalize(q, dim=-1)
    q = torch.where(q[:, :1] < 0, -q, q)
    return q.reshape(r.shape[:-2] + (4,))


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b for wxyz quaternions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Rotation about +z by ``yaw`` radians -> wxyz quaternion."""
    half = yaw * 0.5
    zeros = torch.zeros_like(yaw)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def make_transform(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build (..., 4, 4) homogeneous transforms from rotation and translation."""
    out = torch.zeros(r.shape[:-2] + (4, 4), dtype=r.dtype, device=r.device)
    out[..., :3, :3] = r
    out[..., :3, 3] = t
    out[..., 3, 3] = 1.0
    return out


def invert_transform(tf: torch.Tensor) -> torch.Tensor:
    r = tf[..., :3, :3]
    t = tf[..., :3, 3]
    rt = r.transpose(-1, -2)
    return make_transform(rt, -(rt @ t.unsqueeze(-1)).squeeze(-1))


def transform_points(tf: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply a single (4, 4) transform to (N, 3) points."""
    return pts @ tf[:3, :3].T + tf[:3, 3]


def project_points(pts_world: torch.Tensor, viewmat: torch.Tensor, K: torch.Tensor):
    """Project (N, 3) world points. Returns (uv (N, 2), depth (N,))."""
    pc = transform_points(viewmat, pts_world)
    z = pc[:, 2]
    uv = (pc[:, :2] / z.clamp(min=1e-6).unsqueeze(-1)) @ K[:2, :2].T + K[:2, 2]
    return uv, z


# ----------------------------------------------------------------------------- numpy


def np_quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    return quat_to_rotmat(torch.as_tensor(q, dtype=torch.float64)).numpy()


def np_make_transform(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def np_transform_points(tf: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ tf[:3, :3].T + tf[:3, 3]


def lateral_shift(c2w: torch.Tensor, offset: float, up: float = 0.0) -> torch.Tensor:
    """Shift an OpenCV camera sideways (+x = right) and vertically (+up = up = -y)."""
    out = c2w.clone()
    out[:3, 3] = c2w[:3, 3] + c2w[:3, 0] * offset - c2w[:3, 1] * up
    return out
