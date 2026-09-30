"""Scene-graph nodes: static background, rigid dynamic objects, sky."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nr.scene.gaussians import GaussianModel
from nr.utils.geometry import quat_multiply, rotmat_to_quat, yaw_to_quat


class BackgroundNode(nn.Module):
    """Static Gaussians living directly in world coordinates."""

    def __init__(self, gaussians: GaussianModel):
        super().__init__()
        self.gaussians = gaussians

    def world_gaussians(self, frame_idx: int) -> dict[str, torch.Tensor]:
        return self.gaussians.activated()


class RigidObjectNode(nn.Module):
    """Gaussians defined in a canonical box frame and moved by per-frame box poses.

    world_pose(t) = base_pose(t) @ [Rz(delta_yaw_t) | delta_t_t]   (residual in the box frame)
    """

    def __init__(
        self,
        gaussians: GaussianModel,
        base_poses: torch.Tensor,   # (T, 4, 4)
        valid: torch.Tensor,        # (T,) bool
        size: torch.Tensor,         # (3,)
        instance_id: str = "",
        category: str = "",
        pose_refine: bool = True,
    ):
        super().__init__()
        self.gaussians = gaussians
        self.register_buffer("base_poses", base_poses.float().clone())
        self.register_buffer("valid", valid.bool().clone())
        self.register_buffer("size", size.float().clone())
        self.instance_id = instance_id
        self.category = category
        t = base_poses.shape[0]
        self.delta_trans = nn.Parameter(torch.zeros(t, 3), requires_grad=pose_refine)
        self.delta_yaw = nn.Parameter(torch.zeros(t), requires_grad=pose_refine)

    def is_visible(self, frame_idx: int) -> bool:
        return 0 <= frame_idx < self.valid.shape[0] and bool(self.valid[frame_idx])

    def pose(self, frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns world rotation (3,3) and translation (3,) of the box at ``frame_idx``."""
        base = self.base_poses[frame_idx]
        dyaw = self.delta_yaw[frame_idx]
        c, s = torch.cos(dyaw), torch.sin(dyaw)
        zero, one = torch.zeros_like(c), torch.ones_like(c)
        rz = torch.stack([torch.stack([c, -s, zero]), torch.stack([s, c, zero]), torch.stack([zero, zero, one])])
        rot = base[:3, :3] @ rz
        trans = base[:3, :3] @ self.delta_trans[frame_idx] + base[:3, 3]
        return rot, trans

    def world_gaussians(self, frame_idx: int) -> dict[str, torch.Tensor]:
        g = self.gaussians.activated()
        rot, trans = self.pose(frame_idx)
        base_q = rotmat_to_quat(self.base_poses[frame_idx, :3, :3])
        pose_q = quat_multiply(base_q, yaw_to_quat(self.delta_yaw[frame_idx]))
        g["means"] = g["means"] @ rot.T + trans
        g["quats"] = quat_multiply(pose_q.expand_as(g["quats"]), F.normalize(g["quats"], dim=-1))
        # SH coefficients are defined w.r.t. world view directions; for rigid objects we rotate the
        # view direction into the box frame instead (see SceneGraph.render), so keep a handle on rot.
        g["rot"] = rot
        return g


    @torch.no_grad()
    def interpolate_untrained(self, trained: torch.Tensor) -> int:
        """Replace the pose of every valid frame that had no training image with an interpolation of
        the *refined* poses of the nearest trained frames before and after it.

        Frames without images get no gradient, so they would otherwise keep the raw (noisy)
        annotation. Translation is interpolated linearly, heading (yaw) along the shortest arc.
        Outside the trained range the two nearest trained frames are extrapolated at constant
        velocity. Returns the number of frames changed.
        """
        trained = trained.to(self.valid.device) & self.valid
        known = trained.nonzero().squeeze(1).tolist()
        if not known:
            return 0
        refined = {t: self.pose(t) for t in known}

        def yaw(rot):
            return torch.atan2(rot[1, 0], rot[0, 0])

        changed = 0
        for t in range(self.valid.shape[0]):
            if not self.valid[t] or trained[t]:
                continue
            before = [k for k in known if k < t]
            after = [k for k in known if k > t]
            if before and after:
                a, b = before[-1], after[0]          # interpolate
            elif len(before) >= 2:
                a, b = before[-2], before[-1]        # extrapolate forward at constant velocity
            elif len(after) >= 2:
                a, b = after[0], after[1]            # extrapolate backward
            else:
                a = b = (before or after)[0]
            if a == b:
                rot, trans = refined[a]
            else:
                w = (t - a) / (b - a)
                (ra, ta), (rb, tb) = refined[a], refined[b]
                dy = torch.remainder(yaw(rb) - yaw(ra) + math.pi, 2 * math.pi) - math.pi
                y = yaw(ra) + w * dy
                trans = ta + w * (tb - ta)
                c, s_ = torch.cos(y), torch.sin(y)
                zero, one = torch.zeros_like(c), torch.ones_like(c)
                rot = torch.stack([torch.stack([c, -s_, zero]), torch.stack([s_, c, zero]),
                                   torch.stack([zero, zero, one])])
            self.base_poses[t, :3, :3] = rot
            self.base_poses[t, :3, 3] = trans
            self.delta_trans[t] = 0.0
            self.delta_yaw[t] = 0.0
            changed += 1
        return changed


class SkyNode(nn.Module):
    """Infinitely far background: colour as a function of world-space ray direction."""

    def __init__(self, hidden: int = 64, n_freqs: int = 4):
        super().__init__()
        self.n_freqs = n_freqs
        in_dim = 3 + 3 * 2 * n_freqs
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

    def encode(self, d: torch.Tensor) -> torch.Tensor:
        feats = [d]
        for i in range(self.n_freqs):
            f = (2.0**i) * math.pi
            feats += [torch.sin(f * d), torch.cos(f * d)]
        return torch.cat(feats, -1)

    def forward(self, dirs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(self.encode(dirs)))
