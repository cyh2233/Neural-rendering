"""A tiny procedurally generated driving scene for CPU tests.

A road plane with a checker texture, two side walls, and one car-sized box that drives
faster than the ego vehicle in the left lane. Two forward cameras (yawed +-15 deg) move
along +x. Ground-truth images are rendered with the reference rasterizer, and "LiDAR"
points are noisy samples of the true Gaussian centres, mimicking a real log.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from nr.data.types import BoxTrack, Camera, SceneData
from nr.render.torch_backend import rasterize_torch
from nr.utils.geometry import make_transform, quat_to_rotmat, yaw_to_quat

SKY = torch.tensor([0.55, 0.7, 0.95])

# OpenCV camera looking along world +x with world z up.
_R_FORWARD = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _rz(yaw: float) -> torch.Tensor:
    return quat_to_rotmat(yaw_to_quat(torch.tensor(yaw)))


def _background(n: int, gen: torch.Generator):
    n_ground = n * 2 // 3
    n_wall = n - n_ground
    g = torch.rand(n_ground, 3, generator=gen)
    ground = torch.stack([g[:, 0] * 45 - 5, g[:, 1] * 16 - 8, torch.zeros(n_ground)], -1)
    checker = ((ground[:, 0] // 2 + ground[:, 1] // 2) % 2)[:, None]
    ground_col = 0.25 + 0.2 * checker + 0.05 * torch.rand(n_ground, 3, generator=gen)
    w = torch.rand(n_wall, 3, generator=gen)
    side = torch.where(w[:, 1] > 0.5, 8.0, -8.0)
    walls = torch.stack([w[:, 0] * 45 - 5, side, w[:, 2] * 4], -1)
    stripe = ((walls[:, 0] // 3) % 3)[:, None] / 2
    wall_col = torch.cat([0.6 + 0.3 * stripe, 0.35 + 0.1 * stripe, 0.3 - 0.1 * stripe], -1)
    return torch.cat([ground, walls]), torch.cat([ground_col, wall_col]).clamp(0, 1)


def _car(n: int, size: torch.Tensor, gen: torch.Generator):
    u = torch.rand(n, 3, generator=gen) * 2 - 1
    axis = torch.randint(0, 3, (n,), generator=gen)
    u[torch.arange(n), axis] = torch.sign(u[torch.arange(n), axis])
    pts = u * size / 2
    col = torch.tensor([0.85, 0.1, 0.1]).repeat(n, 1)
    col[pts[:, 2] > size[2] * 0.3] = torch.tensor([0.2, 0.2, 0.3])  # darker "windows" on top
    return pts, col


def make_synthetic_scene(
    num_frames: int = 8,
    width: int = 64,
    height: int = 48,
    num_bg: int = 400,
    num_obj: int = 80,
    seed: int = 0,
    lidar_noise: float = 0.05,
) -> SceneData:
    gen = torch.Generator().manual_seed(seed)
    bg_pts, bg_col = _background(num_bg, gen)
    car_size = torch.tensor([4.0, 2.0, 1.5])
    car_pts, car_col = _car(num_obj, car_size, gen)

    ego_speed, car_speed = 1.0, 2.0
    car_poses = torch.stack(
        [make_transform(torch.eye(3), torch.tensor([10.0 + car_speed * t, 3.0, 0.75])) for t in range(num_frames)]
    )

    f = 0.8 * width
    K = torch.tensor([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])
    yaws = [math.radians(15), math.radians(-15)]

    def gt_render(c2w: torch.Tensor, frame: int):
        pose = car_poses[frame]
        means = torch.cat([bg_pts, car_pts @ pose[:3, :3].T + pose[:3, 3]])
        cols = torch.cat([bg_col, car_col])
        n = means.shape[0]
        scales = torch.cat([torch.full((num_bg, 3), 0.45), torch.full((num_obj, 3), 0.35)])
        quats = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 1)
        opac = torch.full((n,), 0.9)
        rgb, alpha, _, _ = rasterize_torch(
            means, quats, scales, opac, cols, torch.linalg.inv(c2w), K, width, height
        )
        return (rgb + (1 - alpha) * SKY).clamp(0, 1)

    cameras = []
    for t in range(num_frames):
        for cam_id, yaw in enumerate(yaws):
            rot = _rz(yaw) @ _R_FORWARD
            c2w = make_transform(rot, torch.tensor([ego_speed * t, 0.0, 1.5]))
            img = gt_render(c2w, t)
            # Sparse "LiDAR" depth: project a random subset of true surface points.
            pose = car_poses[t]
            surf = torch.cat([bg_pts, car_pts @ pose[:3, :3].T + pose[:3, 3]])
            pc = (surf - c2w[:3, 3]) @ c2w[:3, :3]
            ok = pc[:, 2] > 0.1
            uv = pc[ok, :2] / pc[ok, 2:3] * f + torch.tensor([width / 2, height / 2])
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            lidar = torch.cat([uv[inside], pc[ok][inside, 2:3]], -1)
            cameras.append(
                Camera(K=K.clone(), c2w=c2w, width=width, height=height, cam_id=cam_id,
                       frame_idx=t, image=img, lidar_uvz=lidar, name=f"f{t:03d}_c{cam_id}")
            )

    noisy_bg = bg_pts + lidar_noise * torch.randn(bg_pts.shape, generator=gen)
    noisy_col = (bg_col + 0.1 * torch.randn(bg_col.shape, generator=gen)).clamp(0, 1)
    # Perturb the "annotated" box poses slightly so pose refinement has something to fix.
    ann_poses = car_poses.clone()
    ann_poses[:, :3, 3] += 0.05 * torch.randn(num_frames, 3, generator=gen)
    track = BoxTrack(
        instance_id="car_0",
        category="vehicle.car",
        size=car_size,
        poses=ann_poses,
        valid=torch.ones(num_frames, dtype=torch.bool),
        points=car_pts + lidar_noise * torch.randn(car_pts.shape, generator=gen),
        colors=(car_col + 0.1 * torch.randn(car_col.shape, generator=gen)).clamp(0, 1),
    )
    return SceneData(
        name="synthetic",
        cameras=cameras,
        frame_timestamps=np.arange(num_frames, dtype=np.float64) * 0.5,
        points=noisy_bg,
        colors=noisy_col,
        tracks=[track],
        sensor_names=["CAM_LEFT", "CAM_RIGHT"],
    )
