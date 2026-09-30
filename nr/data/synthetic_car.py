"""A detailed procedural car in a street, for checking that the pipeline can recover fine detail.

The car (about 5,000 small, flat Gaussians) has a blue body, glass windows with pillars, four
wheels with tyres and silver hubs, headlights, a grille, red tail lights and white number plates.
The street has asphalt, dashed lane markings, pavements and building facades with windows.

Scenario: the ego vehicle (2.5 m/frame) overtakes the car (1 m/frame) in the left lane. A front,
a left and a rear camera see the car's back, right side and front in turn, as in a real log.
Ground truth is rendered with the reference rasterizer. "LiDAR" is a noisy subset of the true
Gaussian centres, and per-image depth samples come from the ground-truth depth render, so they
respect occlusion.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from nr.data.types import BoxTrack, Camera, SceneData
from nr.render.torch_backend import rasterize_torch
from nr.utils.geometry import make_transform, quat_from_z, quat_to_rotmat, yaw_to_quat

CAR_SIZE = (4.2, 1.8, 1.5)  # length, width, height of the annotation box
BODY = (0.1, 0.25, 0.75)
GLASS = (0.06, 0.08, 0.12)
TYRE = (0.04, 0.04, 0.04)
HUB = (0.75, 0.75, 0.78)
HEADLIGHT = (1.0, 0.95, 0.75)
TAILLIGHT = (0.95, 0.1, 0.05)
PLATE = (0.95, 0.95, 0.95)
GRILLE = (0.15, 0.15, 0.15)
SKY_HORIZON = torch.tensor([0.85, 0.9, 0.97])
SKY_ZENITH = torch.tensor([0.35, 0.55, 0.9])


class _Builder:
    """Accumulates flat Gaussians (disc-like splats oriented by a surface normal)."""

    def __init__(self, gen: torch.Generator):
        self.gen = gen
        self.means, self.normals, self.colors, self.scales = [], [], [], []

    def rect(self, center, e1, e2, half1, half2, normal, spacing, color_fn, thickness=0.006):
        n1, n2 = max(1, int(2 * half1 / spacing)), max(1, int(2 * half2 / spacing))
        s = (torch.arange(n1) + 0.5) / n1 * 2 * half1 - half1
        t = (torch.arange(n2) + 0.5) / n2 * 2 * half2 - half2
        ss, tt = torch.meshgrid(s, t, indexing="ij")
        ss = ss.reshape(-1) + (torch.rand(ss.numel(), generator=self.gen) - 0.5) * spacing * 0.3
        tt = tt.reshape(-1) + (torch.rand(tt.numel(), generator=self.gen) - 0.5) * spacing * 0.3
        c, e1, e2 = torch.tensor(center), torch.tensor(e1), torch.tensor(e2)
        pts = c + ss[:, None] * e1 + tt[:, None] * e2
        self._add(pts, normal, color_fn(ss, tt, pts), spacing * 0.6, thickness)

    def disc(self, center, normal, radius, spacing, color_fn):
        n = int(2 * radius / spacing)
        g = (torch.arange(n) + 0.5) / n * 2 * radius - radius
        a, b = torch.meshgrid(g, g, indexing="ij")
        a, b = a.reshape(-1), b.reshape(-1)
        keep = a * a + b * b <= radius * radius
        a, b = a[keep], b[keep]
        # disc in the x-z plane (normal along +-y)
        pts = torch.tensor(center) + torch.stack([a, torch.zeros_like(a), b], -1)
        r = torch.sqrt(a * a + b * b)
        self._add(pts, normal, color_fn(r), spacing * 0.6, 0.006)

    def cylinder(self, center, radius, half_width, spacing, color):
        n_ang = int(2 * math.pi * radius / spacing)
        n_w = max(1, int(2 * half_width / spacing))
        ang = (torch.arange(n_ang) + 0.5) / n_ang * 2 * math.pi
        w = (torch.arange(n_w) + 0.5) / n_w * 2 * half_width - half_width
        aa, ww = torch.meshgrid(ang, w, indexing="ij")
        aa, ww = aa.reshape(-1), ww.reshape(-1)
        nrm = torch.stack([torch.cos(aa), torch.zeros_like(aa), torch.sin(aa)], -1)
        pts = torch.tensor(center) + radius * nrm + torch.stack([torch.zeros_like(ww), ww, torch.zeros_like(ww)], -1)
        self._add(pts, nrm, torch.tensor(color).expand(pts.shape[0], 3), spacing * 0.6, 0.006)

    def _add(self, pts, normal, colors, in_plane, thickness):
        n = pts.shape[0]
        nrm = torch.as_tensor(normal, dtype=torch.float32)
        nrm = nrm.expand(n, 3) if nrm.dim() == 1 else nrm
        self.means.append(pts.float())
        self.normals.append(nrm.float())
        self.colors.append(torch.as_tensor(colors, dtype=torch.float32).expand(n, 3).clone())
        self.scales.append(torch.tensor([in_plane, in_plane, thickness]).expand(n, 3))

    def build(self, opacity=0.95):
        means = torch.cat(self.means)
        n = means.shape[0]
        return {
            "means": means,
            "quats": quat_from_z(torch.nn.functional.normalize(torch.cat(self.normals), dim=-1)),
            "scales": torch.cat(self.scales).clone(),
            "opacities": torch.full((n,), opacity),
            "colors": torch.cat(self.colors).clamp(0, 1),
        }


def _where(mask, a, b):
    return torch.where(mask[:, None], torch.tensor(a), torch.tensor(b))


def build_car(gen: torch.Generator, spacing: float = 0.07) -> dict:
    """Car Gaussians in the box frame: +x forward, +y left, z = 0 at the box centre (ground at -0.75)."""
    b = _Builder(gen)
    L, W = 2.1, 0.9          # half length / width of the body
    z0, z1 = -0.45, 0.25     # body bottom / top (0.3 m ground clearance)
    cl0, cl1, cw = -1.1, 0.9, 0.8  # cabin x range and half width
    zc = 0.75                # roof

    def body(*_):
        return torch.tensor(BODY)

    # --- body: top (outside the cabin footprint), sides, front, back
    def top_color(s, t, p):
        return torch.tensor(BODY).expand(p.shape[0], 3)

    b.rect(((L + cl1) / 2, 0, z1), (1, 0, 0), (0, 1, 0), (L - cl1) / 2, W, (0, 0, 1), spacing, top_color)   # bonnet
    b.rect(((cl0 - L) / 2, 0, z1), (1, 0, 0), (0, 1, 0), (L + cl0) / 2, W, (0, 0, 1), spacing, top_color)  # boot
    for sign in (1, -1):
        b.rect((0, sign * W, (z0 + z1) / 2), (1, 0, 0), (0, 0, 1), L, (z1 - z0) / 2, (0, sign, 0), spacing, body)

    def front_color(s, t, p):  # s along +y, t along +z
        y, z = p[:, 1], p[:, 2]
        c = torch.tensor(BODY).expand(p.shape[0], 3).clone()
        c[(y.abs() > 0.5) & (y.abs() < 0.82) & (z > 0.02) & (z < 0.17)] = torch.tensor(HEADLIGHT)
        c[(y.abs() < 0.45) & (z > -0.25) & (z < 0.1)] = torch.tensor(GRILLE)
        c[(y.abs() < 0.26) & (z > -0.4) & (z < -0.29)] = torch.tensor(PLATE)
        return c

    def back_color(s, t, p):
        y, z = p[:, 1], p[:, 2]
        c = torch.tensor(BODY).expand(p.shape[0], 3).clone()
        c[(y.abs() > 0.5) & (y.abs() < 0.85) & (z > 0.0) & (z < 0.18)] = torch.tensor(TAILLIGHT)
        c[(y.abs() < 0.28) & (z > -0.28) & (z < -0.12)] = torch.tensor(PLATE)
        return c

    b.rect((L, 0, (z0 + z1) / 2), (0, 1, 0), (0, 0, 1), W, (z1 - z0) / 2, (1, 0, 0), spacing * 0.8, front_color)
    b.rect((-L, 0, (z0 + z1) / 2), (0, 1, 0), (0, 0, 1), W, (z1 - z0) / 2, (-1, 0, 0), spacing * 0.8, back_color)

    # --- cabin: windows with pillars
    xc, hl = (cl0 + cl1) / 2, (cl1 - cl0) / 2

    def side_window(s, t, p):
        x, z = p[:, 0], p[:, 2]
        glass = (x > cl0 + 0.12) & (x < cl1 - 0.12) & (z > z1 + 0.07) & (z < zc - 0.07) & ((x - (xc - 0.1)).abs() > 0.06)
        return _where(glass, GLASS, BODY)

    def end_window(s, t, p):
        y, z = p[:, 1], p[:, 2]
        glass = (y.abs() < cw - 0.1) & (z > z1 + 0.06) & (z < zc - 0.06)
        return _where(glass, GLASS, BODY)

    for sign in (1, -1):
        b.rect((xc, sign * cw, (z1 + zc) / 2), (1, 0, 0), (0, 0, 1), hl, (zc - z1) / 2, (0, sign, 0), spacing * 0.8, side_window)
    b.rect((cl1, 0, (z1 + zc) / 2), (0, 1, 0), (0, 0, 1), cw, (zc - z1) / 2, (1, 0, 0), spacing * 0.8, end_window)
    b.rect((cl0, 0, (z1 + zc) / 2), (0, 1, 0), (0, 0, 1), cw, (zc - z1) / 2, (-1, 0, 0), spacing * 0.8, end_window)
    b.rect((xc, 0, zc), (1, 0, 0), (0, 1, 0), hl, cw, (0, 0, 1), spacing, top_color)

    # --- wheels
    r_wheel, zw = 0.35, -0.75 + 0.35
    for x in (1.35, -1.35):
        for sign in (1, -1):
            b.disc((x, sign * (W + 0.02), zw), (0, sign, 0), r_wheel, spacing * 0.7,
                   lambda r: _where(r < 0.2, HUB, TYRE))
            b.cylinder((x, sign * (W - 0.1), zw), r_wheel, 0.11, spacing, TYRE)
    return b.build()


def build_street(gen: torch.Generator, x_range=(-20.0, 80.0)) -> dict:
    b = _Builder(gen)
    x0, x1 = x_range
    xm, xh = (x0 + x1) / 2, (x1 - x0) / 2

    def asphalt(s, t, p):
        return (0.28 + 0.04 * torch.randn(p.shape[0], 1, generator=gen)).expand(p.shape[0], 3)

    b.rect((xm, 0, 0), (1, 0, 0), (0, 1, 0), xh, 5.25, (0, 0, 1), 0.5, asphalt, thickness=0.01)
    for y in (1.75, -1.75):
        for xd in np.arange(x0, x1, 9.0):
            b.rect((xd + 1.5, y, 0.01), (1, 0, 0), (0, 1, 0), 1.5, 0.08, (0, 0, 1), 0.08,
                   lambda s, t, p: torch.tensor([0.95, 0.95, 0.9]), thickness=0.005)
    for sign in (1, -1):
        b.rect((xm, sign * 6.6, 0.15), (1, 0, 0), (0, 1, 0), xh, 1.35, (0, 0, 1), 0.6,
               lambda s, t, p: (0.62 + 0.03 * torch.randn(p.shape[0], 1, generator=gen)).expand(p.shape[0], 3),
               thickness=0.01)

        def facade(s, t, p, sign=sign):
            x, z = p[:, 0], p[:, 2]
            block = torch.floor((x - x0) / 12.0).long() % 3
            base = torch.tensor([[0.85, 0.75, 0.6], [0.7, 0.45, 0.35], [0.8, 0.8, 0.75]])[block]
            win = ((x - x0) % 3.0 > 0.8) & ((x - x0) % 3.0 < 2.2) & (z % 3.0 > 1.0) & (z % 3.0 < 2.6) & (z > 0.9)
            base[win] = torch.tensor([0.2, 0.3, 0.45])
            return base

        b.rect((xm, sign * 8.5, 4.0), (1, 0, 0), (0, 0, 1), xh, 4.0, (0, -sign, 0), 0.4, facade, thickness=0.02)
    return b.build(opacity=0.98)


def _rz(yaw: float) -> torch.Tensor:
    return quat_to_rotmat(yaw_to_quat(torch.tensor(yaw)))


_R_FORWARD = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
SENSORS = (("CAM_FRONT", 0.0), ("CAM_LEFT", math.pi / 2), ("CAM_BACK", math.pi))


def _sky(dirs: torch.Tensor) -> torch.Tensor:
    t = dirs[..., 2:3].clamp(0, 1) ** 0.6
    return SKY_HORIZON * (1 - t) + SKY_ZENITH * t


def make_car_scene(
    num_frames: int = 16,
    width: int = 256,
    height: int = 144,
    seed: int = 0,
    car_spacing: float = 0.07,
    lidar_fraction: float = 0.3,
    lidar_noise: float = 0.02,
    depth_samples: int = 1500,
    ego_speed: float = 2.5,
    car_speed: float = 1.0,
    car_start: float = 9.0,
) -> SceneData:
    from nr.scene.graph import camera_ray_dirs

    gen = torch.Generator().manual_seed(seed)
    car = build_car(gen, car_spacing)
    street = build_street(gen)
    size = torch.tensor(CAR_SIZE)
    car_poses = torch.stack(
        [make_transform(torch.eye(3), torch.tensor([car_start + car_speed * t, 3.5, 0.75])) for t in range(num_frames)]
    )
    f = 0.55 * width
    K = torch.tensor([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])

    cameras = []
    for t in range(num_frames):
        pose = car_poses[t]
        rot = pose[:3, :3]
        all_g = {
            "means": torch.cat([street["means"], car["means"] @ rot.T + pose[:3, 3]]),
            "quats": torch.cat([street["quats"], car["quats"]]),  # car yaw is 0, so no rotation needed
            "scales": torch.cat([street["scales"], car["scales"]]),
            "opacities": torch.cat([street["opacities"], car["opacities"]]),
            "colors": torch.cat([street["colors"], car["colors"]]),
        }
        for cam_id, (name, yaw) in enumerate(SENSORS):
            c2w = make_transform(_rz(yaw) @ _R_FORWARD, torch.tensor([ego_speed * t, 0.0, 1.6]))
            cam = Camera(K=K.clone(), c2w=c2w, width=width, height=height, cam_id=cam_id, frame_idx=t,
                         name=f"{t:03d}_{name}")
            rgb, alpha, depth, _ = rasterize_torch(
                all_g["means"], all_g["quats"], all_g["scales"], all_g["opacities"], all_g["colors"],
                torch.linalg.inv(c2w), K, width, height,
            )
            img = (rgb + (1 - alpha) * _sky(camera_ray_dirs(cam, "cpu"))).clamp(0, 1)
            # depth samples where the render is opaque (occlusion-correct, like a real LiDAR return)
            opaque = (alpha[..., 0] > 0.99).reshape(-1).nonzero().squeeze(1)
            pick = opaque[torch.randperm(opaque.numel(), generator=gen)[:depth_samples]]
            vv, uu = pick // width, pick % width
            cam.lidar_uvz = torch.stack([uu.float() + 0.5, vv.float() + 0.5, depth.reshape(-1)[pick]], -1)
            cam.image = img
            cameras.append(cam)

    def lidar(g):
        n = g["means"].shape[0]
        keep = torch.randperm(n, generator=gen)[: int(n * lidar_fraction)]
        pts = g["means"][keep] + lidar_noise * torch.randn(keep.numel(), 3, generator=gen)
        cols = (g["colors"][keep] + 0.08 * torch.randn(keep.numel(), 3, generator=gen)).clamp(0, 1)
        return pts, cols

    bg_pts, bg_cols = lidar(street)
    car_pts, car_cols = lidar(car)
    ann = car_poses.clone()
    ann[:, :3, 3] += 0.05 * torch.randn(num_frames, 3, generator=gen)
    track = BoxTrack(instance_id="car_0", category="vehicle.car", size=size, poses=ann,
                     valid=torch.ones(num_frames, dtype=torch.bool), points=car_pts, colors=car_cols)
    return SceneData(
        name="synthetic_car",
        cameras=cameras,
        frame_timestamps=np.arange(num_frames, dtype=np.float64) * 0.5,
        points=bg_pts,
        colors=bg_cols,
        tracks=[track],
        sensor_names=[n for n, _ in SENSORS],
    )
