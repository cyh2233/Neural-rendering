"""nuScenes -> SceneData.

Only two methods of the devkit ``NuScenes`` object are used, ``get(table, token)`` and
``get_sample_data_path(token)``, so the conversion logic can be unit-tested with a mock.
LiDAR ``.pcd.bin`` files are read with numpy (x, y, z, intensity, ring).

Frames are the 2 Hz annotated keyframes (``sample`` records). For every keyframe we take
the six surround cameras and LIDAR_TOP.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import torch

from nr.data.types import BoxTrack, Camera, SceneData
from nr.utils.geometry import np_make_transform, np_quat_to_rotmat, np_transform_points

CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")
LIDAR = "LIDAR_TOP"


class NuScenesLike(Protocol):
    def get(self, table: str, token: str) -> dict: ...

    def get_sample_data_path(self, sample_data_token: str) -> str: ...


def read_lidar_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)


def record_to_transform(rec: dict) -> np.ndarray:
    """``calibrated_sensor`` / ``ego_pose`` / ``sample_annotation`` record -> 4x4 (child -> parent)."""
    return np_make_transform(np_quat_to_rotmat(np.asarray(rec["rotation"], dtype=np.float64)),
                             np.asarray(rec["translation"], dtype=np.float64))


def sensor_to_global(nusc: NuScenesLike, sd: dict) -> np.ndarray:
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return record_to_transform(ep) @ record_to_transform(cs)


def iter_samples(nusc: NuScenesLike, scene_token: str) -> list[dict]:
    scene = nusc.get("scene", scene_token)
    out, tok = [], scene["first_sample_token"]
    while tok:
        s = nusc.get("sample", tok)
        out.append(s)
        tok = s["next"]
    return out


def points_in_box(pts_local: np.ndarray, size: np.ndarray, margin: float = 0.0) -> np.ndarray:
    half = size / 2 + margin
    return np.all(np.abs(pts_local) <= half, axis=1)


def _category(nusc: NuScenesLike, ann: dict) -> str:
    if "category_name" in ann:
        return ann["category_name"]
    inst = nusc.get("instance", ann["instance_token"])
    return nusc.get("category", inst["category_token"])["name"]


def convert_scene(
    nusc: NuScenesLike,
    scene_name_or_token: str,
    cameras: tuple[str, ...] = CAMERAS,
    categories: tuple[str, ...] = ("vehicle.",),
    moving_threshold: float = 1.0,
    ego_radius: float = 2.5,
    max_lidar_range: float = 80.0,
    lidar_loader: Callable[[str], np.ndarray] = read_lidar_bin,
    image_loader: Callable[[str], np.ndarray] | None = None,
    scene_tokens: dict[str, str] | None = None,
) -> SceneData:
    """Convert one nuScenes scene.

    ``categories``: prefixes of annotation categories modelled as rigid dynamic objects.
    ``moving_threshold``: tracks whose centre moves less than this [m] stay in the background.
    ``image_loader``: if given, LiDAR points are coloured from the camera images.
    ``scene_tokens``: optional {scene name: token} map (else ``scene_name_or_token`` is a token).
    """
    scene_token = (scene_tokens or {}).get(scene_name_or_token, scene_name_or_token)
    scene = nusc.get("scene", scene_token)
    samples = iter_samples(nusc, scene_token)
    T = len(samples)

    # World origin = first ego position (translation only; keep global z-up orientation).
    first_lidar = nusc.get("sample_data", samples[0]["data"][LIDAR])
    origin = np.asarray(nusc.get("ego_pose", first_lidar["ego_pose_token"])["translation"], dtype=np.float64)
    shift = np_make_transform(np.eye(3), -origin)

    # ---------------------------------------------------------------- annotations -> tracks
    tracks: dict[str, dict] = {}
    for t, s in enumerate(samples):
        for ann_tok in s["anns"]:
            ann = nusc.get("sample_annotation", ann_tok)
            cat = _category(nusc, ann)
            if not any(cat.startswith(p) for p in categories):
                continue
            tr = tracks.setdefault(ann["instance_token"], {
                "category": cat,
                "size": np.asarray(ann["size"], dtype=np.float64)[[1, 0, 2]],  # wlh -> l, w, h
                "poses": np.tile(np.eye(4), (T, 1, 1)),
                "valid": np.zeros(T, dtype=bool),
            })
            tr["poses"][t] = shift @ record_to_transform(ann)
            tr["valid"][t] = True
    moving = {}
    for tok, tr in tracks.items():
        centres = tr["poses"][tr["valid"], :3, 3]
        if len(centres) >= 2 and np.linalg.norm(centres - centres[0], axis=1).max() > moving_threshold:
            _fill_invalid_poses(tr["poses"], tr["valid"])
            moving[tok] = tr

    # ---------------------------------------------------------------- cameras
    cams: list[Camera] = []
    for t, s in enumerate(samples):
        for cam_id, name in enumerate(cameras):
            sd = nusc.get("sample_data", s["data"][name])
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            c2w = shift @ sensor_to_global(nusc, sd)
            path = nusc.get_sample_data_path(sd["token"])
            cams.append(Camera(
                K=torch.tensor(cs["camera_intrinsic"], dtype=torch.float32),
                c2w=torch.from_numpy(c2w).float(),
                width=int(sd.get("width", 1600)),
                height=int(sd.get("height", 900)),
                cam_id=cam_id,
                frame_idx=t,
                image_path=path,
                name=f"{t:03d}_{name}",
            ))

    # ---------------------------------------------------------------- LiDAR
    bg_pts, bg_cols = [], []
    obj_pts: dict[str, list] = {k: [] for k in moving}
    obj_cols: dict[str, list] = {k: [] for k in moving}
    for t, s in enumerate(samples):
        sd = nusc.get("sample_data", s["data"][LIDAR])
        pts = lidar_loader(nusc.get_sample_data_path(sd["token"]))
        dist = np.linalg.norm(pts[:, :2], axis=1)
        pts = pts[(dist > ego_radius) & (dist < max_lidar_range)]
        world = np_transform_points(shift @ sensor_to_global(nusc, sd), pts)

        frame_cams = [i for i, c in enumerate(cams) if c.frame_idx == t]
        cols = np.full((len(world), 3), 0.5)
        for i in frame_cams:
            c = cams[i]
            uvz = _project(world, c)
            ok = uvz[:, 2] > 0.5
            ok &= (uvz[:, 0] >= 0) & (uvz[:, 0] < c.width) & (uvz[:, 1] >= 0) & (uvz[:, 1] < c.height)
            c.lidar_uvz = torch.from_numpy(uvz[ok]).float()
            if image_loader is not None:
                img = np.asarray(image_loader(c.image_path))
                u = uvz[ok, 0].astype(int)
                v = uvz[ok, 1].astype(int)
                cols[ok] = img[v, u, :3] / 255.0
        is_obj = np.zeros(len(world), dtype=bool)
        for tok, tr in moving.items():
            if not tr["valid"][t]:
                continue
            local = np_transform_points(np.linalg.inv(tr["poses"][t]), world)
            near_box = points_in_box(local, tr["size"], margin=0.2)
            strict = near_box & points_in_box(local, tr["size"])
            obj_pts[tok].append(local[strict])
            obj_cols[tok].append(cols[strict])
            is_obj |= near_box
        bg_pts.append(world[~is_obj])
        bg_cols.append(cols[~is_obj])

    points = torch.from_numpy(np.concatenate(bg_pts)).float()
    colors = torch.from_numpy(np.concatenate(bg_cols)).float()

    box_tracks = []
    for tok, tr in moving.items():
        p = np.concatenate(obj_pts[tok]) if obj_pts[tok] else np.zeros((0, 3))
        pc = np.concatenate(obj_cols[tok]) if obj_cols[tok] else np.zeros((0, 3))
        box_tracks.append(BoxTrack(
            instance_id=tok,
            category=tr["category"],
            size=torch.from_numpy(tr["size"]).float(),
            poses=torch.from_numpy(tr["poses"]).float(),
            valid=torch.from_numpy(tr["valid"]),
            points=torch.from_numpy(p).float(),
            colors=torch.from_numpy(pc).float(),
        ))

    timestamps = np.array([s["timestamp"] for s in samples], dtype=np.float64) * 1e-6
    return SceneData(
        name=scene["name"],
        cameras=cams,
        frame_timestamps=timestamps - timestamps[0],
        points=points,
        colors=colors,
        tracks=box_tracks,
        sensor_names=list(cameras),
    )


def _project(world: np.ndarray, cam: Camera) -> np.ndarray:
    w2c = np.linalg.inv(cam.c2w.double().numpy())
    pc = np_transform_points(w2c, world)
    K = cam.K.double().numpy()
    z = pc[:, 2]
    uv = pc[:, :2] / np.maximum(z, 1e-6)[:, None] @ K[:2, :2].T + K[:2, 2]
    return np.concatenate([uv, z[:, None]], 1)


def _fill_invalid_poses(poses: np.ndarray, valid: np.ndarray) -> None:
    """Copy the nearest valid pose into frames where the object was not annotated (kept hidden
    by ``valid``, but gives pose-refinement parameters a sane value)."""
    idx = np.nonzero(valid)[0]
    for t in range(len(valid)):
        if not valid[t]:
            poses[t] = poses[idx[np.abs(idx - t).argmin()]]


def load_nuscenes(dataroot: str, version: str = "v1.0-mini"):
    from nuscenes.nuscenes import NuScenes

    return NuScenes(version=version, dataroot=dataroot, verbose=False)


def scene_name_to_token(nusc) -> dict[str, str]:
    return {s["name"]: s["token"] for s in nusc.scene}
