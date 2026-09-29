"""Tests the nuScenes conversion with a tiny in-memory mock of the devkit tables."""

import numpy as np
import pytest
import torch

from nr.data.cache import load_scene, save_scene
from nr.data.nuscenes import convert_scene
from nr.utils.geometry import rotmat_to_quat

T = 3
EGO = [np.array([100.0 + 2 * t, 200.0, 0.0]) for t in range(T)]
CAR_A = [np.array([110.0 + 5 * t, 200.0, 1.0]) for t in range(T)]  # moving car
CAR_B = np.array([130.0, 195.0, 1.0])                              # parked car
LIDAR_H = 1.8
# OpenCV camera -> ego (x fwd, y left, z up)
R_CAM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
CAM_Q = rotmat_to_quat(torch.from_numpy(R_CAM)).tolist()
IDENT_Q = [1.0, 0.0, 0.0, 0.0]


class MockNuScenes:
    def __init__(self):
        t = {k: {} for k in ("scene", "sample", "sample_data", "calibrated_sensor", "ego_pose",
                             "sample_annotation", "instance", "category")}
        t["scene"]["sc"] = {"token": "sc", "name": "scene-mock", "first_sample_token": "s0"}
        t["calibrated_sensor"]["cs_cam"] = {"translation": [1.5, 0.0, 1.5], "rotation": CAM_Q,
                                            "camera_intrinsic": [[100.0, 0, 80], [0, 100, 45], [0, 0, 1]]}
        t["calibrated_sensor"]["cs_lidar"] = {"translation": [0.0, 0.0, LIDAR_H], "rotation": IDENT_Q}
        for i in range(T):
            t["ego_pose"][f"ep{i}"] = {"translation": EGO[i].tolist(), "rotation": IDENT_Q}
            t["sample_data"][f"cam{i}"] = {"token": f"cam{i}", "calibrated_sensor_token": "cs_cam",
                                           "ego_pose_token": f"ep{i}", "width": 160, "height": 90}
            t["sample_data"][f"lid{i}"] = {"token": f"lid{i}", "calibrated_sensor_token": "cs_lidar",
                                           "ego_pose_token": f"ep{i}"}
            anns = []
            for name, pos, cat in (("a", CAR_A[i], "vehicle.car"), ("b", CAR_B, "vehicle.car"),
                                   ("p", CAR_A[i] + [0, 5, 0], "human.pedestrian.adult")):
                tok = f"ann_{name}{i}"
                t["sample_annotation"][tok] = {"instance_token": f"inst_{name}", "translation": pos.tolist(),
                                               "size": [2.0, 4.0, 1.5], "rotation": IDENT_Q, "category_name": cat}
                anns.append(tok)
            t["sample"][f"s{i}"] = {"token": f"s{i}", "timestamp": 1_000_000 + 500_000 * i,
                                    "next": f"s{i + 1}" if i + 1 < T else "",
                                    "data": {"CAM_FRONT": f"cam{i}", "LIDAR_TOP": f"lid{i}"}, "anns": anns}
        self.t = t

    def get(self, table, token):
        return self.t[table][token]

    def get_sample_data_path(self, token):
        return f"/fake/{token}"


def fake_lidar(path: str) -> np.ndarray:
    """Ground grid + a dense cluster inside car A, returned in the LiDAR sensor frame."""
    i = int(path.split("lid")[-1])
    gx, gy = np.meshgrid(np.linspace(95, 150, 40), np.linspace(190, 210, 15))
    ground = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    rng = np.random.default_rng(i)
    car = CAR_A[i] + rng.uniform(-1, 1, (200, 3)) * [1.8, 0.9, 0.6]
    world = np.concatenate([ground, car])
    return world - EGO[i] - [0, 0, LIDAR_H]


@pytest.fixture(scope="module")
def scene():
    return convert_scene(MockNuScenes(), "sc", cameras=("CAM_FRONT",), lidar_loader=fake_lidar)


def test_frames_and_cameras(scene):
    assert scene.name == "scene-mock"
    assert scene.num_frames == T
    assert np.allclose(scene.frame_timestamps, [0.0, 0.5, 1.0])
    assert len(scene.cameras) == T
    for i, c in enumerate(scene.cameras):
        assert c.frame_idx == i and c.width == 160 and c.height == 90
        # world origin = first ego position, so camera is 1.5 m ahead + 2 m per frame, 1.5 m up
        assert torch.allclose(c.c2w[:3, 3], torch.tensor([1.5 + 2 * i, 0.0, 1.5]), atol=1e-5)
        # OpenCV forward (+z) must point along world +x
        assert torch.allclose(c.c2w[:3, 2], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
        assert c.image_path == f"/fake/cam{i}"


def test_only_moving_vehicles_become_tracks(scene):
    assert len(scene.tracks) == 1
    tr = scene.tracks[0]
    assert tr.instance_id == "inst_a" and tr.category == "vehicle.car"
    assert torch.allclose(tr.size, torch.tensor([4.0, 2.0, 1.5]))  # wlh -> l, w, h
    assert tr.valid.all()
    for i in range(T):
        assert torch.allclose(tr.poses[i, :3, 3], torch.tensor([10.0 + 5 * i, 0.0, 1.0]), atol=1e-5)


def test_lidar_split_between_background_and_object(scene):
    tr = scene.tracks[0]
    assert tr.points.shape[0] > 300
    assert (tr.points.abs() <= tr.size / 2 + 1e-4).all()
    for i in range(T):
        local = scene.points - tr.poses[i, :3, 3]
        inside = (local.abs() <= tr.size / 2).all(-1)
        assert not inside.any()  # no ground point happens to sit in the box, and car points were removed
    assert scene.points.shape[0] > 0


def test_lidar_projection_into_cameras(scene):
    for c in scene.cameras:
        uvz = c.lidar_uvz
        assert uvz is not None and uvz.shape[0] > 10
        assert (uvz[:, 2] > 0).all()
        assert (uvz[:, 0] >= 0).all() and (uvz[:, 0] < c.width).all()
        assert (uvz[:, 1] >= 0).all() and (uvz[:, 1] < c.height).all()


def test_cache_roundtrip(scene, tmp_path):
    p = tmp_path / "s.pt"
    save_scene(scene, p)
    s2 = load_scene(p)
    assert s2.name == scene.name and len(s2.cameras) == len(scene.cameras)
    assert torch.equal(s2.points, scene.points)
    assert torch.equal(s2.tracks[0].poses, scene.tracks[0].poses)
    assert torch.equal(s2.cameras[1].lidar_uvz, scene.cameras[1].lidar_uvz)
