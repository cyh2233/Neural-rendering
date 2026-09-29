import math
import socket

import numpy as np
import pytest
import torch

from nr.data.synthetic import make_synthetic_scene
from nr.scene.graph import SceneGraph
from nr.utils.config import load_config
from nr.viz.viewer import (
    ViewState,
    box_corners,
    camera_to_viser,
    colorize_depth,
    render_view,
    viser_to_camera,
)


@pytest.fixture(scope="module")
def scene_and_graph():
    cfg = load_config("configs/synthetic.yaml")
    scene = make_synthetic_scene(num_frames=3, width=32, height=24, num_bg=60, num_obj=20)
    return scene, SceneGraph.from_scene(scene, cfg)


def test_camera_roundtrip(scene_and_graph):
    scene, _ = scene_and_graph
    cam = scene.cameras[3]
    wxyz, pos, fov = camera_to_viser(cam)
    back = viser_to_camera(wxyz, pos, fov, cam.width / cam.height, cam.width, cam.frame_idx)
    assert back.width == cam.width and back.height == cam.height
    assert torch.allclose(back.c2w, cam.c2w, atol=1e-5)
    assert torch.allclose(back.K, cam.K, atol=1e-3)  # principal point at the centre, square pixels


def test_viser_camera_fov():
    cam = viser_to_camera((1.0, 0, 0, 0), (0, 0, 0), math.radians(90), 2.0, 200, 0)
    assert (cam.width, cam.height) == (200, 100)
    assert cam.K[1, 1].item() == pytest.approx(50.0)


def test_render_modes(scene_and_graph):
    scene, graph = scene_and_graph
    cam = scene.cameras[0]
    imgs = {}
    for mode in ("rgb", "depth", "alpha", "background only"):
        img = render_view(graph, cam, ViewState(mode=mode), "torch")
        assert img.shape == (cam.height, cam.width, 3) and img.dtype == np.uint8
        imgs[mode] = img
    # removing the car must change the picture
    assert np.abs(imgs["rgb"].astype(int) - imgs["background only"].astype(int)).sum() > 0


def test_colorize_depth_near_is_warm():
    depth = np.array([[2.0, 50.0]])
    alpha = np.ones_like(depth)
    rgb = colorize_depth(depth, alpha, near=2.0, far=50.0)
    assert rgb[0, 0, 0] > rgb[0, 0, 2] and rgb[0, 1, 2] > rgb[0, 1, 0]
    assert colorize_depth(depth, np.zeros_like(depth)).max() == 0


def test_box_corners():
    c = box_corners(torch.eye(4), torch.tensor([4.0, 2.0, 1.0]))
    assert c.shape == (8, 3)
    assert np.allclose(c.max(0), [2, 1, 0.5]) and np.allclose(c.min(0), [-2, -1, -0.5])


def test_viewer_server_starts(scene_and_graph):
    pytest.importorskip("viser")
    from nr.viz.viewer import SceneViewer

    scene, graph = scene_and_graph
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    viewer = SceneViewer(graph, scene, backend="torch", host="127.0.0.1", port=port, width=64)
    try:
        viewer.gui_frame.value = 2
        viewer.gui_mode.value = "depth"
        viewer._sync_state()
        assert viewer.state.frame == 2 and viewer.state.mode == "depth"
        assert viewer.recorded_camera(2, scene.sensor_names[1]).cam_id == 1
    finally:
        viewer.stop()
