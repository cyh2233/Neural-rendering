import torch

from nr.data.synthetic_car import build_car, make_car_scene
from nr.eval.nvs import evaluate_views, object_bbox
from nr.scene.graph import SceneGraph
from nr.utils.config import load_config
from nr.utils.geometry import make_transform


def test_car_has_detail_colours():
    car = build_car(torch.Generator().manual_seed(0))
    n = car["means"].shape[0]
    assert 3000 < n < 8000
    cols = car["colors"]
    # body blue, glass, tyres, lights and plates are all present
    for rgb in ([0.1, 0.25, 0.75], [0.06, 0.08, 0.12], [0.04, 0.04, 0.04], [0.95, 0.1, 0.05], [0.95, 0.95, 0.95]):
        assert ((cols - torch.tensor(rgb)).abs().sum(-1) < 1e-3).sum() > 10, rgb
    # splats are flat: the smallest scale is much smaller than the other two
    s = car["scales"].sort(-1).values
    assert (s[:, 0] * 3 < s[:, 1]).all()
    # everything fits in the annotation box (4.2 x 1.8 x 1.5 m)
    assert (car["means"].abs() <= torch.tensor([2.1, 0.95, 0.75]) + 1e-4).all()


def test_small_car_scene_and_object_metrics():
    scene = make_car_scene(num_frames=4, width=160, height=90, depth_samples=100)
    assert len(scene.cameras) == 12 and scene.sensor_names == ["CAM_FRONT", "CAM_LEFT", "CAM_BACK"]
    cam = scene.cameras[0]  # front camera, frame 0: the car is 9 m ahead in the left lane
    bb = object_bbox(cam, scene.tracks[0].poses[0], scene.tracks[0].size)
    assert bb is not None
    x0, y0, x1, y1 = bb
    crop = cam.image[y0:y1, x0:x1]
    blue = (crop[..., 2] > 0.5) & (crop[..., 0] < 0.3)
    assert blue.float().mean() > 0.1
    assert cam.lidar_uvz.shape == (100, 3) and (cam.lidar_uvz[:, 2] > 0).all()

    cfg = load_config("configs/synthetic_car.yaml", ["model.object_points_per_box=500"])
    graph = SceneGraph.from_scene(scene, cfg)
    views = [(c, (c.image * 255).round().to(torch.uint8)) for c in scene.cameras[:3]]
    m = evaluate_views(graph, views, "torch", "cpu", use_lpips=False)
    assert m["num_object_crops"] >= 1 and "obj_psnr" in m


def test_object_bbox_behind_camera_is_none():
    scene = make_car_scene(num_frames=1, width=32, height=18, depth_samples=10)
    back = scene.cameras[2]  # rear camera: car is ahead
    assert object_bbox(back, scene.tracks[0].poses[0], scene.tracks[0].size) is None
    far = make_transform(torch.eye(3), torch.tensor([500.0, 3.5, 0.75]))
    assert object_bbox(scene.cameras[0], far, scene.tracks[0].size) is None  # too small
