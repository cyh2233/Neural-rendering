import math

import torch

from nr.data.synthetic import make_synthetic_scene
from nr.scene.gaussians import GaussianModel
from nr.scene.graph import SceneGraph
from nr.scene.nodes import BackgroundNode, RigidObjectNode
from nr.utils.config import load_config
from nr.utils.geometry import make_transform, quat_to_rotmat, yaw_to_quat


def _object_node(pose_refine=True):
    g = GaussianModel.from_points(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), sh_degree=1)
    with torch.no_grad():
        g.quats.copy_(yaw_to_quat(torch.tensor([0.0, 0.3, -0.2])))
    rot = quat_to_rotmat(yaw_to_quat(torch.tensor(math.pi / 2)))
    poses = torch.stack([make_transform(rot, torch.tensor([10.0 + t, 5.0, 0.0])) for t in range(3)])
    valid = torch.tensor([True, True, False])
    return RigidObjectNode(g, poses, valid, torch.tensor([4.0, 2.0, 1.5]), pose_refine=pose_refine), poses


def test_object_node_world_transform():
    node, poses = _object_node()
    out = node.world_gaussians(1)
    local = node.gaussians.means.detach()
    expected = local @ poses[1, :3, :3].T + poses[1, :3, 3]
    assert torch.allclose(out["means"], expected, atol=1e-5)
    # rotations compose: R_world = R_pose @ R_local
    r_world = quat_to_rotmat(out["quats"])
    r_expected = poses[1, :3, :3] @ quat_to_rotmat(node.gaussians.quats.detach())
    assert torch.allclose(r_world, r_expected, atol=1e-5)


def test_pose_residual_is_in_box_frame():
    node, poses = _object_node()
    with torch.no_grad():
        node.delta_trans[0] = torch.tensor([1.0, 0.0, 0.0])  # 1 m forward in the box frame
    _, trans = node.pose(0)
    # the box is yawed by 90 deg, so box-forward is world +y
    assert torch.allclose(trans, poses[0, :3, 3] + torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_compose_includes_only_visible_objects():
    node, _ = _object_node()
    bg = BackgroundNode(GaussianModel.from_points(torch.randn(10, 3), sh_degree=1))
    graph = SceneGraph(bg, {"car": node})
    batch, slices = graph.compose(0, torch.zeros(3), sh_degree=1)
    assert slices == {"background": (0, 10), "car": (10, 13)}
    assert batch["means"].shape == (13, 3)
    batch, slices = graph.compose(2, torch.zeros(3), sh_degree=1)
    assert "car" not in slices and batch["means"].shape == (10, 3)


def test_state_dict_roundtrip_and_render():
    cfg = load_config("configs/synthetic.yaml")
    scene = make_synthetic_scene(num_frames=2, width=24, height=16, num_bg=50, num_obj=20)
    graph = SceneGraph.from_scene(scene, cfg)
    graph2 = SceneGraph.from_state_dict(graph.state_dict(), cfg)
    cam = scene.cameras[1]
    with torch.no_grad():
        a = graph.render(cam, backend="torch")
        b = graph2.render(cam, backend="torch")
    assert a["rgb"].shape == (16, 24, 3)
    assert torch.allclose(a["rgb"], b["rgb"])


def test_ply_roundtrip(tmp_path):
    g = GaussianModel.from_points(torch.randn(30, 3), torch.rand(30, 3), sh_degree=2)
    with torch.no_grad():
        g.shN.normal_()
        g.quats.normal_()
    path = tmp_path / "g.ply"
    g.save_ply(str(path))
    h = GaussianModel.load_ply(str(path))
    for name in ("means", "quats", "scales", "opacities", "sh0", "shN"):
        assert torch.allclose(getattr(g, name), getattr(h, name)), name
