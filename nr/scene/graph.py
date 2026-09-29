"""Scene graph: compose static background + dynamic objects + sky for one frame and render it."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nr.data.types import Camera, SceneData
from nr.render.backend import rasterize
from nr.scene.appearance import AppearanceModel
from nr.scene.gaussians import GaussianModel
from nr.scene.nodes import BackgroundNode, RigidObjectNode, SkyNode
from nr.utils.sh import eval_sh


def camera_ray_dirs(cam: Camera, device) -> torch.Tensor:
    """(H, W, 3) unit world-space ray directions through pixel centres."""
    ys, xs = torch.meshgrid(
        torch.arange(cam.height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(cam.width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    K = cam.K.to(device)
    d = torch.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], torch.ones_like(xs)], -1)
    d = d @ cam.c2w[:3, :3].to(device).T
    return F.normalize(d, dim=-1)


class SceneGraph(nn.Module):
    def __init__(
        self,
        background: BackgroundNode,
        objects: dict[str, RigidObjectNode] | None = None,
        sky: SkyNode | None = None,
        appearance: AppearanceModel | None = None,
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        super().__init__()
        self.background = background
        self.objects = nn.ModuleDict(objects or {})
        self.sky = sky
        self.appearance = appearance
        self.register_buffer("bg_color", torch.tensor(bg_color, dtype=torch.float32))

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_scene(cls, scene: SceneData, cfg) -> SceneGraph:
        m = cfg.model
        pts, cols = scene.points, scene.colors
        if pts.shape[0] > m.max_bg_points:
            keep = torch.randperm(pts.shape[0])[: m.max_bg_points]
            pts = pts[keep]
            cols = cols[keep] if cols is not None else None
        bg = BackgroundNode(
            GaussianModel.from_points(pts, cols, m.sh_degree, m.init_opacity, m.init_scale_knn)
        )
        objects = {}
        for i, tr in enumerate(scene.tracks):
            obj_pts, obj_cols = _object_init_points(tr, m.object_points_per_box)
            g = GaussianModel.from_points(
                obj_pts, obj_cols, m.sh_degree, m.init_opacity, m.init_scale_knn,
                max_init_scale=float(tr.size.min()) * 0.25,
            )
            objects[f"obj_{i:03d}"] = RigidObjectNode(
                g, tr.poses, tr.valid, tr.size, tr.instance_id, tr.category, pose_refine=m.pose_refine
            )
        sky = SkyNode(m.sky_hidden) if m.use_sky else None
        app = AppearanceModel(scene.num_sensors) if m.use_appearance else None
        return cls(bg, objects, sky, app)

    @classmethod
    def from_state_dict(cls, state: dict, cfg) -> SceneGraph:
        """Rebuild an empty graph with the right shapes, then load ``state``."""
        m = cfg.model
        bg = BackgroundNode(GaussianModel.empty_like_state(state, "background.gaussians."))
        objects = {}
        names = sorted({k.split(".")[1] for k in state if k.startswith("objects.")})
        for name in names:
            p = f"objects.{name}."
            g = GaussianModel.empty_like_state(state, p + "gaussians.")
            objects[name] = RigidObjectNode(
                g, state[p + "base_poses"], state[p + "valid"], state[p + "size"], pose_refine=m.pose_refine
            )
        sky = SkyNode(m.sky_hidden) if any(k.startswith("sky.") for k in state) else None
        app = None
        if "appearance.affine" in state:
            app = AppearanceModel(state["appearance.affine"].shape[0])
        graph = cls(bg, objects, sky, app)
        graph.load_state_dict(state)
        return graph

    # ------------------------------------------------------------------ nodes
    def gaussian_nodes(self) -> dict[str, GaussianModel]:
        out = {"background": self.background.gaussians}
        for name, node in self.objects.items():
            out[name] = node.gaussians
        return out

    def num_gaussians(self) -> int:
        return sum(len(g) for g in self.gaussian_nodes().values())

    # ------------------------------------------------------------------ composition
    def compose(self, frame_idx: int, cam_center: torch.Tensor, sh_degree: int, include_objects: bool = True):
        """Collect all Gaussians visible at ``frame_idx`` in world space, with colours evaluated
        for a camera at ``cam_center``. Returns (batch dict, slices {node_name: (start, end)})."""
        parts = [("background", self.background.world_gaussians(frame_idx), None)]
        for name, node in self.objects.items():
            if include_objects and node.is_visible(frame_idx):
                g = node.world_gaussians(frame_idx)
                parts.append((name, g, g.pop("rot")))

        means, quats, scales, opac, colors = [], [], [], [], []
        slices, start = {}, 0
        for name, g, rot in parts:
            dirs = F.normalize(g["means"] - cam_center, dim=-1)
            if rot is not None:
                dirs = dirs @ rot  # world -> box frame, so object appearance moves with the object
            rgb = eval_sh(sh_degree, g["sh"], dirs) + 0.5
            means.append(g["means"])
            quats.append(g["quats"])
            scales.append(g["scales"])
            opac.append(g["opacities"])
            colors.append(rgb.clamp(min=0.0))
            n = g["means"].shape[0]
            slices[name] = (start, start + n)
            start += n
        batch = {
            "means": torch.cat(means),
            "quats": torch.cat(quats),
            "scales": torch.cat(scales),
            "opacities": torch.cat(opac),
            "colors": torch.cat(colors),
        }
        return batch, slices

    def render(
        self, cam: Camera, sh_degree: int | None = None, backend: str = "auto", include_objects: bool = True
    ) -> dict:
        device = self.bg_color.device
        sh_degree = self.background.gaussians.sh_degree if sh_degree is None else sh_degree
        c2w = cam.c2w.to(device)
        K = cam.K.to(device)
        batch, slices = self.compose(cam.frame_idx, c2w[:3, 3], sh_degree, include_objects)
        viewmat = torch.linalg.inv(c2w)
        rgb, alpha, depth, info = rasterize(
            batch["means"], batch["quats"], batch["scales"], batch["opacities"], batch["colors"],
            viewmat, K, cam.width, cam.height, backend=backend,
        )
        if self.sky is not None:
            sky_rgb = self.sky(camera_ray_dirs(cam, device))
        else:
            sky_rgb = self.bg_color.expand(cam.height, cam.width, 3)
        img = rgb + (1.0 - alpha) * sky_rgb
        if self.appearance is not None:
            img = self.appearance(img, cam.cam_id)
        return {
            "rgb": img,
            "rgb_raw": rgb,
            "alpha": alpha,
            "depth": depth,
            "sky": sky_rgb,
            "info": info,
            "slices": slices,
        }


def _object_init_points(track, n_target: int):
    """LiDAR points inside the box (box frame) topped up with uniform samples on the box surface."""
    pts = track.points if track.points is not None else torch.zeros(0, 3)
    cols = track.colors if track.colors is not None else torch.full((pts.shape[0], 3), 0.5)
    n_fill = max(0, min(n_target, 2000) - pts.shape[0]) if pts.shape[0] < n_target else 0
    if n_fill > 0:
        half = track.size.float() / 2
        u = torch.rand(n_fill, 3) * 2 - 1
        axis = torch.randint(0, 3, (n_fill,))
        u[torch.arange(n_fill), axis] = torch.sign(u[torch.arange(n_fill), axis])  # push to a face
        pts = torch.cat([pts.float(), u * half])
        cols = torch.cat([cols.float(), torch.full((n_fill, 3), 0.5)])
    if pts.shape[0] > n_target:
        keep = torch.randperm(pts.shape[0])[:n_target]
        pts, cols = pts[keep], cols[keep]
    return pts, cols
