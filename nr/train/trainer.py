"""Training loop for the scene graph."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from nr.data.cache import load_image
from nr.data.loader import load_scene_from_config
from nr.data.types import Camera, SceneData
from nr.render.backend import resolve_backend
from nr.scene.graph import SceneGraph
from nr.train.losses import lidar_depth_loss, opacity_regularizer, photometric_loss
from nr.train.strategy import DensifyStrategy


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_scene_scale(cameras: list[Camera]) -> float:
    centers = torch.stack([c.c2w[:3, 3] for c in cameras])
    return max(1.0, float((centers - centers.mean(0)).norm(dim=-1).max()) * 1.1)


def prepare_views(cams: list[Camera], scale: float) -> list[tuple[Camera, torch.Tensor]]:
    """Scale cameras and load their images once (kept as uint8 on CPU to save memory)."""
    views = []
    for c in cams:
        img = load_image(c, scale)
        views.append((c.scaled(scale), (img * 255).round().to(torch.uint8)))
    return views


def build_optimizer(graph: SceneGraph, cfg, scene_scale: float) -> torch.optim.Optimizer:
    lr = cfg.train.lr
    groups = []
    for node_name, g in graph.gaussian_nodes().items():
        for pname, p in g.params().items():
            base = lr[pname] * (scene_scale if pname == "means" and node_name == "background" else 1.0)
            groups.append({"params": [p], "lr": base, "name": f"{node_name}.{pname}", "base_lr": base})
    pose_params = [p for n in graph.objects.values() for p in (n.delta_trans, n.delta_yaw) if p.requires_grad]
    if pose_params:
        groups.append({"params": pose_params, "lr": lr.pose, "name": "pose", "base_lr": lr.pose})
    if graph.sky is not None:
        groups.append({"params": list(graph.sky.parameters()), "lr": lr.sky, "name": "sky", "base_lr": lr.sky})
    if graph.appearance is not None:
        groups.append({"params": list(graph.appearance.parameters()), "lr": lr.appearance,
                       "name": "appearance", "base_lr": lr.appearance})
    return torch.optim.Adam(groups, eps=1e-15)


class Trainer:
    def __init__(self, cfg, scene: SceneData | None = None, out_dir: str | Path | None = None):
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.device = resolve_device(cfg.device)
        self.backend = resolve_backend(cfg.backend)
        self.scene = scene if scene is not None else load_scene_from_config(cfg)
        train_cams, test_cams = self.scene.split(cfg.data.holdout_every)
        s = cfg.data.image_scale
        self.train_views = prepare_views(train_cams, s)
        self.test_views = prepare_views(test_cams, s)
        self.scene_scale = compute_scene_scale(train_cams)
        self.graph = SceneGraph.from_scene(self.scene, cfg).to(self.device)
        self.optimizer = build_optimizer(self.graph, cfg, self.scene_scale)
        self.strategy = DensifyStrategy(cfg.strategy, self.scene_scale)
        self.out_dir = Path(out_dir or Path(cfg.output_dir) / self.scene.name)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.history: list[dict] = []
        self.final_metrics: dict = {}
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(str(self.out_dir / "tb"))
        except Exception:  # tensorboard not installed
            self._writer = None

    # ------------------------------------------------------------------ helpers
    def active_sh_degree(self) -> int:
        every = self.cfg.train.sh_increase_every
        return min(self.cfg.model.sh_degree, self.step // every if every > 0 else self.cfg.model.sh_degree)

    def _update_lr(self) -> None:
        t = self.cfg.train
        ratio = t.lr.means_final_ratio ** min(1.0, self.step / max(1, t.max_steps))
        for g in self.optimizer.param_groups:
            if g["name"].endswith(".means"):
                g["lr"] = g["base_lr"] * ratio

    # ------------------------------------------------------------------ one step
    def train_step(self) -> dict:
        t = self.cfg.train
        cam, img_u8 = random.choice(self.train_views)
        gt = img_u8.to(self.device).float() / 255.0
        out = self.graph.render(cam, self.active_sh_degree(), self.backend)
        pred = out["rgb"]
        loss_rgb = photometric_loss(pred, gt, t.ssim_lambda)
        loss = loss_rgb
        logs = {"loss_rgb": loss_rgb.item()}
        if t.depth_lambda > 0 and cam.lidar_uvz is not None:
            ld = lidar_depth_loss(out["depth"], out["alpha"], cam.lidar_uvz)
            loss = loss + t.depth_lambda * ld
            logs["loss_depth"] = ld.item()
        if t.opacity_reg > 0:
            opa = torch.cat([torch.sigmoid(g.opacities) for g in self.graph.gaussian_nodes().values()])
            loss = loss + t.opacity_reg * opacity_regularizer(opa)
        if self.graph.appearance is not None:
            loss = loss + 1e-3 * self.graph.appearance.regularizer()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.strategy.step(self.graph, self.optimizer, out, self.step)
        self.optimizer.step()
        self._update_lr()
        logs["loss"] = loss.item()
        with torch.no_grad():
            logs["psnr"] = float(-10 * torch.log10(((pred.clamp(0, 1) - gt) ** 2).mean()))
        return logs

    # ------------------------------------------------------------------ loop
    def train(self, max_steps: int | None = None, verbose: bool = True) -> list[dict]:
        t = self.cfg.train
        max_steps = max_steps or t.max_steps
        t0 = time.time()
        running: dict[str, float] = {}
        n_run = 0
        while self.step < max_steps:
            logs = self.train_step()
            self.step += 1
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            n_run += 1
            if self.step % t.log_every == 0 or self.step == max_steps:
                avg = {k: v / n_run for k, v in running.items()}
                avg.update(step=self.step, num_gaussians=self.graph.num_gaussians(), elapsed=time.time() - t0)
                self.history.append(avg)
                self._log(avg, "train")
                if verbose:
                    print(
                        f"[{self.step:6d}] loss {avg['loss']:.4f} psnr {avg['psnr']:.2f} "
                        f"#G {avg['num_gaussians']} ({avg['elapsed']:.0f}s)",
                        flush=True,
                    )
                running, n_run = {}, 0
            if t.eval_every and self.step % t.eval_every == 0 and self.test_views:
                self._log(self.evaluate(), "test", verbose)
            if t.save_every and self.step % t.save_every == 0:
                self.save()
        self.save()
        if self.test_views:
            metrics = self.evaluate()
            self._log(metrics, "test", verbose)
            (self.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            self.final_metrics = metrics
        return self.history

    def _log(self, d: dict, prefix: str, verbose: bool = False) -> None:
        if self._writer is not None:
            for k, v in d.items():
                if isinstance(v, (int, float)) and k != "step":
                    self._writer.add_scalar(f"{prefix}/{k}", v, self.step)
        with open(self.out_dir / "log.jsonl", "a") as f:
            f.write(json.dumps({"split": prefix, "step": self.step, **d}) + "\n")
        if verbose and prefix == "test":
            print("  test: " + " ".join(f"{k} {v:.4f}" for k, v in d.items() if isinstance(v, float)), flush=True)

    def fill_untrained_poses(self) -> None:
        """Held-out frames have no images, so their object poses are never refined. Interpolate them
        from the refined poses of the neighbouring training frames instead of keeping the raw annotation."""
        trained = torch.zeros(self.scene.num_frames, dtype=torch.bool)
        for cam, _ in self.train_views:
            trained[cam.frame_idx] = True
        for node in self.graph.objects.values():
            node.interpolate_untrained(trained)

    @torch.no_grad()
    def evaluate(self) -> dict:
        from nr.eval.nvs import evaluate_views

        self.fill_untrained_poses()
        return evaluate_views(self.graph, self.test_views, self.backend, self.device)

    def save(self, name: str = "last.pt") -> Path:
        self.fill_untrained_poses()
        path = self.out_dir / name
        torch.save(
            {
                "step": self.step,
                "config": self.cfg.to_dict(),
                "state_dict": self.graph.state_dict(),
                "scene_scale": self.scene_scale,
                "scene_name": self.scene.name,
            },
            path,
        )
        return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    from nr.utils.config import Config

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config.wrap(ckpt["config"])
    graph = SceneGraph.from_state_dict(ckpt["state_dict"], cfg).to(device)
    return graph, cfg, ckpt

