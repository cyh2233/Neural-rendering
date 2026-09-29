"""Adaptive density control (clone / split / prune / opacity reset), aware of the scene graph.

The logic follows the original 3DGS ``DefaultStrategy`` (also implemented by gsplat), but runs
on each node's own :class:`GaussianModel` so dynamic objects keep their Gaussians in their box
frame. It works with both rasterizer backends because it only needs ``info["means2d"].grad``
and ``info["radii"]``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from nr.scene.gaussians import PARAM_NAMES, GaussianModel
from nr.utils.geometry import quat_to_rotmat


def _update_param(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    param_fn: Callable[[str, torch.Tensor], torch.Tensor],
    state_fn: Callable[[str, torch.Tensor], torch.Tensor],
) -> None:
    """Replace every Gaussian parameter of ``model`` and keep Adam state consistent."""
    for name in PARAM_NAMES:
        old = getattr(model, name)
        new = nn.Parameter(param_fn(name, old.data), requires_grad=old.requires_grad)
        for group in optimizer.param_groups:
            for i, p in enumerate(group["params"]):
                if p is old:
                    group["params"][i] = new
        if old in optimizer.state:
            st = optimizer.state.pop(old)
            for k, v in st.items():
                if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == old.shape[0]:
                    st[k] = state_fn(k, v)
            optimizer.state[new] = st
        setattr(model, name, new)


@torch.no_grad()
def duplicate(model: GaussianModel, optimizer, mask: torch.Tensor) -> None:
    idx = mask.nonzero().squeeze(1)
    _update_param(
        model, optimizer,
        lambda _, p: torch.cat([p, p[idx]]),
        lambda _, v: torch.cat([v, torch.zeros_like(v[idx])]),
    )


@torch.no_grad()
def split(model: GaussianModel, optimizer, mask: torch.Tensor, n_split: int = 2) -> None:
    sel = mask.nonzero().squeeze(1)
    keep = (~mask).nonzero().squeeze(1)
    scales = torch.exp(model.scales[sel])
    rot = quat_to_rotmat(model.quats[sel])
    samples = torch.randn(n_split, sel.numel(), 3, device=scales.device) * scales
    new_means = (torch.einsum("nij,snj->sni", rot, samples) + model.means[sel]).reshape(-1, 3)

    def pfn(name, p):
        if name == "means":
            new = new_means
        elif name == "scales":
            new = torch.log(scales / (0.8 * n_split)).repeat(n_split, 1)
        else:
            new = p[sel].repeat(n_split, *([1] * (p.dim() - 1)))
        return torch.cat([p[keep], new])

    def sfn(_, v):
        return torch.cat([v[keep], torch.zeros((n_split * sel.numel(),) + v.shape[1:], device=v.device, dtype=v.dtype)])

    _update_param(model, optimizer, pfn, sfn)


@torch.no_grad()
def remove(model: GaussianModel, optimizer, mask: torch.Tensor) -> None:
    keep = (~mask).nonzero().squeeze(1)
    _update_param(model, optimizer, lambda _, p: p[keep], lambda _, v: v[keep])


@torch.no_grad()
def reset_opacity(model: GaussianModel, optimizer, value: float) -> None:
    logit = float(torch.logit(torch.tensor(value)))

    def pfn(name, p):
        return p.clamp(max=logit) if name == "opacities" else p

    def sfn(_, v):
        return v  # only zero the opacity state below

    _update_param(model, optimizer, pfn, sfn)
    st = optimizer.state.get(model.opacities, {})
    for k, v in st.items():
        if torch.is_tensor(v) and v.dim() > 0:
            st[k] = torch.zeros_like(v)


class DensifyStrategy:
    def __init__(self, cfg, scene_scale: float):
        self.cfg = cfg
        self.scene_scale = scene_scale
        self.stats: dict[str, dict[str, torch.Tensor]] = {}

    def _node_scale(self, graph, name: str) -> float:
        if name == "background":
            return self.scene_scale
        return float(graph.objects[name].size.max())

    def _stats(self, name: str, model: GaussianModel) -> dict[str, torch.Tensor]:
        st = self.stats.get(name)
        if st is None or st["grad2d"].shape[0] != len(model):
            dev = model.means.device
            st = {"grad2d": torch.zeros(len(model), device=dev), "count": torch.zeros(len(model), device=dev)}
            self.stats[name] = st
        return st

    @torch.no_grad()
    def update_stats(self, graph, out: dict) -> None:
        info = out["info"]
        grad = info["means2d"].grad
        if grad is None:
            return
        if grad.dim() == 3:
            grad = grad[0]
        grad = grad.clone()
        grad[:, 0] *= info["width"] / 2.0
        grad[:, 1] *= info["height"] / 2.0
        norm = grad.norm(dim=-1)
        visible = info["radii"] > 0
        nodes = graph.gaussian_nodes()
        for name, (s, e) in out["slices"].items():
            st = self._stats(name, nodes[name])
            vis = visible[s:e]
            st["grad2d"][vis] += norm[s:e][vis]
            st["count"][vis] += 1

    @torch.no_grad()
    def step(self, graph, optimizer, out: dict, step: int) -> dict[str, int]:
        c = self.cfg
        report = {}
        if not c.enabled or step >= c.refine_stop:
            return report
        self.update_stats(graph, out)
        if step > c.refine_start and step % c.refine_every == 0:
            for name, model in graph.gaussian_nodes().items():
                report[name] = self._refine(graph, name, model, optimizer, step)
        if step > 0 and step % c.reset_every == 0:
            for model in graph.gaussian_nodes().values():
                reset_opacity(model, optimizer, c.prune_opa * 2.0)
        return report

    def _refine(self, graph, name, model, optimizer, step) -> int:
        c = self.cfg
        st = self._stats(name, model)
        scale = self._node_scale(graph, name)
        avg = st["grad2d"] / st["count"].clamp(min=1)
        high = avg > c.grow_grad2d
        small = torch.exp(model.scales).max(-1).values <= c.grow_scale3d * scale
        dup_mask = high & small
        split_mask = high & ~small
        n0 = len(model)
        if dup_mask.any():
            duplicate(model, optimizer, dup_mask)
            split_mask = torch.cat([split_mask, torch.zeros(len(model) - n0, dtype=torch.bool, device=split_mask.device)])
        if split_mask.any():
            split(model, optimizer, split_mask)
        opa = torch.sigmoid(model.opacities)
        prune = opa < c.prune_opa
        if step > c.reset_every:
            prune |= torch.exp(model.scales).max(-1).values > c.prune_scale3d * scale
        if prune.all():  # never delete a whole node
            prune[opa.argmax()] = False
        if prune.any():
            remove(model, optimizer, prune)
        self.stats.pop(name, None)
        return len(model) - n0
