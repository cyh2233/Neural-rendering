"""Build a SceneData from a config."""

from __future__ import annotations

from nr.data.types import SceneData


def load_scene_from_config(cfg) -> SceneData:
    src = cfg.data.source
    if src == "synthetic":
        from nr.data.synthetic import make_synthetic_scene

        s = cfg.get("synthetic", {})
        return make_synthetic_scene(
            num_frames=s.get("num_frames", 8),
            width=s.get("width", 64),
            height=s.get("height", 48),
            num_bg=s.get("num_bg", 400),
            num_obj=s.get("num_obj", 80),
            seed=cfg.seed,
        )
    if src == "synthetic_car":
        from nr.data.synthetic_car import make_car_scene

        s = cfg.get("synthetic", {})
        return make_car_scene(
            num_frames=s.get("num_frames", 16),
            width=s.get("width", 256),
            height=s.get("height", 144),
            seed=cfg.seed,
        )
    if src == "cache":
        from nr.data.cache import load_scene

        if not cfg.data.cache_path:
            raise ValueError("data.cache_path must be set when data.source == 'cache'")
        return load_scene(cfg.data.cache_path)
    raise ValueError(f"unknown data.source {src!r}")
