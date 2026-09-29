# Neural Rendering for Driving Logs

3D Gaussian Splatting (3DGS) reconstruction and novel view synthesis for autonomous-driving logs.
The scene is a graph: a static background, rigid dynamic objects that move with their 3D boxes,
and a sky model. A trained scene can be re-rendered from new viewpoints, such as a lane change.

The CUDA rasterizer comes from [gsplat](https://github.com/nerfstudio-project/gsplat).
Everything else is implemented here: the data conversion, scene graph, training loop,
densification and evaluation. A pure-PyTorch reference rasterizer with the same interface
runs on CPU, so the whole pipeline can be tested and read without a GPU.

## Layout

```
nr/
  data/     types.py (Camera, BoxTrack, SceneData), nuscenes.py, synthetic.py, cache.py, loader.py
  scene/    gaussians.py, nodes.py (background / rigid object / sky), graph.py, appearance.py
  render/   backend.py (one rasterize() entry point), gsplat_backend.py, torch_backend.py
  train/    losses.py, strategy.py (clone / split / prune per node), trainer.py
  eval/     metrics.py (PSNR, SSIM, LPIPS), nvs.py (held-out eval, shifted trajectories)
  utils/    geometry.py, sh.py, config.py
scripts/    prepare_nuscenes.py, train.py, render.py, export_ply.py
configs/    base.yaml, synthetic.yaml, nuscenes_mini.yaml
tests/
```

## Conventions

- World frame is z-up with the origin at the first ego position of the log.
- Cameras use the OpenCV convention: +x right, +y down, +z forward. Poses are 4x4 camera-to-world.
- Box frame is +x forward (length), +y left (width), +z up, origin at the box centre.
- Quaternions are (w, x, y, z).

## Install

```bash
pip install -e ".[dev]"                          # CPU: reference rasterizer, tests
pip install -e ".[gpu,nuscenes,eval,video]"      # GPU machine: gsplat, nuScenes devkit, LPIPS, mp4
```

gsplat compiles its CUDA kernels on first use, so install a PyTorch build that matches your CUDA toolkit first.

## Quick start on CPU (synthetic scene)

```bash
pytest -q
python scripts/train.py --config configs/synthetic.yaml --out outputs/synthetic
python scripts/render.py --ckpt outputs/synthetic/last.pt --mode heldout
python scripts/render.py --ckpt outputs/synthetic/last.pt --mode shift --lateral -2.0 --out outputs/synthetic/shift
```

The synthetic scene is a 64x48 road with a car overtaking the ego vehicle. 300 steps take about 30 s on CPU.

## nuScenes (GPU)

1. Download `v1.0-mini` from https://www.nuscenes.org/nuscenes#download and unpack it to `/data/nuscenes`.
2. Convert one scene. This reads the six cameras and LIDAR_TOP at the 2 Hz keyframes. It splits LiDAR into
   background and moving vehicles, colours points from the images, and writes `data/cache/<scene>.pt`.

   ```bash
   python scripts/prepare_nuscenes.py --dataroot /data/nuscenes --version v1.0-mini --scene scene-0061
   ```

3. Train, evaluate and render:

   ```bash
   python scripts/train.py --config configs/nuscenes_mini.yaml --scene scene-0061
   python scripts/render.py --ckpt outputs/scene-0061/last.pt --mode heldout
   python scripts/render.py --ckpt outputs/scene-0061/last.pt --mode shift --camera 0 --lateral 1.5 --out shift.mp4
   python scripts/export_ply.py --ckpt outputs/scene-0061/last.pt --frame 10
   pytest -q -m gpu          # checks that gsplat and the torch rasterizer agree
   ```

Any config value can be overridden on the command line, for example `train.max_steps=7000 data.image_scale=0.25`.

## How it works

- **Initialisation.** Background Gaussians start from accumulated LiDAR with points inside moving boxes removed.
  Each moving vehicle starts from the LiDAR points inside its box, in the box frame, topped up with samples
  on the box surface.
- **Composition.** For a frame, every visible object's Gaussians are moved by its box pose and concatenated with
  the background into one batch. The rasterizer never sees the graph. Object colour is evaluated with the view
  direction in the box frame, so appearance moves with the car.
- **Pose refinement.** Each object has a learnable per-frame translation and yaw residual on top of the annotated box.
- **Sky and exposure.** A small MLP maps ray direction to sky colour behind the Gaussians. A per-camera
  affine colour transform absorbs exposure differences between the six cameras.
- **Loss.** `0.8 * L1 + 0.2 * (1 - SSIM)`, plus relative L1 between rendered depth and projected LiDAR.
- **Densification.** 3DGS-style clone, split, prune and opacity reset, run separately for each node.
  It works with both rasterizers.

## Limitations and next steps

- Only rigid vehicles are dynamic. Pedestrians and cyclists stay in the background.
- Only keyframes are used. The 12 Hz camera sweeps and LiDAR timing offsets are ignored.
- Nothing in this repo has been trained on real nuScenes data yet. The GPU path is written but untested here.
- Planned: Waymo and KITTI-360 loaders, a sky segmentation mask, LiDAR rendering, and an interactive viser viewer.
