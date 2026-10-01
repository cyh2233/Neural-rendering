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
  data/     types.py (Camera, BoxTrack, SceneData), nuscenes.py, synthetic.py, synthetic_car.py, cache.py, loader.py
  scene/    gaussians.py, nodes.py (background / rigid object / sky), graph.py, appearance.py
  render/   backend.py (one rasterize() entry point), gsplat_backend.py, torch_backend.py
  train/    losses.py, strategy.py (clone / split / prune per node), trainer.py
  eval/     metrics.py (PSNR, SSIM, LPIPS), nvs.py (held-out eval, shifted trajectories), compare.py
  viz/      viewer.py (interactive viser viewer)
  utils/    geometry.py, sh.py, config.py, preflight.py (environment checks)
scripts/    run_nuscenes.py, prepare_nuscenes.py, train.py, render.py, export_ply.py, view.py, compare_car.py
configs/    base.yaml, synthetic.yaml, synthetic_car.yaml, nuscenes_mini.yaml
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
pip install -e ".[gpu,nuscenes,eval,video,viewer]"   # GPU machine: gsplat, nuScenes devkit, LPIPS, mp4, viser
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
It only checks that the pipeline runs; its "car" is a 20x8-pixel blob, so it says nothing about detail.

## Detailed car test on CPU

`configs/synthetic_car.yaml` builds a procedural car with glass and pillars, wheels with hubs, head and tail
lights, a grille and number plates, in a street with lane markings and building facades. The ego vehicle
overtakes it, so the front, left and rear cameras see the car's back, side and front. Images are 256x144.

```bash
python scripts/train.py --config configs/synthetic_car.yaml --out outputs/synthetic_car
python scripts/compare_car.py --ckpt outputs/synthetic_car/last.pt
```

`compare_car.py` prints whole-frame and object-region metrics (`obj_psnr`, `obj_ssim`) on held-out frames,
and writes image grids of ground truth vs render, a background-only render, and a laterally shifted view.
Object-region metrics matter because sky and road dominate whole-frame PSNR.

Held-out results of the default config (2,000 steps, about 20 minutes on 4 CPU cores):

| Step | PSNR | SSIM | Car-region PSNR | Car-region SSIM |
|---:|---:|---:|---:|---:|
| 500 | 26.0 | 0.878 | 22.5 | 0.846 |
| 1000 | 29.0 | 0.922 | 24.8 | 0.907 |
| 2000 | 30.8 | 0.940 | 26.4 | 0.931 |

Held-out frames have no images, so their object poses cannot be refined by training. They are interpolated
from the refined poses of neighbouring training frames, which removes most of the annotation noise. At
step 1000 this raised car-region PSNR from 18.7 dB (raw annotation) to 25.4 dB.

## First real run on a GPU server (nuScenes, one command)

`scripts/run_nuscenes.py` runs everything in order: environment checks, nuScenes conversion, training,
held-out evaluation, comparison images and a trajectory video, and writes a `report.md`.

1. **Install.** Install a PyTorch build that matches the server's CUDA first, then:

   ```bash
   pip install -e ".[gpu,nuscenes,eval,video,viewer]"
   ```

2. **Get the data.** Register at https://www.nuscenes.org and download `v1.0-mini` (about 4 GB).
   After unpacking, the folder must contain `v1.0-mini/`, `samples/` and `sweeps/`.

3. **Check the environment.** This takes seconds, except that the first gsplat call compiles CUDA kernels,
   which can take a few minutes once.

   ```bash
   python scripts/run_nuscenes.py --dataroot /data/nuscenes --check-only
   ```

   Every line must read `ok` or `warn`. A `FAIL` names what to fix.

4. **Run.** Start with a short run, then use the default 30,000 steps.

   ```bash
   python scripts/run_nuscenes.py --dataroot /data/nuscenes --scene scene-0061 --steps 7000
   python scripts/run_nuscenes.py --dataroot /data/nuscenes --scene scene-0061
   ```

5. **Look at the results** in `outputs/scene-0061/`:
   - `report.md`: environment, time per stage, held-out PSNR / SSIM / LPIPS and vehicle-region metrics
   - `compare_frames.png`: held-out views as ground truth, reconstruction, background only and a 1.5 m shifted view
   - `compare_crops.png`: enlarged vehicle crops, ground truth next to reconstruction
   - `front_trajectory.mp4`: front camera through the log, ground truth and reconstruction on top,
     1.5 m left and right shifts below

Nothing here has been run on real nuScenes data yet, so there are no reference timings or scores.
Please send back `report.md` and the two comparison images from the first run.

**If something goes wrong**
- **gsplat fails to compile.** The CUDA toolkit used for compilation must match PyTorch's CUDA version.
- **Out of GPU memory.** Use `--image-scale 0.25` or add `model.max_bg_points=200000`.
  Under 10 GB of VRAM the script already uses 0.25.
- **Any config value** can be appended as `key.sub=value`, for example `train.depth_lambda=0`.

## nuScenes, step by step (GPU)

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

## Interactive viewer

```bash
python scripts/view.py --ckpt outputs/scene-0061/last.pt --port 8080
# on a remote GPU machine: ssh -L 8080:localhost:8080 <host>, then open http://localhost:8080
```

The viewer renders the scene from your browser camera, so you can fly anywhere in the reconstruction.

- **Frame slider and Play.** Moves the dynamic objects through the log.
- **Recorded camera.** Pick a sensor, set a lateral or vertical offset, and press *Snap to camera*.
  With *Follow while playing* on, the view follows that sensor, which gives a live lane-change view.
- **Render mode.** RGB, depth (turbo colormap, near is warm), alpha, or background only.
- **Overlays.** Wireframe object boxes and the frustums of the recorded cameras.
- **Resolution.** Render width in pixels. The CPU rasterizer is slow, so the viewer starts at 160 px without gsplat.

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
- **CPU rasterizer.** The PyTorch reference renderer composites per 16x16 tile, using only the Gaussians whose
  3-sigma footprint overlaps the tile, as gsplat does. That is about 30 times faster than evaluating every
  Gaussian at every pixel, which is kept as `composite_dense` for testing.

## Limitations and next steps

- Only rigid vehicles are dynamic. Pedestrians and cyclists stay in the background.
- Only keyframes are used. The 12 Hz camera sweeps and LiDAR timing offsets are ignored.
- Nothing in this repo has been trained on real nuScenes data yet. The GPU path is written but untested here.
- Planned: Waymo and KITTI-360 loaders, a sky segmentation mask, and LiDAR rendering.
