"""Interactive viewer for a trained scene graph, built on viser.

Open the printed URL in a browser. The viewer renders the scene from the browser camera and
streams the image back as the scene background, so you can fly anywhere. It also provides:

  * a frame slider (dynamic objects move with it) and play / pause,
  * snapping to any recorded camera, plus a lateral / vertical offset for lane-change views,
  * RGB, depth, alpha or background-only render modes,
  * wireframes of the object boxes and frustums of the recorded cameras.

viser's camera uses the OpenCV convention (+z forward, -y up), the same as ``Camera``.
The GUI-free helpers at the top of this module are unit-tested without a browser.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from nr.data.types import Camera, SceneData
from nr.utils.geometry import lateral_shift, quat_to_rotmat, rotmat_to_quat

RENDER_MODES = ("rgb", "depth", "alpha", "background only")

# Anchor colours of the "turbo" colormap, enough for a smooth depth visualisation.
_TURBO = np.array(
    [
        [0.19, 0.07, 0.23], [0.27, 0.42, 0.93], [0.13, 0.75, 0.87], [0.30, 0.97, 0.52],
        [0.73, 0.97, 0.21], [0.99, 0.72, 0.21], [0.95, 0.35, 0.09], [0.48, 0.02, 0.01],
    ]
)


# ----------------------------------------------------------------------------- pure helpers


def viser_to_camera(
    wxyz, position, fov_y: float, aspect: float, width: int, frame_idx: int, cam_id: int = 0
) -> Camera:
    """Browser camera (OpenCV pose, vertical FOV in radians) -> ``Camera`` at ``width`` pixels."""
    width = max(8, int(width))
    height = max(8, int(round(width / aspect)))
    fy = 0.5 * height / np.tan(0.5 * fov_y)
    K = torch.tensor([[fy, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]], dtype=torch.float32)
    c2w = torch.eye(4)
    c2w[:3, :3] = quat_to_rotmat(torch.as_tensor(np.asarray(wxyz), dtype=torch.float64)).float()
    c2w[:3, 3] = torch.as_tensor(np.asarray(position), dtype=torch.float32)
    return Camera(K=K, c2w=c2w, width=width, height=height, cam_id=cam_id, frame_idx=frame_idx, name="viewer")


def camera_to_viser(cam: Camera) -> tuple[np.ndarray, np.ndarray, float]:
    """``Camera`` -> (wxyz, position, vertical FOV in radians) for viser."""
    wxyz = rotmat_to_quat(cam.c2w[:3, :3].double()).numpy()
    fov_y = 2.0 * float(np.arctan(0.5 * cam.height / float(cam.K[1, 1])))
    return wxyz, cam.c2w[:3, 3].double().numpy(), fov_y


def colorize_depth(depth: np.ndarray, alpha: np.ndarray, near: float | None = None, far: float | None = None) -> np.ndarray:
    """(H, W) metric depth -> (H, W, 3) uint8 turbo image; inverse-depth scaling, black where empty."""
    valid = alpha > 0.5
    if not valid.any():
        return np.zeros(depth.shape + (3,), np.uint8)
    d = depth[valid]
    near = float(np.percentile(d, 2)) if near is None else near
    far = float(np.percentile(d, 98)) if far is None else far
    inv = 1.0 / np.clip(depth, 1e-3, None)
    t = (inv - 1.0 / max(far, near + 1e-3)) / (1.0 / max(near, 1e-3) - 1.0 / max(far, near + 1e-3))
    t = np.clip(t, 0.0, 1.0)  # near = 1 (warm), far = 0 (cool)
    x = t * (len(_TURBO) - 1)
    i = np.clip(x.astype(int), 0, len(_TURBO) - 2)
    f = (x - i)[..., None]
    rgb = _TURBO[i] * (1 - f) + _TURBO[i + 1] * f
    rgb[~valid] = 0.0
    return (rgb * 255).astype(np.uint8)


@dataclass
class ViewState:
    frame: int = 0
    mode: str = "rgb"
    width: int = 640
    show_objects: bool = True


@torch.no_grad()
def render_view(graph, cam: Camera, state: ViewState, backend: str) -> np.ndarray:
    """Render ``cam`` according to ``state`` and return an (H, W, 3) uint8 image."""
    out = graph.render(cam, backend=backend, include_objects=state.show_objects and state.mode != "background only")
    if state.mode == "depth":
        return colorize_depth(out["depth"][..., 0].cpu().numpy(), out["alpha"][..., 0].cpu().numpy())
    if state.mode == "alpha":
        a = out["alpha"].clamp(0, 1).cpu().numpy()
        return (np.repeat(a, 3, axis=-1) * 255).astype(np.uint8)
    return (out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


# ----------------------------------------------------------------------------- viewer


class SceneViewer:
    def __init__(
        self,
        graph,
        scene: SceneData,
        backend: str = "auto",
        host: str = "0.0.0.0",
        port: int = 8080,
        width: int = 640,
    ):
        import viser

        self.graph = graph.eval()
        self.scene = scene
        self.backend = backend
        self.state = ViewState(width=width)
        self._lock = threading.Lock()  # one render at a time (GPU memory, torch state)
        self._dirty: dict[int, bool] = {}
        self._offset = (0.0, 0.0)
        self.server = viser.ViserServer(host=host, port=port, verbose=False)
        self.server.scene.set_up_direction("+z")
        self._build_gui()
        self._update_scene_overlays()
        self.server.on_client_connect(self._on_connect)
        self.server.on_client_disconnect(lambda c: self._dirty.pop(c.client_id, None))

    # ------------------------------------------------------------------ GUI
    def _build_gui(self) -> None:
        gui = self.server.gui
        n_frames = max(1, self.scene.num_frames)
        self.gui_frame = gui.add_slider("Frame", 0, n_frames - 1, 1, 0)
        self.gui_play = gui.add_checkbox("Play", False)
        self.gui_fps = gui.add_slider("Play FPS", 1, 10, 1, 2)
        self.gui_mode = gui.add_dropdown("Render mode", RENDER_MODES, initial_value="rgb")
        self.gui_width = gui.add_slider("Resolution (px)", 32, 1920, 32, min(1920, max(32, self.state.width)))
        self.gui_objects = gui.add_checkbox("Render objects", True)
        self.gui_boxes = gui.add_checkbox("Show boxes", True)
        self.gui_frustums = gui.add_checkbox("Show cameras", False)

        with gui.add_folder("Recorded camera"):
            names = self.scene.sensor_names or [str(i) for i in range(self.scene.num_sensors)]
            self.gui_sensor = gui.add_dropdown("Sensor", names, initial_value=names[0])
            self.gui_lateral = gui.add_slider("Lateral offset (m)", -5.0, 5.0, 0.1, 0.0)
            self.gui_up = gui.add_slider("Vertical offset (m)", -2.0, 5.0, 0.1, 0.0)
            self.gui_snap = gui.add_button("Snap to camera")
            self.gui_follow = gui.add_checkbox("Follow while playing", False)
        self.gui_info = gui.add_markdown("")

        for h in (self.gui_mode, self.gui_width, self.gui_objects):
            h.on_update(lambda _: self._sync_state())
        for h in (self.gui_boxes, self.gui_frustums):
            h.on_update(lambda _: self._update_scene_overlays())
        self.gui_frame.on_update(lambda _: self._on_frame_change())
        self.gui_snap.on_click(lambda _: self.snap_all())
        self.gui_play.on_update(lambda _: self._start_player() if self.gui_play.value else None)

    def _sync_state(self) -> None:
        self.state.mode = self.gui_mode.value
        self.state.width = int(self.gui_width.value)
        self.state.show_objects = bool(self.gui_objects.value)
        self.state.frame = int(self.gui_frame.value)
        self.mark_dirty()

    def _on_frame_change(self) -> None:
        self._sync_state()
        self._update_scene_overlays()
        if self.gui_play.value and self.gui_follow.value:
            self.snap_all()

    def _start_player(self) -> None:
        def loop():
            while self.gui_play.value:
                self.gui_frame.value = (int(self.gui_frame.value) + 1) % max(1, self.scene.num_frames)
                time.sleep(1.0 / float(self.gui_fps.value))

        threading.Thread(target=loop, daemon=True).start()

    # ------------------------------------------------------------------ scene overlays
    def recorded_camera(self, frame: int, sensor_name: str) -> Camera | None:
        names = self.scene.sensor_names
        cam_id = names.index(sensor_name) if sensor_name in names else int(sensor_name)
        for c in self.scene.cameras:
            if c.frame_idx == frame and c.cam_id == cam_id:
                return c
        return None

    def _update_scene_overlays(self) -> None:
        scene_api = self.server.scene
        frame = int(self.gui_frame.value)
        for name, node in self.graph.objects.items():
            path = f"/boxes/{name}"
            if self.gui_boxes.value and node.is_visible(frame):
                with torch.no_grad():
                    rot, trans = node.pose(frame)
                wxyz = rotmat_to_quat(rot.detach().cpu().double()).numpy()
                scene_api.add_box(
                    path, color=(255, 170, 0), dimensions=tuple(node.size.cpu().tolist()),
                    wireframe=True, wxyz=wxyz, position=trans.detach().cpu().numpy(),
                )
            else:
                scene_api.add_box(path, dimensions=(1e-3, 1e-3, 1e-3), visible=False)
        for c in self.scene.cameras:
            path = f"/cameras/{c.cam_id}"
            if c.frame_idx != frame:
                continue
            wxyz, pos, fov = camera_to_viser(c)
            scene_api.add_camera_frustum(
                path, fov=fov, aspect=c.width / c.height, scale=0.5, color=(60, 120, 255),
                wxyz=wxyz, position=pos, visible=bool(self.gui_frustums.value),
            )

    # ------------------------------------------------------------------ cameras
    def snap_all(self) -> None:
        cam = self.recorded_camera(int(self.gui_frame.value), self.gui_sensor.value)
        if cam is None:
            return
        c2w = lateral_shift(cam.c2w, float(self.gui_lateral.value), float(self.gui_up.value))
        wxyz, pos, fov = camera_to_viser(cam.with_pose(c2w))
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.wxyz = wxyz
                client.camera.position = pos
                client.camera.fov = fov
            self._dirty[client.client_id] = True

    def mark_dirty(self) -> None:
        for k in self._dirty:
            self._dirty[k] = True

    def _on_connect(self, client) -> None:
        self._dirty[client.client_id] = True
        client.camera.on_update(lambda _: self._dirty.__setitem__(client.client_id, True))
        # Start from the first recorded camera instead of viser's default pose.
        cam = self.recorded_camera(int(self.gui_frame.value), self.gui_sensor.value)
        if cam is not None:
            wxyz, pos, fov = camera_to_viser(cam)
            with client.atomic():
                client.camera.up_direction = (0.0, 0.0, 1.0)
                client.camera.wxyz = wxyz
                client.camera.position = pos
                client.camera.fov = fov
        threading.Thread(target=self._render_loop, args=(client,), daemon=True).start()

    def _render_loop(self, client) -> None:
        while client.client_id in self._dirty:
            if not self._dirty.get(client.client_id):
                time.sleep(0.01)
                continue
            self._dirty[client.client_id] = False
            cam = viser_to_camera(
                client.camera.wxyz, client.camera.position, client.camera.fov, client.camera.aspect,
                self.state.width, self.state.frame,
                cam_id=self._sensor_id(),
            )
            t0 = time.time()
            with self._lock:
                img = render_view(self.graph, cam, self.state, self.backend)
            dt = time.time() - t0
            client.scene.set_background_image(img, format="jpeg", jpeg_quality=90)
            self.gui_info.content = (
                f"**{img.shape[1]}x{img.shape[0]}** in {dt * 1000:.0f} ms, "
                f"{self.graph.num_gaussians():,} Gaussians, frame {self.state.frame}"
            )

    def _sensor_id(self) -> int:
        names = self.scene.sensor_names
        v = self.gui_sensor.value
        return names.index(v) if v in names else 0

    def run(self) -> None:
        print(f"viewer running at http://localhost:{self.server.get_port()}  (Ctrl+C to stop)", flush=True)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._dirty.clear()
        self.gui_play.value = False
        self.server.stop()
