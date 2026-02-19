import os
import math
import numpy as np
import torch
import cv2

try:
    import open3d as o3d
except Exception:
    o3d = None


def get_dataset_spec(dataset):
    dataset = dataset.lower()
    if dataset in ("nusc", "nuscenes"):
        return {
            "voxel_size": np.array([0.5, 0.5, 0.5], dtype=np.float32),
            "vox_origin": np.array([-50.0, -50.0, -5.0], dtype=np.float32),
            "empty_label": 17,
            "class_range": (0, 16),
            "ray_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
    if dataset == "kitti":
        return {
            "voxel_size": np.array([0.2, 0.2, 0.2], dtype=np.float32),
            "vox_origin": np.array([0.0, -25.6, -2.0], dtype=np.float32),
            "empty_label": 0,
            "class_range": (1, 19),
            "ray_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
    if dataset == "kitti360":
        return {
            "voxel_size": np.array([0.2, 0.2, 0.2], dtype=np.float32),
            "vox_origin": np.array([0.0, -25.6, -2.0], dtype=np.float32),
            "empty_label": 0,
            "class_range": (1, 18),
            "ray_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_nuscenes_colormap_uint8():
    return np.array(
        [
            [0, 0, 0],
            [255, 120, 50],
            [255, 192, 203],
            [255, 255, 0],
            [0, 150, 245],
            [0, 255, 255],
            [255, 127, 0],
            [255, 0, 0],
            [255, 240, 150],
            [135, 60, 0],
            [160, 32, 240],
            [255, 0, 255],
            [139, 137, 137],
            [75, 0, 75],
            [150, 240, 80],
            [230, 230, 250],
            [0, 175, 0],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )


def occ_to_points(occ, dataset):
    spec = get_dataset_spec(dataset)
    occ = np.asarray(occ, dtype=np.int64)
    min_label, max_label = spec["class_range"]
    mask = (occ >= min_label) & (occ <= max_label)
    idx = np.argwhere(mask)
    if idx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    xyz = (idx.astype(np.float32) + 0.5) * spec["voxel_size"][None] + spec["vox_origin"][None]
    labels = occ[mask]
    return xyz, labels


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def _get_camera_preset(view_name):
    view_name = view_name.lower()
    if view_name == "top":
        return {
            "position": np.array([0.0, 0.0, 95.0], dtype=np.float32),
            "lookat": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "fov_deg": 45.0,
        }
    if view_name == "front":
        return {
            "position": np.array([-80.0, 0.0, 20.0], dtype=np.float32),
            "lookat": np.array([15.0, 0.0, 0.0], dtype=np.float32),
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "fov_deg": 55.0,
        }
    if view_name in ("bird45", "bird_45", "sky45"):
        return {
            "position": np.array([-55.0, -55.0, 55.0], dtype=np.float32),
            "lookat": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "fov_deg": 50.0,
        }
    raise ValueError(f"Unknown camera view preset: {view_name}")


def render_occ_snapshot(
    occ,
    dataset,
    view_name="bird45",
    width=800,
    height=800,
    point_radius=1,
    max_points=180000,
):
    points, labels = occ_to_points(occ, dataset)
    canvas = np.ones((int(height), int(width), 3), dtype=np.uint8) * 255
    if points.shape[0] == 0:
        return canvas

    if points.shape[0] > int(max_points):
        step = int(np.ceil(points.shape[0] / float(max_points)))
        points = points[::step]
        labels = labels[::step]

    preset = _get_camera_preset(view_name)
    pos = preset["position"]
    lookat = preset["lookat"]
    up = _normalize(preset["up"])
    forward = _normalize(lookat - pos)
    right = _normalize(np.cross(forward, up))
    cam_up = _normalize(np.cross(right, forward))

    rel = points - pos[None, :]
    xc = rel @ right
    yc = rel @ cam_up
    zc = rel @ forward

    valid = zc > 1e-2
    if not np.any(valid):
        return canvas
    xc = xc[valid]
    yc = yc[valid]
    zc = zc[valid]
    labels = labels[valid]

    f = 0.5 * float(width) / np.tan(np.deg2rad(0.5 * preset["fov_deg"]))
    px = (0.5 * float(width) + f * (xc / zc)).astype(np.int32)
    py = (0.5 * float(height) - f * (yc / zc)).astype(np.int32)
    inside = (px >= 0) & (px < int(width)) & (py >= 0) & (py < int(height))
    if not np.any(inside):
        return canvas
    px = px[inside]
    py = py[inside]
    zc = zc[inside]
    labels = labels[inside]

    colors = get_nuscenes_colormap_uint8()[np.clip(labels, 0, 16)]
    # Draw far -> near
    order = np.argsort(zc)[::-1]
    for i in order:
        bgr = (int(colors[i, 2]), int(colors[i, 1]), int(colors[i, 0]))
        cv2.circle(canvas, (int(px[i]), int(py[i])), int(point_radius), bgr, -1, lineType=cv2.LINE_AA)
    return canvas


def save_open3d_occ_video(
    occ,
    save_path,
    dataset,
    width=1280,
    height=720,
    fps=12,
    num_frames=120,
    point_size=3.0,
):
    if o3d is None:
        raise ImportError("open3d is not installed. Please install open3d to use Open3D video visualization.")

    points, labels = occ_to_points(occ, dataset)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {save_path}")

    vis = o3d.visualization.Visualizer()
    created = vis.create_window(window_name="occ", width=int(width), height=int(height), visible=False)
    if not created:
        writer.release()
        _save_fallback_matplotlib_video(
            points=points,
            labels=labels,
            save_path=save_path,
            width=width,
            height=height,
            fps=fps,
            num_frames=num_frames,
        )
        return

    opt = vis.get_render_option()
    if opt is None:
        vis.destroy_window()
        writer.release()
        _save_fallback_matplotlib_video(
            points=points,
            labels=labels,
            save_path=save_path,
            width=width,
            height=height,
            fps=fps,
            num_frames=num_frames,
        )
        return
    opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    opt.point_size = float(point_size)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cmap = get_nuscenes_colormap_uint8().astype(np.float32) / 255.0
    colors = cmap[np.clip(labels, 0, len(cmap) - 1)]
    if colors.shape[0] == 0:
        colors = np.zeros((0, 3), dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0.0, 0.0, 0.0]))

    ctr = vis.get_view_control()
    ctr.set_lookat(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    ctr.set_up(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    ctr.set_front(np.array([1.0, 1.0, 0.7], dtype=np.float64))
    ctr.set_zoom(0.4)

    for i in range(int(num_frames)):
        ctr.rotate(8.0, 0.0)
        vis.poll_events()
        vis.update_renderer()
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=np.float32)
        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()
    vis.destroy_window()


def _save_fallback_matplotlib_video(points, labels, save_path, width, height, fps, num_frames):
    """Headless fallback when Open3D cannot create an offscreen context."""
    cmap = get_nuscenes_colormap_uint8().astype(np.float32) / 255.0
    colors = cmap[np.clip(labels, 0, len(cmap) - 1)] if labels.size > 0 else np.zeros((0, 3), dtype=np.float32)

    writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open fallback video writer: {save_path}")

    if points.shape[0] == 0:
        blank = np.ones((int(height), int(width), 3), dtype=np.uint8) * 255
        for _ in range(int(num_frames)):
            writer.write(blank)
        writer.release()
        return

    pts = points.astype(np.float32)
    center = pts.mean(axis=0, keepdims=True)
    pts = pts - center
    scale = np.max(np.linalg.norm(pts, axis=1))
    scale = max(scale, 1e-3)
    pts = pts / scale
    cols = (colors[:, ::-1] * 255.0).astype(np.uint8)  # RGB->BGR

    f = 0.8 * min(width, height)
    cam_dist = 3.0
    tilt = np.deg2rad(25.0)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt), -np.sin(tilt)],
            [0.0, np.sin(tilt), np.cos(tilt)],
        ],
        dtype=np.float32,
    )

    for i in range(int(num_frames)):
        theta = 2.0 * np.pi * float(i) / max(1, int(num_frames))
        rz = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        rot = rz @ rx
        p = pts @ rot.T

        depth = p[:, 1] + cam_dist
        valid = depth > 1e-3
        if not np.any(valid):
            frame = np.ones((int(height), int(width), 3), dtype=np.uint8) * 255
            writer.write(frame)
            continue

        u = p[valid, 0]
        v = p[valid, 2]
        z = depth[valid]
        px = (width * 0.5 + f * (u / z)).astype(np.int32)
        py = (height * 0.5 - f * (v / z)).astype(np.int32)
        c = cols[valid]

        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        px, py, z, c = px[inside], py[inside], z[inside], c[inside]
        order = np.argsort(z)[::-1]

        frame = np.ones((int(height), int(width), 3), dtype=np.uint8) * 255
        for j in order:
            cv2.circle(frame, (int(px[j]), int(py[j])), 1, tuple(int(x) for x in c[j]), -1, lineType=cv2.LINE_AA)
        writer.write(frame)

    writer.release()


def generate_lidar_rays(num_azimuth=360):
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    rays = []
    for pitch in pitch_angles:
        for azim in np.linspace(0.0, 360.0, int(num_azimuth), endpoint=False):
            az = np.deg2rad(azim)
            x = np.cos(pitch) * np.cos(az)
            y = np.cos(pitch) * np.sin(az)
            z = np.sin(pitch)
            rays.append((x, y, z))
    return np.asarray(rays, dtype=np.float32)


def occ2bev_image(semantics, empty_label):
    h, w, d = semantics.shape
    semantics_2d = np.ones((h, w), dtype=np.int32) * int(empty_label)
    for i in range(d):
        sem_i = semantics[..., i]
        non_empty = sem_i != int(empty_label)
        semantics_2d[non_empty] = sem_i[non_empty]
    color_map = get_nuscenes_colormap_uint8()
    image = color_map[np.clip(semantics_2d, 0, len(color_map) - 1)][..., :3]
    return image


def ray_cast_occ(
    occ,
    dataset,
    max_range=80.0,
    step=0.5,
    num_azimuth=360,
):
    spec = get_dataset_spec(dataset)
    occ = np.asarray(occ, dtype=np.int64)
    rays = generate_lidar_rays(num_azimuth=num_azimuth)  # [R, 3]
    dists = np.arange(float(step), float(max_range) + 1e-6, float(step), dtype=np.float32)  # [S]

    origin = spec["ray_origin"][None, None, :]
    points = origin + rays[:, None, :] * dists[None, :, None]  # [R, S, 3]
    idx = np.floor((points - spec["vox_origin"][None, None, :]) / spec["voxel_size"][None, None, :]).astype(np.int32)

    valid = (
        (idx[..., 0] >= 0) & (idx[..., 0] < occ.shape[0]) &
        (idx[..., 1] >= 0) & (idx[..., 1] < occ.shape[1]) &
        (idx[..., 2] >= 0) & (idx[..., 2] < occ.shape[2])
    )

    sampled = np.ones(idx.shape[:2], dtype=np.int64) * int(spec["empty_label"])
    valid_idx = np.where(valid)
    if valid_idx[0].size > 0:
        sampled[valid_idx] = occ[
            idx[valid_idx][..., 0],
            idx[valid_idx][..., 1],
            idx[valid_idx][..., 2],
        ]

    hit_mask = sampled != int(spec["empty_label"])
    has_hit = hit_mask.any(axis=1)
    first_hit_idx = np.argmax(hit_mask, axis=1)
    ray_ids = np.where(has_hit)[0]

    hit_points = np.zeros((0, 3), dtype=np.float32)
    hit_classes = np.zeros((0,), dtype=np.int64)
    if ray_ids.size > 0:
        hit_steps = first_hit_idx[ray_ids]
        hit_points = points[ray_ids, hit_steps]
        hit_classes = sampled[ray_ids, hit_steps]

    dense = np.ones_like(occ, dtype=np.int64) * int(spec["empty_label"])
    if hit_points.shape[0] > 0:
        hit_idx = np.floor((hit_points - spec["vox_origin"][None, :]) / spec["voxel_size"][None, :]).astype(np.int32)
        valid2 = (
            (hit_idx[:, 0] >= 0) & (hit_idx[:, 0] < occ.shape[0]) &
            (hit_idx[:, 1] >= 0) & (hit_idx[:, 1] < occ.shape[1]) &
            (hit_idx[:, 2] >= 0) & (hit_idx[:, 2] < occ.shape[2])
        )
        hit_idx = hit_idx[valid2]
        hit_classes = hit_classes[valid2]
        dense[hit_idx[:, 0], hit_idx[:, 1], hit_idx[:, 2]] = hit_classes

    bev = occ2bev_image(dense, empty_label=spec["empty_label"])
    return bev, hit_points, hit_classes


def tensor_to_numpy_occ(tensor_occ, occ_shape):
    if isinstance(tensor_occ, torch.Tensor):
        return tensor_occ.detach().cpu().reshape(*occ_shape).to(torch.int64).numpy()
    return np.asarray(tensor_occ).reshape(*occ_shape).astype(np.int64)
