from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.linalg import solve
from scipy.spatial import ConvexHull


CANVAS_SIZE = 2048
RADIAL_PROFILE_BINS = 128
WIDTH_PROFILE_BINS = 32
CONTOUR_POINTS = 256
TEMPLATE_BASE_CENTER = np.asarray([1536.0, 1024.0], dtype=np.float64)
TEMPLATE_SCALE = 1400.0
REGION_NAMES = ["proximal", "distal_outer_margin", "posterior_notch", "dorsal_ventral_transition"]
SEGMENT_LANDMARK_ORDER = ["base_dorsal", "distal_tip", "posterior_notch", "base_ventral"]


def otsu_threshold(gray: np.ndarray) -> int:
    flat = np.clip(gray.astype(np.uint8).ravel(), 0, 255)
    if flat.size == 0:
        return 0
    hist = np.bincount(flat, minlength=256).astype(np.float64)
    total = float(flat.size)
    sum_total = float(np.dot(np.arange(256), hist))
    sum_background = 0.0
    weight_background = 0.0
    best_threshold = 0
    best_variance = -1.0
    for threshold in range(256):
        weight_background += hist[threshold]
        if weight_background <= 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground <= 0:
            break
        sum_background += threshold * hist[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return int(best_threshold)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count <= 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return labels == largest


def extract_wing_mask(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image, dtype=np.uint8)
    threshold = max(5, otsu_threshold(gray))
    mask = gray > threshold
    if int(mask.sum()) < 64:
        mask = gray > 0
    mask = largest_component(mask)
    structure = np.ones((3, 3), dtype=bool)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_closing(mask, structure=structure, iterations=2)
    mask = ndi.binary_opening(mask, structure=structure, iterations=1)
    mask = largest_component(mask)
    return mask.astype(bool)


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    return mask & ~eroded


def extract_contour_points(mask: np.ndarray, n_points: int = CONTOUR_POINTS) -> np.ndarray:
    boundary = boundary_mask(mask)
    ys, xs = np.nonzero(boundary)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    points = points[order]
    if len(points) <= n_points:
        return points
    sample_idx = np.linspace(0, len(points) - 1, n_points).round().astype(int)
    return points[sample_idx]


def _neighbor_offsets() -> list[tuple[int, int]]:
    return [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]


def trace_contour_pixels(mask: np.ndarray) -> np.ndarray:
    boundary = boundary_mask(mask)
    points = np.argwhere(boundary)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    point_set = {tuple(int(v) for v in point) for point in points}
    start = min(point_set, key=lambda item: (item[0], item[1]))
    ordered = [start]
    visited = {start}
    current = start
    prev_dir = 0
    offsets = _neighbor_offsets()
    max_steps = max(len(point_set) * 3, 32)
    while len(ordered) < max_steps:
        found = None
        for step in range(8):
            idx = (prev_dir + step) % 8
            dy, dx = offsets[idx]
            candidate = (current[0] + dy, current[1] + dx)
            if candidate in point_set and (candidate not in visited or candidate == start):
                found = candidate
                prev_dir = (idx + 5) % 8
                break
        if found is None:
            break
        if found == start and len(ordered) > 8:
            break
        ordered.append(found)
        visited.add(found)
        current = found
    coords = np.asarray([[point[1], point[0]] for point in ordered], dtype=np.float64)
    if len(coords) < 8:
        return extract_contour_points(mask, n_points=min(CONTOUR_POINTS, max(16, len(points))))
    return coords


def sample_ordered_contour(contour: np.ndarray, n_points: int = CONTOUR_POINTS) -> np.ndarray:
    contour = np.asarray(contour, dtype=np.float64)
    if len(contour) == 0:
        return contour
    if len(contour) <= n_points:
        return contour
    distances = np.sqrt(np.sum(np.diff(np.vstack([contour, contour[:1]]), axis=0) ** 2, axis=1))
    cumulative = np.r_[0.0, np.cumsum(distances[:-1])]
    total = float(distances.sum())
    if total <= 1e-6:
        sample_idx = np.linspace(0, len(contour) - 1, n_points).round().astype(int)
        return contour[sample_idx]
    targets = np.linspace(0.0, total, n_points, endpoint=False)
    sampled = []
    for target in targets:
        idx = int(np.searchsorted(cumulative, target, side="right") - 1)
        idx = np.clip(idx, 0, len(contour) - 1)
        next_idx = (idx + 1) % len(contour)
        local = target - cumulative[idx]
        segment = distances[idx] if idx < len(distances) else distances[-1]
        alpha = 0.0 if segment <= 1e-6 else local / segment
        sampled.append((1.0 - alpha) * contour[idx] + alpha * contour[next_idx])
    return np.asarray(sampled, dtype=np.float64)


def extract_ordered_contour(mask: np.ndarray, n_points: int = CONTOUR_POINTS) -> np.ndarray:
    return sample_ordered_contour(trace_contour_pixels(mask), n_points=n_points)


def nearest_contour_index(contour: np.ndarray, point: np.ndarray) -> int:
    contour = np.asarray(contour, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    if len(contour) == 0:
        return 0
    return int(np.argmin(np.linalg.norm(contour - point.reshape(1, 2), axis=1)))


def sample_contour_segment(contour: np.ndarray, start_idx: int, end_idx: int, count: int) -> np.ndarray:
    contour = np.asarray(contour, dtype=np.float64)
    n = len(contour)
    if n == 0 or count <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    if start_idx == end_idx:
        return np.repeat(contour[start_idx : start_idx + 1], count, axis=0)
    if end_idx < start_idx:
        indices = list(range(start_idx, n)) + list(range(0, end_idx + 1))
    else:
        indices = list(range(start_idx, end_idx + 1))
    segment = contour[indices]
    return sample_ordered_contour(segment, n_points=count)


def stable_contour_correspondence(
    mask: np.ndarray,
    landmarks: np.ndarray,
    points_per_segment: int = 24,
) -> tuple[np.ndarray, dict[str, int]]:
    contour = extract_ordered_contour(mask, n_points=max(4 * points_per_segment, 128))
    named = named_landmarks_from_points(np.asarray(landmarks, dtype=np.float64))
    anchor_indices = {
        name: nearest_contour_index(contour, np.asarray([named[name]["x"], named[name]["y"]], dtype=np.float64))
        for name in SEGMENT_LANDMARK_ORDER
    }
    sampled_segments = []
    order = SEGMENT_LANDMARK_ORDER
    for idx, start_name in enumerate(order):
        end_name = order[(idx + 1) % len(order)]
        sampled_segments.append(
            sample_contour_segment(
                contour,
                anchor_indices[start_name],
                anchor_indices[end_name],
                count=points_per_segment,
            )
        )
    correspondence = np.vstack(sampled_segments).astype(np.float64)
    return correspondence, anchor_indices


def points_from_named_landmarks(landmarks: dict[str, dict[str, float]]) -> np.ndarray:
    order = ["base_dorsal", "base_ventral", "distal_tip", "posterior_notch"]
    return np.asarray([[float(landmarks[name]["x"]), float(landmarks[name]["y"])] for name in order], dtype=np.float64)


def named_landmarks_from_points(points: np.ndarray) -> dict[str, dict[str, float]]:
    order = ["base_dorsal", "base_ventral", "distal_tip", "posterior_notch"]
    return {
        name: {"x": float(points[idx, 0]), "y": float(points[idx, 1])}
        for idx, name in enumerate(order)
    }


def base_center_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    return landmarks[:2].mean(axis=0)


def canonical_basis_from_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    base_center = base_center_from_landmarks(landmarks)
    distal = landmarks[2] - base_center
    scale = float(np.linalg.norm(distal))
    if scale <= 1e-6:
        scale = 1.0
        x_axis = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        x_axis = distal / scale
    y_axis = np.asarray([-x_axis[1], x_axis[0]], dtype=np.float64)
    posterior_local = (landmarks[3] - base_center) @ y_axis
    if posterior_local < 0:
        y_axis = -y_axis
    basis = np.vstack([x_axis, y_axis])
    return base_center, basis, scale


def to_canonical(points: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    base_center, basis, scale = canonical_basis_from_landmarks(landmarks)
    centered = points.astype(np.float64) - base_center
    return (centered @ basis.T) / max(scale, 1e-6)


def from_canonical(points: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    base_center, basis, scale = canonical_basis_from_landmarks(landmarks)
    return (points.astype(np.float64) * scale) @ basis + base_center


def align_contour_with_landmarks(contour: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    return to_canonical(contour, landmarks)


def canonical_to_template(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    x = TEMPLATE_BASE_CENTER[0] - TEMPLATE_SCALE * points[:, 0]
    y = TEMPLATE_BASE_CENTER[1] + TEMPLATE_SCALE * points[:, 1]
    return np.column_stack([x, y])


def template_to_canonical(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    x = (TEMPLATE_BASE_CENTER[0] - points[:, 0]) / TEMPLATE_SCALE
    y = (points[:, 1] - TEMPLATE_BASE_CENTER[1]) / TEMPLATE_SCALE
    return np.column_stack([x, y])


def smooth_circular(values: np.ndarray, window: int = 7) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(3, int(window) | 1)
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    extended = np.r_[values[-pad:], values, values[:pad]]
    return np.convolve(extended, kernel, mode="valid")


def fill_profile(profile: np.ndarray) -> np.ndarray:
    values = profile.astype(np.float64).copy()
    valid = values > 0
    if valid.all():
        return values
    if valid.sum() < 2:
        fill = float(np.median(values[valid])) if valid.any() else 0.0
        values[~valid] = fill
        return values
    idx = np.arange(len(values))
    valid_idx = idx[valid]
    valid_values = values[valid]
    idx_ext = np.r_[valid_idx - len(values), valid_idx, valid_idx + len(values)]
    values_ext = np.r_[valid_values, valid_values, valid_values]
    values[~valid] = np.interp(idx[~valid], idx_ext, values_ext)
    return values


def compute_radial_profile(contour: np.ndarray, center: np.ndarray, bins: int = RADIAL_PROFILE_BINS) -> np.ndarray:
    if len(contour) == 0:
        return np.zeros(bins, dtype=np.float64)
    contour = np.asarray(contour, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    dx = contour[:, 0] - center[0]
    dy = contour[:, 1] - center[1]
    angles = (np.arctan2(dy, dx) + 2.0 * np.pi) % (2.0 * np.pi)
    radii = np.sqrt(dx**2 + dy**2)
    profile = np.zeros(bins, dtype=np.float64)
    bin_index = np.floor(angles / (2.0 * np.pi) * bins).astype(int)
    bin_index = np.clip(bin_index, 0, bins - 1)
    for idx in range(bins):
        values = radii[bin_index == idx]
        if len(values):
            profile[idx] = float(values.max())
    return fill_profile(profile)


def compute_width_profile(mask: np.ndarray, landmarks: np.ndarray, bins: int = WIDTH_PROFILE_BINS) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(bins, dtype=np.float64)
    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    canonical = to_canonical(coords, landmarks)
    x = canonical[:, 0]
    y = canonical[:, 1]
    edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    widths = np.zeros(bins, dtype=np.float64)
    for idx in range(bins):
        if idx == bins - 1:
            in_bin = (x >= edges[idx]) & (x <= edges[idx + 1])
        else:
            in_bin = (x >= edges[idx]) & (x < edges[idx + 1])
        if np.any(in_bin):
            widths[idx] = float(y[in_bin].max() - y[in_bin].min())
    return fill_profile(widths)


def convex_hull_points(contour: np.ndarray) -> np.ndarray:
    if len(contour) < 3:
        return contour
    try:
        hull = ConvexHull(contour.astype(np.float64))
    except Exception:
        return contour
    return contour[hull.vertices]


def point_nearest_angle(contour: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    if len(contour) == 0:
        return np.asarray(center, dtype=np.float64)
    contour = np.asarray(contour, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    angles = np.arctan2(contour[:, 1] - center[1], contour[:, 0] - center[0])
    delta = np.angle(np.exp(1j * (angles - angle)))
    idx = int(np.argmin(np.abs(delta)))
    return contour[idx]


def compute_notch_features(contour: np.ndarray, hull: np.ndarray) -> dict[str, float | np.ndarray]:
    center = np.asarray([0.0, 0.0], dtype=np.float64)
    contour_profile = smooth_circular(compute_radial_profile(contour, center), window=9)
    hull_profile = smooth_circular(compute_radial_profile(hull, center), window=9)
    gap = np.maximum(0.0, hull_profile - contour_profile)
    idx = int(np.argmax(gap)) if len(gap) else 0
    theta = (idx + 0.5) * 2.0 * np.pi / max(1, len(gap))
    point = point_nearest_angle(contour, center, theta)
    return {
        "notch_depth": float(gap[idx]) if len(gap) else 0.0,
        "notch_angle_deg": float(np.degrees(theta)),
        "notch_point": point.astype(np.float64),
        "hull_gap_profile": gap.astype(np.float64),
    }


def compute_region_anchors(contour: np.ndarray) -> dict[str, np.ndarray]:
    contour = np.asarray(contour, dtype=np.float64)
    if len(contour) == 0:
        zero = np.asarray([0.0, 0.0], dtype=np.float64)
        return {name: zero.copy() for name in REGION_NAMES}
    hull = convex_hull_points(contour)
    notch = compute_notch_features(contour, hull)
    anchors = {
        "proximal": contour[int(np.argmin(contour[:, 0]))].astype(np.float64),
        "distal_outer_margin": contour[int(np.argmax(contour[:, 0]))].astype(np.float64),
        "posterior_notch": notch["notch_point"].astype(np.float64),
        "dorsal_ventral_transition": contour[int(np.argmin(contour[:, 1]))].astype(np.float64),
    }
    return anchors


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / len(src)
    u, singular, vt = np.linalg.svd(cov)
    reflection = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        reflection[-1, -1] = -1
    rotation = u @ reflection @ vt
    variance = np.mean(np.sum(src_centered**2, axis=1))
    scale = float(np.trace(np.diag(singular) @ reflection) / max(variance, 1e-12))
    translation = dst_mean - scale * (rotation @ src_mean)
    return rotation, translation, scale


def apply_similarity(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    return (scale * (rotation @ points.T)).T + translation


def affine_coefficients(rotation: np.ndarray, translation: np.ndarray, scale: float) -> tuple[float, ...]:
    forward = scale * rotation
    inverse = np.linalg.inv(forward)
    offset = -inverse @ translation
    return (
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(offset[0]),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(offset[1]),
    )


def transform_image_similarity(image_arr: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    image = Image.fromarray(image_arr.astype(np.uint8), mode="L")
    registered = image.transform(
        (CANVAS_SIZE, CANVAS_SIZE),
        Image.Transform.AFFINE,
        affine_coefficients(rotation, translation, scale),
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )
    return np.asarray(registered, dtype=np.uint8)


def transform_mask_similarity(mask: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    registered = image.transform(
        (CANVAS_SIZE, CANVAS_SIZE),
        Image.Transform.AFFINE,
        affine_coefficients(rotation, translation, scale),
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
    )
    return np.asarray(registered, dtype=np.uint8) > 127


def radial_basis(distances: np.ndarray) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        squared = distances**2
        values = squared * np.log(np.maximum(squared, 1e-12))
    values[~np.isfinite(values)] = 0.0
    return values


def fit_tps(control_src: np.ndarray, control_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(control_src)
    k = radial_basis(np.linalg.norm(control_src[:, None, :] - control_src[None, :, :], axis=2))
    # Small landmark sets with partially aligned control points can become close to singular.
    # A slightly stronger diagonal regularizer keeps the warp stable for batch registration.
    k = k + np.eye(n, dtype=np.float64) * 1e-3
    p = np.column_stack([np.ones(n), control_src])
    upper = np.hstack([k, p])
    lower = np.hstack([p.T, np.zeros((3, 3), dtype=np.float64)])
    system = np.vstack([upper, lower])
    params_x = solve(system, np.r_[control_dst[:, 0], np.zeros(3)], assume_a="sym")
    params_y = solve(system, np.r_[control_dst[:, 1], np.zeros(3)], assume_a="sym")
    return params_x, params_y, k


def evaluate_tps(control_src: np.ndarray, params: np.ndarray, query: np.ndarray) -> np.ndarray:
    n = len(control_src)
    w = params[:n]
    a = params[n:]
    distances = np.linalg.norm(query[:, None, :] - control_src[None, :, :], axis=2)
    basis = radial_basis(distances)
    affine = a[0] + a[1] * query[:, 0] + a[2] * query[:, 1]
    return basis @ w + affine


def tps_bending_energy(params_x: np.ndarray, params_y: np.ndarray, kernel: np.ndarray) -> float:
    n = kernel.shape[0]
    wx = params_x[:n]
    wy = params_y[:n]
    energy = float(wx.T @ kernel @ wx + wy.T @ kernel @ wy)
    return max(0.0, energy)


def tps_warp(image_arr: np.ndarray, control_src: np.ndarray, control_dst: np.ndarray, order: int = 1, energy_only: bool = False) -> tuple[np.ndarray | None, float]:
    params_x, params_y, kernel = fit_tps(control_dst, control_src)
    energy = tps_bending_energy(params_x, params_y, kernel)
    if energy_only:
        return None, energy
    height, width = image_arr.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    sample_x = evaluate_tps(control_dst, params_x, query).reshape(height, width)
    sample_y = evaluate_tps(control_dst, params_y, query).reshape(height, width)
    warped = ndi.map_coordinates(
        image_arr.astype(np.float32),
        [sample_y, sample_x],
        order=order,
        mode="constant",
        cval=0.0,
    )
    return np.clip(warped, 0, 255).astype(np.uint8), energy


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = float(np.logical_or(a, b).sum())
    if union <= 0:
        return 0.0
    intersection = float(np.logical_and(a, b).sum())
    return intersection / union


def basic_morphology_features(mask: np.ndarray, contour: np.ndarray, landmarks: np.ndarray) -> dict[str, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {
            "morph_area_ratio": 0.0,
            "morph_bbox_aspect": 0.0,
            "morph_axis_ratio": 0.0,
            "morph_radial_std": 0.0,
            "morph_notch_depth": 0.0,
        }
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    area = float(len(xs))
    bbox_width = float(max_x - min_x + 1)
    bbox_height = float(max_y - min_y + 1)
    canonical = to_canonical(np.column_stack([xs.astype(np.float64), ys.astype(np.float64)]), landmarks)
    cov = np.cov(canonical, rowvar=False) if len(canonical) > 1 else np.eye(2)
    eigenvalues, _ = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    major = float(math.sqrt(eigenvalues[order[0]]) * 4.0)
    minor = float(math.sqrt(eigenvalues[order[1]]) * 4.0) if len(eigenvalues) > 1 else 0.0
    contour_canonical = to_canonical(contour, landmarks)
    radial = compute_radial_profile(contour_canonical, np.asarray([0.0, 0.0], dtype=np.float64))
    notch = compute_notch_features(contour_canonical, convex_hull_points(contour_canonical))
    return {
        "morph_area_ratio": area / float(mask.shape[0] * mask.shape[1]),
        "morph_bbox_aspect": bbox_width / max(1.0, bbox_height),
        "morph_axis_ratio": major / max(minor, 1e-6),
        "morph_radial_std": float(np.std(radial)),
        "morph_notch_depth": float(notch["notch_depth"]),
    }


def basic_mask_morphology_features(mask: np.ndarray, contour: np.ndarray) -> dict[str, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {
            "morph_area_ratio": 0.0,
            "morph_bbox_aspect": 0.0,
            "morph_extent": 0.0,
            "morph_radial_std": 0.0,
            "morph_notch_depth": 0.0,
        }
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    area = float(len(xs))
    bbox_width = float(max_x - min_x + 1)
    bbox_height = float(max_y - min_y + 1)
    center = np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float64)
    radial = compute_radial_profile(contour.astype(np.float64), center)
    hull = convex_hull_points(contour.astype(np.float64))
    notch = compute_notch_features(contour.astype(np.float64) - center, hull.astype(np.float64) - center)
    bbox_area = bbox_width * bbox_height
    return {
        "morph_area_ratio": area / float(mask.shape[0] * mask.shape[1]),
        "morph_bbox_aspect": bbox_width / max(1.0, bbox_height),
        "morph_extent": area / max(1.0, bbox_area),
        "morph_radial_std": float(np.std(radial)),
        "morph_notch_depth": float(notch["notch_depth"]),
    }


def save_mask_png(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(output_path)


def save_registration_qc(
    source_image: np.ndarray,
    base_aligned_image: np.ndarray,
    stage_registered_image: np.ndarray,
    base_landmarks: np.ndarray,
    stage_landmarks: np.ndarray,
    source_controls: np.ndarray,
    stage_controls: np.ndarray,
    output_path: Path,
) -> None:
    def to_rgb(arr: np.ndarray) -> Image.Image:
        return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")

    panels = [to_rgb(source_image), to_rgb(base_aligned_image), to_rgb(stage_registered_image)]
    colors = ["#ff5d73", "#ffb84d", "#4cc38a", "#57a0ff", "#b565f5", "#15b5ff", "#ffd166", "#06d6a0"]

    for image, points in [(panels[1], base_landmarks), (panels[2], stage_landmarks)]:
        draw = ImageDraw.Draw(image)
        for idx, point in enumerate(points):
            x, y = point
            color = colors[idx % len(colors)]
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=3)

    draw = ImageDraw.Draw(panels[2])
    for idx, point in enumerate(source_controls):
        x, y = point
        color = colors[idx % len(colors)]
        draw.line((x - 5, y, x + 5, y), fill=color, width=2)
        draw.line((x, y - 5, x, y + 5), fill=color, width=2)
    for idx, point in enumerate(stage_controls):
        x, y = point
        color = colors[idx % len(colors)]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)

    canvas = Image.new("RGB", (panels[0].width * 3, panels[0].height), (0, 0, 0))
    for idx, image in enumerate(panels):
        canvas.paste(image, (idx * image.width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), "Source", fill="white")
    draw.text((panels[0].width + 20, 20), "Base aligned", fill="white")
    draw.text((panels[0].width * 2 + 20, 20), "Stage registered", fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _as_template_landmarks(payload_or_points: dict[str, object] | np.ndarray) -> np.ndarray:
    if isinstance(payload_or_points, np.ndarray):
        return np.asarray(payload_or_points, dtype=np.float64)
    if "_landmarks" in payload_or_points:
        return np.asarray(payload_or_points["_landmarks"], dtype=np.float64)
    return points_from_named_landmarks(payload_or_points["landmarks"])


def _as_template_mask(payload: dict[str, object]) -> np.ndarray:
    if "_mask" in payload:
        return np.asarray(payload["_mask"], dtype=bool)
    mask_path = payload.get("mask_path", "")
    if not mask_path:
        return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    with Image.open(mask_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def compute_radial_anchor_points(
    contour: np.ndarray,
    center: np.ndarray | None = None,
    count: int = 16,
) -> np.ndarray:
    contour = np.asarray(contour, dtype=np.float64)
    if center is None:
        center = contour.mean(axis=0) if len(contour) else np.asarray([0.0, 0.0], dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if len(contour) == 0 or count <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, int(count), endpoint=False)
    return np.vstack([point_nearest_angle(contour, center, angle) for angle in angles]).astype(np.float64)


def compute_width_anchor_points(
    mask: np.ndarray,
    landmarks: np.ndarray,
    count: int = 8,
) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    bins = max(1, int(count) // 2)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    canonical = to_canonical(coords, landmarks)
    x_values = canonical[:, 0]
    y_values = canonical[:, 1]
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    if x_max - x_min < 1e-6:
        return np.zeros((0, 2), dtype=np.float64)
    edges = np.linspace(x_min, x_max, bins + 2)[1:-1]
    anchors_canonical = []
    for center_x in edges:
        span = max(0.03, (x_max - x_min) / max(12.0, float(bins) * 2.0))
        in_bin = np.abs(x_values - center_x) <= span
        if not np.any(in_bin):
            idx = int(np.argmin(np.abs(x_values - center_x)))
            in_bin = np.zeros_like(x_values, dtype=bool)
            in_bin[idx] = True
        y_bin = y_values[in_bin]
        anchors_canonical.append([center_x, float(y_bin.min())])
        anchors_canonical.append([center_x, float(y_bin.max())])
    anchors = from_canonical(np.asarray(anchors_canonical, dtype=np.float64), landmarks)
    return anchors.astype(np.float64)


def build_dense_template_controls(
    template_payload: dict[str, object],
    radial_anchor_count: int = 16,
    width_anchor_count: int = 8,
) -> dict[str, object]:
    landmarks = _as_template_landmarks(template_payload)
    mask = _as_template_mask(template_payload)
    contour = extract_contour_points(mask)
    center = base_center_from_landmarks(landmarks)
    region_anchors = (
        {
            name: np.asarray([float(point["x"]), float(point["y"])], dtype=np.float64)
            for name, point in template_payload.get("region_anchors", {}).items()
        }
        if template_payload.get("region_anchors")
        else {
            name: np.asarray(point, dtype=np.float64)
            for name, point in compute_region_anchors(template_to_canonical(contour)).items()
        }
    )
    if not template_payload.get("region_anchors"):
        region_anchors = {
            name: canonical_to_template(np.asarray([point], dtype=np.float64))[0]
            for name, point in region_anchors.items()
        }
    radial_anchors = compute_radial_anchor_points(contour, center=center, count=radial_anchor_count)
    width_anchors = compute_width_anchor_points(mask, landmarks, count=width_anchor_count)
    point_groups = [
        ("landmark", landmarks),
        ("region", np.vstack([region_anchors[name] for name in REGION_NAMES])),
        ("radial", radial_anchors),
        ("width", width_anchors),
    ]
    all_points = []
    names = []
    for group_name, points in point_groups:
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            continue
        for idx, point in enumerate(points):
            all_points.append(point)
            names.append(f"{group_name}_{idx:02d}")
    return {
        "landmarks": landmarks,
        "mask": mask,
        "contour": contour,
        "center": center,
        "region_anchors": region_anchors,
        "radial_anchors": radial_anchors,
        "width_anchors": width_anchors,
        "points": np.asarray(all_points, dtype=np.float64),
        "names": names,
    }


def evaluate_template_alignment_metrics(
    aligned_mask: np.ndarray,
    aligned_contour: np.ndarray,
    aligned_landmarks: np.ndarray,
    template_payload: dict[str, object],
) -> dict[str, float]:
    center = base_center_from_landmarks(aligned_landmarks)
    radial = smooth_circular(compute_radial_profile(aligned_contour, center), window=9)
    width = smooth_circular(compute_width_profile(aligned_mask, aligned_landmarks), window=5)
    canonical_contour = template_to_canonical(aligned_contour)
    notch = compute_notch_features(canonical_contour, convex_hull_points(canonical_contour))
    notch_anchor = canonical_to_template(np.asarray([notch["notch_point"]], dtype=np.float64))[0]
    template_landmarks = _as_template_landmarks(template_payload)
    template_mask = _as_template_mask(template_payload)
    template_notch = (
        np.asarray(
            [
                float(template_payload["notch_anchor"]["x"]),
                float(template_payload["notch_anchor"]["y"]),
            ],
            dtype=np.float64,
        )
        if "notch_anchor" in template_payload
        else np.asarray([0.0, 0.0], dtype=np.float64)
    )
    template_radial = np.asarray(template_payload.get("radial_profile", []), dtype=np.float64)
    template_width = np.asarray(template_payload.get("width_profile", []), dtype=np.float64)
    landmark_error = float(np.linalg.norm(aligned_landmarks - template_landmarks, axis=1).mean())
    radial_l2 = float(np.sqrt(np.mean((radial - template_radial) ** 2))) if len(template_radial) else 0.0
    width_l2 = float(np.sqrt(np.mean((width - template_width) ** 2))) if len(template_width) else 0.0
    notch_offset = float(np.linalg.norm(notch_anchor - template_notch))
    mask_score = float(mask_iou(aligned_mask, template_mask))
    return {
        "landmark_similarity_error": landmark_error,
        "radial_profile_l2": radial_l2,
        "width_profile_l2": width_l2,
        "notch_offset": notch_offset,
        "mask_iou_to_template": mask_score,
    }


def select_best_seed_image(
    candidates: list[dict[str, object]],
    preferred_label: str = "0.75",
    preferred_tolerance: float = 0.03,
) -> dict[str, object]:
    if not candidates:
        raise ValueError("No seed candidates were provided.")
    ordered = sorted(candidates, key=lambda row: (float(row["overall_template_cost"]), str(row["id"])))
    global_best = ordered[0]
    preferred = [row for row in ordered if str(row.get("label", "")) == preferred_label]
    if not preferred:
        return global_best
    preferred_best = preferred[0]
    if float(global_best["overall_template_cost"]) <= float(preferred_best["overall_template_cost"]) * (1.0 - preferred_tolerance):
        return global_best
    return preferred_best


def warp_image_between_templates(
    image_arr: np.ndarray,
    source_controls: np.ndarray,
    destination_controls: np.ndarray,
    order: int = 1,
) -> tuple[np.ndarray, float]:
    return tps_warp(image_arr, source_controls, destination_controls, order=order)


def tps_map_points(control_src: np.ndarray, control_dst: np.ndarray, query: np.ndarray) -> np.ndarray:
    params_x, params_y, _ = fit_tps(control_src, control_dst)
    mapped_x = evaluate_tps(control_src, params_x, query)
    mapped_y = evaluate_tps(control_src, params_y, query)
    return np.column_stack([mapped_x, mapped_y]).astype(np.float64)


def render_deformation_grid(
    source_controls: np.ndarray,
    destination_controls: np.ndarray,
    spacing: int = 128,
    line_width: int = 2,
) -> np.ndarray:
    grid = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(grid)
    color = 190
    for x in range(0, CANVAS_SIZE, int(spacing)):
        draw.line((x, 0, x, CANVAS_SIZE - 1), fill=color, width=line_width)
    for y in range(0, CANVAS_SIZE, int(spacing)):
        draw.line((0, y, CANVAS_SIZE - 1, y), fill=color, width=line_width)
    warped, _ = warp_image_between_templates(
        np.asarray(grid, dtype=np.uint8),
        source_controls,
        destination_controls,
        order=1,
    )
    return warped


def render_quiver_plot(
    source_controls: np.ndarray,
    destination_controls: np.ndarray,
    output_path: Path,
    background: np.ndarray | None = None,
    title: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    if background is not None:
        ax.imshow(background, cmap="gray", origin="upper")
    ax.quiver(
        source_controls[:, 0],
        source_controls[:, 1],
        destination_controls[:, 0] - source_controls[:, 0],
        destination_controls[:, 1] - source_controls[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#ff8c42",
        width=0.002,
    )
    ax.scatter(source_controls[:, 0], source_controls[:, 1], s=14, c="#57a0ff", label="source")
    ax.scatter(destination_controls[:, 0], destination_controls[:, 1], s=14, c="#4cc38a", label="dest")
    ax.set_xlim(0, CANVAS_SIZE)
    ax.set_ylim(CANVAS_SIZE, 0)
    ax.set_title(title or "Template deformation vectors")
    ax.legend(frameon=False, loc="upper right")
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def render_area_change_map(
    source_controls: np.ndarray,
    destination_controls: np.ndarray,
    source_mask: np.ndarray,
    output_path: Path,
    title: str = "",
    step: int = 16,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = max(8, int(step))
    xs = np.arange(0, CANVAS_SIZE, step, dtype=np.float64)
    ys = np.arange(0, CANVAS_SIZE, step, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    mapped = tps_map_points(source_controls, destination_controls, query).reshape(len(ys), len(xs), 2)
    mapped_x = mapped[:, :, 0]
    mapped_y = mapped[:, :, 1]
    grad_x_x = np.gradient(mapped_x, step, axis=1)
    grad_x_y = np.gradient(mapped_x, step, axis=0)
    grad_y_x = np.gradient(mapped_y, step, axis=1)
    grad_y_y = np.gradient(mapped_y, step, axis=0)
    jacobian = grad_x_x * grad_y_y - grad_x_y * grad_y_x
    signed_log = np.sign(jacobian) * np.log1p(np.abs(jacobian - 1.0))
    coarse_mask = source_mask[::step, ::step]
    if coarse_mask.shape != signed_log.shape:
        coarse_mask = coarse_mask[: signed_log.shape[0], : signed_log.shape[1]]
    signed_log = np.where(coarse_mask, signed_log, np.nan)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(signed_log, cmap="coolwarm", origin="upper")
    ax.set_title(title or "Local area change")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
