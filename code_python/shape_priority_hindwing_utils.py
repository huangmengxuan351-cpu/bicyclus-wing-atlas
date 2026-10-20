from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFile
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

from pose_landmark_utils import (
    IMAGE_SUFFIXES,
    LABELS,
    LANDMARK_NAMES,
    ROOT,
    read_csv_rows,
    write_csv_rows,
)
from register_with_stage_templates import build_registration_features, load_stage_templates, make_feature_row
from stage_template_geometry import (
    apply_similarity,
    base_center_from_landmarks,
    compute_radial_profile,
    compute_region_anchors,
    compute_width_profile,
    convex_hull_points,
    extract_contour_points,
    extract_ordered_contour,
    fill_profile,
    mask_iou,
    template_to_canonical,
    tps_warp,
    transform_image_similarity,
    transform_mask_similarity,
    umeyama_similarity,
)


DEFAULT_CLASSIFIER_DIR = ROOT / "output_results" / "shape_priority_hindwing_classifier"
DEFAULT_GT100_CLEANING_CSV = ROOT / "output_results" / "gt100_gene_cleaning_project" / "gt100_cleaning_status.csv"
DEFAULT_STAGE_FEATURES_CSV = ROOT / "output_results" / "landmark_pose_project" / "stage_registration_features.csv"
DEFAULT_LANDMARK_WEIGHTS = (
    ROOT / "output_results" / "landmark_pose_project" / "yolo_pose_runs" / "hindwing_landmark_pose" / "weights" / "best.pt"
)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def infer_modality(path: str | Path) -> str:
    text = str(path).lower()
    if "dapi" in text:
        return "dapi"
    if "hcr" in text:
        return "hcr"
    if "c001t001" in text or "c001z" in text:
        return "rgb_confocal"
    if any(token in text for token in ["488", "514", "546", "647"]):
        return "fluorescence"
    return "unknown"


def normalized_stem(text: str) -> str:
    stem = Path(text).stem.lower()
    stem = re.sub(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", "", stem)
    stem = re.sub(r"(^|_)(0_5|0_75|1_0)(_|$)", "_", stem)
    stem = re.sub(r"(^|_)(forewing|hindwing|uncertain)(_|$)", "_", stem)
    stem = re.sub(r"(^|_)(dapi|hcr|channel_c\d+|channel_c\d|c001t001|c001z\d+)(_|$)", "_", stem)
    stem = re.sub(r"(^|_)(488|514|546|647)hcr(_|$)", "_", stem)
    stem = re.sub(r"(^|_)\d{4,}(_|$)", "_", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "sample"


def derive_split_group(item_id: str, image_path: str | Path, source_group: str) -> str:
    image_path = Path(image_path)
    parts = [str(source_group).strip().lower()]
    parent = image_path.parent.name.lower().strip()
    if parent:
        parts.append(parent)
    parts.append(normalized_stem(item_id or image_path.stem))
    return "::".join(part for part in parts if part)


def iter_image_paths(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def load_gray_image(image_path: str | Path) -> np.ndarray:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(image_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def percentile_normalize(gray: np.ndarray, lower_q: float = 1.0, upper_q: float = 99.0) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    if gray.size == 0:
        return gray
    lo = float(np.percentile(gray, lower_q))
    hi = float(np.percentile(gray, upper_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        return gray.copy()
    normalized = (gray.astype(np.float32) - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def otsu_threshold(gray: np.ndarray) -> int:
    flat = np.clip(np.asarray(gray, dtype=np.uint8).ravel(), 0, 255)
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
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return labels == largest


def finalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) < 64:
        return np.zeros_like(mask, dtype=bool)
    mask = largest_component(mask)
    structure = np.ones((3, 3), dtype=bool)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_closing(mask, structure=structure, iterations=2)
    mask = ndi.binary_opening(mask, structure=structure, iterations=1)
    mask = largest_component(mask)
    return mask.astype(bool)


def _mask_quality(mask: np.ndarray) -> float:
    if mask.size == 0 or int(mask.sum()) < 64:
        return -1e9
    ys, xs = np.nonzero(mask)
    if len(xs) < 64:
        return -1e9
    height, width = mask.shape
    area_ratio = float(mask.mean())
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    bbox_width = float(max_x - min_x + 1)
    bbox_height = float(max_y - min_y + 1)
    bbox_area = max(1.0, bbox_width * bbox_height)
    extent = float(len(xs)) / bbox_area
    touches_border = float(
        mask[0, :].any() + mask[-1, :].any() + mask[:, 0].any() + mask[:, -1].any()
    ) / 4.0
    aspect = bbox_width / max(1.0, bbox_height)
    component_score = 1.0
    shape_penalty = 0.0
    if area_ratio < 0.005 or area_ratio > 0.92:
        shape_penalty += 2.5
    if extent < 0.12 or extent > 0.98:
        shape_penalty += 0.8
    if aspect < 0.12 or aspect > 8.0:
        shape_penalty += 0.8
    compactness = float(len(xs)) / max(1.0, (bbox_width + bbox_height))
    score = (
        2.4 * extent
        + 0.4 * compactness / max(1.0, math.sqrt(height * width))
        + 0.3 * component_score
        - 1.2 * touches_border
        - shape_penalty
    )
    return float(score)


def extract_best_wing_mask(gray: np.ndarray) -> dict[str, object]:
    normalized = percentile_normalize(gray)
    threshold = max(5, otsu_threshold(normalized))
    positive_raw = normalized > threshold
    inverse_raw = normalized < threshold
    positive = finalize_mask(positive_raw)
    inverse = finalize_mask(inverse_raw)
    positive_score = _mask_quality(positive)
    inverse_score = _mask_quality(inverse)
    chosen_mask = positive if positive_score >= inverse_score else inverse
    polarity = "positive" if positive_score >= inverse_score else "inverse"
    chosen_score = positive_score if polarity == "positive" else inverse_score
    if int(chosen_mask.sum()) < 64:
        fallback = finalize_mask(normalized > 0)
        fallback_score = _mask_quality(fallback)
        if fallback_score > chosen_score:
            chosen_mask = fallback
            polarity = "fallback_nonzero"
            chosen_score = fallback_score
    return {
        "gray": np.asarray(gray, dtype=np.uint8),
        "normalized_gray": normalized,
        "mask": chosen_mask.astype(bool),
        "threshold": int(threshold),
        "polarity": polarity,
        "mask_quality": float(chosen_score),
        "positive_score": float(positive_score),
        "inverse_score": float(inverse_score),
    }


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def resize_mask_crop(mask: np.ndarray, size: int = 160, padding_fraction: float = 0.08) -> np.ndarray:
    bbox = mask_bbox(mask)
    if bbox is None:
        return np.zeros((size, size), dtype=bool)
    min_x, min_y, max_x, max_y = bbox
    crop = mask[min_y : max_y + 1, min_x : max_x + 1]
    height, width = crop.shape
    side = int(math.ceil(max(height, width) * (1.0 + padding_fraction * 2.0)))
    side = max(side, max(height, width), 8)
    canvas = np.zeros((side, side), dtype=np.uint8)
    offset_y = (side - height) // 2
    offset_x = (side - width) // 2
    canvas[offset_y : offset_y + height, offset_x : offset_x + width] = crop.astype(np.uint8) * 255
    resized = Image.fromarray(canvas, mode="L").resize((size, size), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 127


@lru_cache(maxsize=4)
def _cached_base_template_mask(project_dir: str) -> np.ndarray:
    _, stage_templates = load_stage_templates(Path(project_dir))
    base_template, _ = load_stage_templates(Path(project_dir))
    mask = np.asarray(base_template.get("_mask"), dtype=bool)
    if mask.size == 0 or int(mask.sum()) == 0:
        candidate = next(iter(stage_templates.values()))
        mask = np.asarray(candidate.get("_mask"), dtype=bool)
    return resize_mask_crop(mask)


def coarse_template_similarity(mask: np.ndarray, project_dir: str | Path) -> dict[str, float]:
    if int(mask.sum()) < 16:
        return {
            "hindwing_template_iou": 0.0,
            "hindwing_template_dice": 0.0,
            "hindwing_template_diff": 1.0,
        }
    template = _cached_base_template_mask(str(Path(project_dir).resolve()))
    sample = resize_mask_crop(mask, size=template.shape[0])
    variants = []
    base = sample.astype(bool)
    for rotation in range(4):
        rotated = np.rot90(base, k=rotation)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    best_iou = 0.0
    best_dice = 0.0
    best_diff = 1.0
    template_bool = template.astype(bool)
    for variant in variants:
        union = float(np.logical_or(template_bool, variant).sum())
        intersection = float(np.logical_and(template_bool, variant).sum())
        iou = 0.0 if union <= 0 else intersection / union
        denom = float(template_bool.sum() + variant.sum())
        dice = 0.0 if denom <= 0 else (2.0 * intersection) / denom
        diff = float(np.mean(np.logical_xor(template_bool, variant)))
        if iou > best_iou:
            best_iou = iou
            best_dice = dice
            best_diff = diff
    return {
        "hindwing_template_iou": float(best_iou),
        "hindwing_template_dice": float(best_dice),
        "hindwing_template_diff": float(best_diff),
    }


def pca_aligned_coordinates(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.asarray([0.0, 0.0]), np.eye(2, dtype=np.float64)
    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center = coords.mean(axis=0)
    centered = coords - center
    cov = np.cov(centered, rowvar=False) if len(coords) > 1 else np.eye(2, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    if eigenvectors[0, 0] < 0:
        eigenvectors[:, 0] *= -1.0
    aligned = centered @ eigenvectors
    return aligned, center, eigenvectors


def summarize_profile(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_cv": 0.0,
            f"{prefix}_q10": 0.0,
            f"{prefix}_q50": 0.0,
            f"{prefix}_q90": 0.0,
        }
    mean = float(values.mean())
    std = float(values.std())
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_cv": 0.0 if abs(mean) < 1e-6 else std / abs(mean),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
    }


def hu_moments(mask: np.ndarray) -> dict[str, float]:
    binary = np.asarray(mask, dtype=np.float64)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return {f"hu_{index}": 0.0 for index in range(1, 8)}
    m00 = float(binary.sum())
    x_bar = float((xs * 1.0).sum() / m00)
    y_bar = float((ys * 1.0).sum() / m00)
    x = xs.astype(np.float64) - x_bar
    y = ys.astype(np.float64) - y_bar

    def mu(p: int, q: int) -> float:
        return float(np.sum((x**p) * (y**q)))

    def eta(p: int, q: int) -> float:
        gamma = 1.0 + (p + q) / 2.0
        return mu(p, q) / max(m00**gamma, 1e-12)

    n20 = eta(2, 0)
    n02 = eta(0, 2)
    n11 = eta(1, 1)
    n30 = eta(3, 0)
    n12 = eta(1, 2)
    n21 = eta(2, 1)
    n03 = eta(0, 3)

    hu = [
        n20 + n02,
        (n20 - n02) ** 2 + 4.0 * (n11**2),
        (n30 - 3.0 * n12) ** 2 + (3.0 * n21 - n03) ** 2,
        (n30 + n12) ** 2 + (n21 + n03) ** 2,
        (n30 - 3.0 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3.0 * (n21 + n03) ** 2)
        + (3.0 * n21 - n03) * (n21 + n03) * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
        (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4.0 * n11 * (n30 + n12) * (n21 + n03),
        (3.0 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3.0 * (n21 + n03) ** 2)
        - (n30 - 3.0 * n12) * (n21 + n03) * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
    ]
    transformed = {}
    for index, value in enumerate(hu, start=1):
        signed_log = 0.0
        if abs(value) > 1e-30:
            signed_log = -math.copysign(1.0, value) * math.log10(abs(value))
        transformed[f"hu_{index}"] = float(signed_log)
    return transformed


def width_profile_without_landmarks(mask: np.ndarray, bins: int = 24) -> np.ndarray:
    aligned, _, _ = pca_aligned_coordinates(mask)
    if aligned.shape[0] == 0:
        return np.zeros(bins, dtype=np.float64)
    x = aligned[:, 0]
    y = aligned[:, 1]
    edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    widths = np.zeros(bins, dtype=np.float64)
    for index in range(bins):
        if index == bins - 1:
            in_bin = (x >= edges[index]) & (x <= edges[index + 1])
        else:
            in_bin = (x >= edges[index]) & (x < edges[index + 1])
        if np.any(in_bin):
            widths[index] = float(y[in_bin].max() - y[in_bin].min())
    return fill_profile(widths)


def contour_fourier_descriptors(mask: np.ndarray, harmonics: int = 6) -> dict[str, float]:
    contour = extract_ordered_contour(mask, n_points=192)
    if contour.shape[0] == 0:
        return {f"fourier_{index}": 0.0 for index in range(1, harmonics + 1)}
    centered = contour - contour.mean(axis=0, keepdims=True)
    signal = centered[:, 0] + 1j * centered[:, 1]
    coeffs = np.fft.fft(signal)
    normalizer = max(float(np.abs(coeffs[1])), 1e-6)
    return {
        f"fourier_{index}": float(np.abs(coeffs[index]) / normalizer) if index < len(coeffs) else 0.0
        for index in range(1, harmonics + 1)
    }


def contour_solidity(mask: np.ndarray) -> float:
    contour = extract_contour_points(mask)
    if contour.shape[0] < 3:
        return 0.0
    try:
        hull = ConvexHull(contour.astype(np.float64))
    except Exception:
        return 0.0
    hull_area = float(hull.volume)
    if hull_area <= 1e-6:
        return 0.0
    return float(mask.sum()) / hull_area


def compute_shape_features(mask: np.ndarray, project_dir: str | Path) -> dict[str, float]:
    ys, xs = np.nonzero(mask)
    contour = extract_contour_points(mask)
    if len(xs) == 0:
        zero = {
            "area_ratio": 0.0,
            "bbox_width": 0.0,
            "bbox_height": 0.0,
            "bbox_aspect": 0.0,
            "extent": 0.0,
            "centroid_x_norm": 0.0,
            "centroid_y_norm": 0.0,
            "major_axis": 0.0,
            "minor_axis": 0.0,
            "axis_ratio": 0.0,
            "eccentricity": 0.0,
            "orientation_deg": 0.0,
            "perimeter": 0.0,
            "circularity": 0.0,
            "solidity": 0.0,
            "notch_depth": 0.0,
        }
        zero.update({f"hu_{index}": 0.0 for index in range(1, 8)})
        zero.update(summarize_profile(np.zeros(1, dtype=np.float64), "radial"))
        zero.update(summarize_profile(np.zeros(1, dtype=np.float64), "width"))
        zero.update({f"fft_radial_{index}": 0.0 for index in range(1, 5)})
        zero.update({f"fourier_{index}": 0.0 for index in range(1, 7)})
        zero.update(coarse_template_similarity(mask, project_dir))
        return zero

    height, width = mask.shape
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    bbox_width = float(max_x - min_x + 1)
    bbox_height = float(max_y - min_y + 1)
    bbox_area = max(1.0, bbox_width * bbox_height)
    area = float(len(xs))
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    aligned, _, basis = pca_aligned_coordinates(mask)
    if aligned.shape[0] > 1:
        cov = np.cov(aligned, rowvar=False)
        eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 0.0)
        eigenvalues = np.sort(eigenvalues)[::-1]
    else:
        eigenvalues = np.asarray([0.0, 0.0], dtype=np.float64)
    major_axis = float(math.sqrt(max(eigenvalues[0], 0.0)) * 4.0)
    minor_axis = float(math.sqrt(max(eigenvalues[1], 0.0)) * 4.0) if len(eigenvalues) > 1 else 0.0
    axis_ratio = major_axis / max(minor_axis, 1e-6)
    eccentricity = math.sqrt(max(0.0, 1.0 - (minor_axis / max(major_axis, 1e-6)) ** 2)) if major_axis > 1e-6 else 0.0
    orientation_deg = float(np.degrees(math.atan2(basis[1, 0], basis[0, 0])))

    center = np.asarray([centroid_x, centroid_y], dtype=np.float64)
    radial_profile = compute_radial_profile(contour.astype(np.float64), center)
    radial_summary = summarize_profile(radial_profile, "radial")
    radial_fft = np.abs(np.fft.rfft(radial_profile))
    radial_fft_norm = max(float(radial_fft[1]) if len(radial_fft) > 1 else 0.0, 1e-6)
    radial_fft_features = {
        f"fft_radial_{index}": float(radial_fft[index] / radial_fft_norm) if index < len(radial_fft) else 0.0
        for index in range(1, 5)
    }

    width_profile = width_profile_without_landmarks(mask)
    width_summary = summarize_profile(width_profile, "width")

    contour_centered = contour.astype(np.float64) - center.reshape(1, 2)
    notch = compute_region_anchors(template_to_canonical(contour.astype(np.float64))) if len(contour) else {}
    posterior_notch = notch.get("posterior_notch", np.asarray([0.0, 0.0], dtype=np.float64))
    notch_depth = float(np.linalg.norm(np.asarray(posterior_notch, dtype=np.float64)))

    perimeter = float(contour.shape[0])
    circularity = 0.0 if perimeter <= 1e-6 else 4.0 * math.pi * area / max(perimeter * perimeter, 1e-6)
    solidity = contour_solidity(mask)

    features = {
        "area_ratio": area / float(height * width),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_aspect": bbox_width / max(1.0, bbox_height),
        "extent": area / bbox_area,
        "centroid_x_norm": centroid_x / max(1.0, float(width)),
        "centroid_y_norm": centroid_y / max(1.0, float(height)),
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "axis_ratio": axis_ratio,
        "eccentricity": float(eccentricity),
        "orientation_deg": orientation_deg,
        "perimeter": perimeter,
        "circularity": float(circularity),
        "solidity": float(solidity),
        "notch_depth": notch_depth,
    }
    features.update(hu_moments(mask))
    features.update(radial_summary)
    features.update(width_summary)
    features.update(radial_fft_features)
    features.update(contour_fourier_descriptors(mask))
    features.update(coarse_template_similarity(mask, project_dir))
    return features


def build_wing_feature_row(
    item_id: str,
    image_path: str | Path,
    project_dir: str | Path,
    source_group: str = "",
    modality: str = "",
    wing_type_label: str = "",
    stage_label: str = "",
    is_other_stage: int = 0,
    split_group: str = "",
) -> dict[str, object]:
    preprocess = extract_best_wing_mask(load_gray_image(image_path))
    mask = np.asarray(preprocess["mask"], dtype=bool)
    row: dict[str, object] = {
        "id": item_id,
        "image_path": str(Path(image_path).resolve()),
        "source_group": source_group,
        "modality": modality or infer_modality(image_path),
        "wing_type_label": wing_type_label,
        "stage_label": stage_label,
        "is_other_stage": int(is_other_stage),
        "split_group": split_group or derive_split_group(item_id, image_path, source_group),
        "mask_threshold": int(preprocess["threshold"]),
        "mask_polarity": str(preprocess["polarity"]),
        "mask_quality": float(preprocess["mask_quality"]),
        "mask_area_pixels": int(mask.sum()),
    }
    row.update(compute_shape_features(mask, project_dir))
    return row


def add_shape_features_to_stage_row(row: dict[str, str], project_dir: str | Path) -> dict[str, object]:
    item_id = str(row.get("id", "")).strip()
    image_path = str(row.get("image_path", "")).strip()
    if not image_path:
        raise ValueError("Stage feature row missing image_path")
    preprocess = extract_best_wing_mask(load_gray_image(image_path))
    mask = np.asarray(preprocess["mask"], dtype=bool)
    feature_row: dict[str, object] = dict(row)
    feature_row["split_group"] = derive_split_group(item_id, image_path, "hindwing_stage")
    feature_row["modality"] = infer_modality(image_path)
    feature_row["mask_threshold"] = int(preprocess["threshold"])
    feature_row["mask_polarity"] = str(preprocess["polarity"])
    feature_row["mask_quality"] = float(preprocess["mask_quality"])
    for key, value in compute_shape_features(mask, project_dir).items():
        feature_row[f"shape_{key}"] = value
    return feature_row


def stage_feature_numeric_columns(rows: list[dict[str, object]]) -> list[str]:
    excluded = {
        "id",
        "true_label",
        "image_path",
        "best_template_label",
        "stage_label",
        "wing_type_label",
        "source_group",
        "modality",
        "mask_polarity",
        "split_group",
        "is_other_stage",
        "review_reason",
        "decision",
    }
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in excluded or key in seen:
                continue
            try:
                float(value)  # type: ignore[arg-type]
            except Exception:
                continue
            seen.add(key)
            columns.append(key)
    return columns


def wing_feature_numeric_columns(rows: list[dict[str, object]]) -> list[str]:
    excluded = {
        "id",
        "image_path",
        "source_group",
        "modality",
        "wing_type_label",
        "stage_label",
        "mask_polarity",
        "split_group",
        "decision",
        "review_reason",
    }
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in excluded or key in seen:
                continue
            try:
                float(value)  # type: ignore[arg-type]
            except Exception:
                continue
            seen.add(key)
            columns.append(key)
    return columns


def rows_to_matrix(rows: list[dict[str, object]], feature_names: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for feature_index, feature_name in enumerate(feature_names):
            matrix[row_index, feature_index] = safe_float(row.get(feature_name, 0.0), 0.0)
    return matrix


def load_training_manifest_rows(
    gt100_cleaning_csv: str | Path = DEFAULT_GT100_CLEANING_CSV,
    stage_project_manifest_csv: str | Path | None = None,
    other_stage_manifest_csv: str | Path | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    gt100_rows = read_csv_rows(gt100_cleaning_csv)
    for row in gt100_rows:
        if str(row.get("quality_status", "")).strip().lower() != "usable":
            continue
        image_path = str(row.get("image_path", "")).strip()
        wing_type = str(row.get("wing_type", "")).strip().lower()
        stage = str(row.get("stage", "")).strip()
        if not image_path or wing_type not in {"forewing", "hindwing"}:
            continue
        item_id = str(row.get("id", "")).strip() or Path(image_path).stem
        rows.append(
            {
                "id": item_id,
                "image_path": image_path,
                "source_group": "gt100",
                "modality": infer_modality(image_path),
                "wing_type_label": wing_type,
                "stage_label": stage if stage in LABELS else "",
                "is_other_stage": 0,
                "split_group": derive_split_group(item_id, image_path, "gt100"),
            }
        )

    if stage_project_manifest_csv:
        for row in read_csv_rows(stage_project_manifest_csv):
            image_path = str(row.get("image_path", "")).strip()
            label = str(row.get("label", "")).strip()
            if not image_path or label not in LABELS:
                continue
            item_id = str(row.get("id", "")).strip() or Path(image_path).stem
            rows.append(
                {
                    "id": item_id,
                    "image_path": image_path,
                    "source_group": "hindwing_landmark_project",
                    "modality": infer_modality(image_path),
                    "wing_type_label": "hindwing",
                    "stage_label": label,
                    "is_other_stage": 0,
                    "split_group": derive_split_group(item_id, image_path, "hindwing_landmark_project"),
                }
            )

    if other_stage_manifest_csv:
        for row in read_csv_rows(other_stage_manifest_csv):
            image_path = str(row.get("image_path", "")).strip()
            if not image_path:
                continue
            item_id = str(row.get("id", "")).strip() or Path(image_path).stem
            source_group = str(row.get("source_group", "")).strip() or "other_stage"
            rows.append(
                {
                    "id": item_id,
                    "image_path": image_path,
                    "source_group": source_group,
                    "modality": str(row.get("modality", "")).strip() or infer_modality(image_path),
                    "wing_type_label": str(row.get("wing_type_label", "")).strip().lower(),
                    "stage_label": "",
                    "is_other_stage": 1,
                    "split_group": str(row.get("split_group", "")).strip()
                    or derive_split_group(item_id, image_path, source_group),
                }
            )

    return rows


def prepare_landmark_input_image(gray: np.ndarray, mask: np.ndarray, image_mode: str = "raw") -> Image.Image:
    normalized = percentile_normalize(gray)
    mode = str(image_mode or "raw").strip().lower()
    gray_rgb = np.stack([normalized, normalized, normalized], axis=-1)
    if mode in {"", "raw"}:
        return Image.fromarray(gray_rgb, mode="RGB")
    if mode == "normalized":
        return Image.fromarray(gray_rgb, mode="RGB")
    mask_bool = np.asarray(mask, dtype=bool)
    boundary = mask_bool ^ ndi.binary_erosion(mask_bool, structure=np.ones((3, 3), dtype=bool))
    if mode == "mask":
        mask_img = (mask_bool.astype(np.uint8) * 255)
        rgb = np.stack([mask_img, mask_img, mask_img], axis=-1)
        return Image.fromarray(rgb, mode="RGB")
    if mode == "mask_overlay":
        rgb = gray_rgb.astype(np.float32)
        rgb[mask_bool, 1] = np.maximum(rgb[mask_bool, 1], 220.0)
        rgb[mask_bool, 0] *= 0.65
        rgb[boundary, 0] = 255.0
        rgb[boundary, 1] = 80.0
        rgb[boundary, 2] = 80.0
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    raise ValueError(f"Unsupported image mode: {image_mode}")


def build_stage_feature_row_from_landmarks(
    item_id: str,
    image_path: str | Path,
    landmarks_xy: np.ndarray,
    project_dir: str | Path,
) -> dict[str, object]:
    image_path = Path(image_path)
    preprocess = extract_best_wing_mask(load_gray_image(image_path))
    gray = np.asarray(preprocess["normalized_gray"], dtype=np.uint8)
    mask = np.asarray(preprocess["mask"], dtype=bool)
    contour = extract_contour_points(mask)
    landmarks = np.asarray(landmarks_xy, dtype=np.float64)
    if contour.shape[0] == 0 or landmarks.shape != (4, 2):
        raise ValueError(f"Invalid stage feature inputs for {image_path}")

    base_template, stage_templates = load_stage_templates(Path(project_dir))
    base_rotation, base_translation, base_scale = umeyama_similarity(landmarks, base_template["_landmarks"])
    base_aligned_image = transform_image_similarity(gray, base_rotation, base_translation, base_scale)
    base_aligned_mask = transform_mask_similarity(mask, base_rotation, base_translation, base_scale)
    base_aligned_landmarks = apply_similarity(landmarks, base_rotation, base_translation, base_scale)
    base_aligned_contour = apply_similarity(contour, base_rotation, base_translation, base_scale)

    canonical_contour = template_to_canonical(base_aligned_contour)
    region_anchors = compute_region_anchors(canonical_contour)
    source_region_anchors = {
        name: np.asarray([[1536.0 - 1400.0 * point[0], 1024.0 + 1400.0 * point[1]]], dtype=np.float64)[0]
        for name, point in region_anchors.items()
    }
    source_controls = np.vstack(
        [
            base_aligned_landmarks,
            np.vstack(
                [
                    source_region_anchors[name]
                    for name in ["proximal", "distal_outer_margin", "posterior_notch", "dorsal_ventral_transition"]
                ]
            ),
        ]
    )

    per_stage: dict[str, dict[str, float]] = {}
    for template_label, template_payload in stage_templates.items():
        destination_controls = np.vstack(
            [
                template_payload["_landmarks"],
                np.vstack(
                    [
                        template_payload["_region_anchors"][name]
                        for name in ["proximal", "distal_outer_margin", "posterior_notch", "dorsal_ventral_transition"]
                    ]
                ),
            ]
        )
        _, tps_energy = tps_warp(base_aligned_image, source_controls, destination_controls, order=1, energy_only=True)
        metrics = build_registration_features(base_aligned_mask, base_aligned_contour, base_aligned_landmarks, template_payload)
        metrics["tps_bending_energy"] = float(tps_energy)
        metrics["overall_template_cost"] = float(
            0.25 * metrics["landmark_similarity_error"]
            + 0.25 * metrics["tps_bending_energy"]
            + 0.20 * metrics["radial_profile_l2"]
            + 0.15 * metrics["width_profile_l2"]
            + 0.10 * metrics["notch_offset"]
            + 0.05 * (1.0 - metrics["mask_iou_to_template"])
        )
        per_stage[template_label] = metrics

    row = make_feature_row(item_id, "", str(image_path.resolve()), per_stage)
    row["mask_threshold"] = int(preprocess["threshold"])
    row["mask_polarity"] = str(preprocess["polarity"])
    row["mask_quality"] = float(preprocess["mask_quality"])
    row["modality"] = infer_modality(image_path)
    row["split_group"] = derive_split_group(item_id, image_path, "prediction")
    for key, value in compute_shape_features(mask, project_dir).items():
        row[f"shape_{key}"] = value
    return row


def landmarks_array_from_prediction_row(row: dict[str, object]) -> np.ndarray | None:
    points: list[list[float]] = []
    for name in LANDMARK_NAMES:
        x_value = row.get(f"{name}_x", "")
        y_value = row.get(f"{name}_y", "")
        if x_value in ("", None) or y_value in ("", None):
            return None
        points.append([safe_float(x_value), safe_float(y_value)])
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (4, 2):
        return None
    return array


def write_review_contact_sheet(
    rows: list[dict[str, object]],
    output_path: str | Path,
    title: str = "Review Contact Sheet",
    cell_size: int = 220,
    columns: int = 4,
) -> Path:
    output_path = Path(output_path)
    if not rows:
        blank = Image.new("RGB", (cell_size * max(1, columns), cell_size + 60), (18, 18, 22))
        draw = ImageDraw.Draw(blank)
        draw.text((20, 20), f"{title}: no review items", fill="white")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        blank.save(output_path)
        return output_path

    grouped = sorted(
        rows,
        key=lambda row: (
            str(row.get("review_reason", "")),
            str(row.get("decision", "")),
            str(row.get("id", "")),
        ),
    )
    total_rows = int(math.ceil(len(grouped) / max(1, columns)))
    canvas = Image.new("RGB", (columns * cell_size, total_rows * cell_size + 70), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 18), title, fill="white")

    for index, row in enumerate(grouped):
        image_path = str(row.get("image_path", "")).strip()
        reason = str(row.get("review_reason", "")).strip() or str(row.get("decision", "")).strip()
        sample_id = str(row.get("id", "")).strip()
        x0 = (index % columns) * cell_size
        y0 = 60 + (index // columns) * cell_size
        tile = Image.new("RGB", (cell_size, cell_size), (26, 26, 32))
        if image_path and Path(image_path).exists():
            with Image.open(image_path) as image:
                preview = image.convert("RGB")
                preview.thumbnail((cell_size - 12, cell_size - 42))
                tile.paste(preview, ((cell_size - preview.width) // 2, 6))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((8, cell_size - 34), sample_id[:28], fill="white")
        tile_draw.text((8, cell_size - 18), reason[:30], fill=(255, 190, 128))
        canvas.paste(tile, (x0, y0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def summarize_prediction_counts(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("decision", "")).strip() or "unknown"
        counts[decision] = counts.get(decision, 0) + 1
    return counts


def load_embedding_baseline_rows() -> list[dict[str, str]]:
    path = ROOT / "output_results" / "outputs_stage_classifier" / "model_comparison.csv"
    return read_csv_rows(path)


def save_json(path: str | Path, payload: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    if not rows:
        return write_csv_rows(path, ["empty"], [{"empty": ""}])
    return write_csv_rows(path, list(rows[0].keys()), rows)
