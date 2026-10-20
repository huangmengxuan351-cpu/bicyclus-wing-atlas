from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from aligned_wing_classifier_utils import build_aligned_wing_feature_row, prototype_hind_probability, row_matrix
from forewing_stage_classifier_utils import apply_forewing_stage_bundle
from hindwing_stage_classifier_utils import apply_hindwing_stage_bundle
from forewing_stage_template_utils import DEFAULT_PROJECT_DIR as DEFAULT_FOREWING_PROJECT_DIR
from forewing_stage_template_utils import LABELS as STAGE_LABELS
from forewing_stage_template_utils import project_paths as forewing_project_paths
from pose_landmark_utils import DEFAULT_PROJECT_DIR as DEFAULT_HINDWING_PROJECT_DIR
from pose_landmark_utils import IMAGE_SUFFIXES, read_csv_rows, write_csv_rows
from register_with_stage_templates import build_registration_features, load_template
from shape_priority_hindwing_utils import build_wing_feature_row, compute_shape_features, extract_best_wing_mask
from stage_template_geometry import (
    apply_similarity,
    compute_notch_features,
    compute_region_anchors,
    convex_hull_points,
    extract_contour_points,
    extract_ordered_contour,
    extract_wing_mask,
    from_canonical,
    template_to_canonical,
    to_canonical,
    tps_warp,
    transform_image_similarity,
    transform_mask_similarity,
    umeyama_similarity,
)
from train_shape_priority_hindwing_classifier import apply_binary_probability_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "output_results" / "dapi_wing_stage_classifier"
DEFAULT_FOREWING_REFERENCE_ROOT = ROOT / "output_results" / "dapi_forewing_standardized_manual_exported"
DEFAULT_HINDWING_REFERENCE_SUMMARY = ROOT / "output_results" / "gt100_hindwing_dapi_manual_review_by_stage" / "review_alignment_export_summary.csv"
DEFAULT_WING_MODEL_DIR = ROOT / "output_results" / "shape_priority_no_other_stage_conf095"
LABEL_TO_MATURITY = {"0.5": 0.0, "0.75": 0.5, "1.0": 1.0}
FEATURE_NAMES = [
    "area_ratio",
    "bbox_aspect",
    "extent",
    "eccentricity",
    "circularity",
    "solidity",
    "notch_depth",
    "radial_mean",
    "radial_std",
    "radial_q10",
    "radial_q50",
    "radial_q90",
    "width_mean",
    "width_std",
    "width_q10",
    "width_q50",
    "width_q90",
    "hu_1",
    "hu_2",
    "hu_3",
    "hu_4",
    "fft_radial_2",
    "fft_radial_3",
    "fourier_2",
    "fourier_3",
    "protrusion_rate",
    "protrusion_radius",
    "protrusion_depth",
    "left_margin_curvature_peak",
]


@dataclass(frozen=True)
class FamilyConfig:
    wing_type: str
    project_dir: Path


@dataclass(frozen=True)
class ReferenceEntry:
    ref_id: str
    stage: str
    wing_type: str
    image_path: Path
    mask_path: Path
    source: str
    features: tuple[float, ...]
    mask: np.ndarray


@dataclass(frozen=True)
class CandidateScore:
    wing_type: str
    stage: str
    template_cost: float
    family_prior_cost: float
    feature_distance: float
    nn_distance: float
    topk_distance: float
    reference_count: int
    nearest_reference_id: str
    nearest_reference_source: str
    nearest_reference_path: str
    normalized_template_cost: float = 1.0
    normalized_feature_distance: float = 1.0
    normalized_nn_distance: float = 1.0
    combined_cost: float = 1.0
    class_probability: float = 0.0


FAMILY_CONFIGS = {
    "forewing": FamilyConfig("forewing", DEFAULT_FOREWING_PROJECT_DIR.resolve()),
    "hindwing": FamilyConfig("hindwing", DEFAULT_HINDWING_PROJECT_DIR.resolve()),
}


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_wing_model_bundle() -> dict[str, object] | None:
    candidate_paths = [
        ROOT / "output_results" / "stage_upgrade_v1" / "aligned_wing_type_classifier" / "wing_model_bundle.pkl",
        ROOT / "output_results" / "aligned_wing_type_classifier" / "wing_model_bundle.pkl",
        DEFAULT_WING_MODEL_DIR / "wing_model_bundle.pkl",
    ]
    for bundle_path in candidate_paths:
        if not bundle_path.exists():
            continue
        with bundle_path.open("rb") as handle:
            return pickle.load(handle)
    return None


@lru_cache(maxsize=1)
def load_hindwing_stage_model_bundle() -> dict[str, object] | None:
    candidate_paths = [
        ROOT / "output_results" / "stage_upgrade_v1" / "hindwing_stage_classifier" / "stage_model_bundle.pkl",
        ROOT / "output_results" / "hindwing_stage_classifier" / "stage_model_bundle.pkl",
    ]
    for bundle_path in candidate_paths:
        if not bundle_path.exists():
            continue
        with bundle_path.open("rb") as handle:
            return pickle.load(handle)
    return None


@lru_cache(maxsize=1)
def load_forewing_stage_model_bundle() -> dict[str, object] | None:
    candidate_paths = [
        ROOT / "output_results" / "stage_upgrade_v1_1" / "forewing_stage_classifier" / "stage_model_bundle.pkl",
        ROOT / "output_results" / "stage_upgrade_v1" / "forewing_stage_classifier" / "stage_model_bundle.pkl",
        ROOT / "output_results" / "forewing_stage_classifier" / "stage_model_bundle.pkl",
    ]
    for bundle_path in candidate_paths:
        if not bundle_path.exists():
            continue
        with bundle_path.open("rb") as handle:
            return pickle.load(handle)
    return None


def apply_aligned_wing_bundle(
    bundle: dict[str, object],
    item_id: str,
    image_path: Path,
    mask_path: Path | None,
    gray_override: np.ndarray | None = None,
    mask_override: np.ndarray | None = None,
    mask_source_override: str = "",
) -> tuple[str, float, float]:
    feature_row = build_aligned_wing_feature_row(
        item_id=item_id,
        image_path=str(image_path.resolve()),
        mask_path=str(mask_path.resolve()) if mask_path is not None else None,
        source_group="dapi_classifier",
        wing_type_label="",
        stage_label="",
        modality="dapi",
        split_group=f"inference::{item_id.lower()}",
        include_mask_vectors=True,
        gray_override=gray_override,
        mask_override=mask_override,
        mask_source_override=mask_source_override,
    )
    bundle_kind = str(bundle.get("bundle_kind", "")).strip().lower()
    if bundle_kind == "aligned_mask_logistic":
        feature_names = list(bundle.get("feature_names", []))
        probability = float(bundle["model"].predict_proba(row_matrix([feature_row], feature_names))[:, 1][0])
    elif bundle_kind == "aligned_mask_prototype":
        mask_size = int(bundle.get("mask_size", 48))
        prefix = f"mask{mask_size}_px_"
        feature_names = [key for key in feature_row.keys() if key.startswith(prefix)]
        matrix = row_matrix([feature_row], feature_names)
        probability = float(
            prototype_hind_probability(
                matrix[0],
                np.asarray(bundle.get("forewing_prototype", []), dtype=np.float64),
                np.asarray(bundle.get("hindwing_prototype", []), dtype=np.float64),
            )
        )
    else:
        raise ValueError(f"Unsupported aligned wing bundle kind: {bundle_kind}")
    prediction = "hindwing" if probability >= 0.5 else "forewing"
    confidence = float(max(probability, 1.0 - probability))
    return prediction, confidence, probability


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    shifted = values - float(values.max())
    exp = np.exp(shifted)
    total = float(exp.sum())
    if total <= 0.0:
        return np.full(values.shape, 1.0 / max(1, values.size), dtype=np.float64)
    return exp / total


def dice_score(left: np.ndarray, right: np.ndarray) -> float:
    left_bool = np.asarray(left, dtype=bool)
    right_bool = np.asarray(right, dtype=bool)
    denom = float(left_bool.sum() + right_bool.sum())
    if denom <= 0.0:
        return 0.0
    intersection = float(np.logical_and(left_bool, right_bool).sum())
    return (2.0 * intersection) / denom


def minmax_normalize(values: list[float], higher_is_worse: bool = True) -> list[float]:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return [1.0 for _ in values]
    lo = min(finite)
    hi = max(finite)
    if hi <= lo + 1e-9:
        return [0.0 for _ in values]
    normalized = []
    for value in values:
        if not np.isfinite(value):
            normalized.append(1.0 if higher_is_worse else 0.0)
            continue
        scaled = (value - lo) / (hi - lo)
        normalized.append(float(scaled if higher_is_worse else 1.0 - scaled))
    return normalized


def stage_numeric(label: str) -> float:
    return LABEL_TO_MATURITY.get(label, 0.0)


def derive_item_id(image_path: Path) -> str:
    generic_stems = {"standardized_dapi_gray", "review_aligned_gt100", "manual_aligned_dapi_gray"}
    if image_path.stem.lower() in generic_stems and image_path.parent.name:
        return image_path.parent.name
    return image_path.stem


def safe_output_name(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return "item"
    for token in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        cleaned = cleaned.replace(token, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip(" ._") or "item"


def pca_sign_aligned_contour(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contour = np.asarray(contour, dtype=np.float64)
    if len(contour) == 0:
        return contour, np.asarray([0.0, 0.0], dtype=np.float64), np.eye(2, dtype=np.float64)
    center = contour.mean(axis=0)
    centered = contour - center
    cov = np.cov(centered, rowvar=False) if len(contour) > 1 else np.eye(2, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]
    if basis[0, 0] < 0:
        basis[:, 0] *= -1.0
    local = centered @ basis
    x_values = local[:, 0]
    high_band = local[x_values >= np.quantile(x_values, 0.88)]
    low_band = local[x_values <= np.quantile(x_values, 0.12)]
    high_span = float(high_band[:, 1].max() - high_band[:, 1].min()) if len(high_band) else 0.0
    low_span = float(low_band[:, 1].max() - low_band[:, 1].min()) if len(low_band) else 0.0
    if low_span > high_span:
        local[:, 0] *= -1.0
        basis[:, 0] *= -1.0
    return local, center, basis


def contour_point_from_local(local_point: np.ndarray, center: np.ndarray, basis: np.ndarray, contour: np.ndarray) -> np.ndarray:
    image_point = np.asarray(local_point, dtype=np.float64) @ basis.T + center
    distances = np.linalg.norm(contour - image_point.reshape(1, 2), axis=1)
    return contour[int(np.argmin(distances))].astype(np.float64)


def infer_posterior_notch(contour: np.ndarray, base_dorsal: np.ndarray, base_ventral: np.ndarray, distal_tip: np.ndarray) -> np.ndarray:
    temporary_landmarks = np.asarray(
        [
            base_dorsal.astype(np.float64),
            base_ventral.astype(np.float64),
            distal_tip.astype(np.float64),
            base_ventral.astype(np.float64),
        ],
        dtype=np.float64,
    )
    contour_canonical = to_canonical(contour, temporary_landmarks)
    hull_canonical = convex_hull_points(contour_canonical)
    notch_info = compute_notch_features(contour_canonical, hull_canonical)
    notch_canonical = np.asarray(notch_info["notch_point"], dtype=np.float64)
    candidate_sector = contour_canonical[
        (contour_canonical[:, 1] >= 0.02)
        & (contour_canonical[:, 0] >= -0.25)
        & (contour_canonical[:, 0] <= 0.55)
    ]
    if len(candidate_sector):
        score = 1.6 * candidate_sector[:, 1] - 0.4 * np.abs(candidate_sector[:, 0] - np.median(candidate_sector[:, 0]))
        fallback = candidate_sector[int(np.argmax(score))]
    else:
        fallback = np.asarray([0.05, 0.18], dtype=np.float64)
    notch_distance = float(np.linalg.norm(notch_canonical - to_canonical(base_ventral.reshape(1, 2), temporary_landmarks)[0]))
    if (
        notch_distance < 0.08
        or notch_canonical[1] < 0.01
        or notch_canonical[0] < -0.30
        or notch_canonical[0] > 0.55
    ):
        notch_canonical = fallback
    notch_image = from_canonical(notch_canonical.reshape(1, 2), temporary_landmarks)[0]
    distances = np.linalg.norm(contour - notch_image.reshape(1, 2), axis=1)
    return contour[int(np.argmin(distances))].astype(np.float64)


def infer_landmarks_from_mask(mask: np.ndarray) -> np.ndarray:
    contour = extract_contour_points(mask, n_points=512)
    if len(contour) < 32:
        raise ValueError("Mask contour too small for landmark inference.")
    contour_local, center, basis = pca_sign_aligned_contour(contour)
    x_values = contour_local[:, 0]
    base_points = contour_local[x_values >= np.quantile(x_values, 0.90)]
    if len(base_points) < 8:
        base_points = contour_local[x_values >= np.quantile(x_values, 0.82)]
    if len(base_points) < 4:
        raise ValueError("Unable to locate base band from contour.")
    base_dorsal_local = base_points[int(np.argmin(base_points[:, 1]))]
    base_ventral_local = base_points[int(np.argmax(base_points[:, 1]))]
    base_center_local = (base_dorsal_local + base_ventral_local) / 2.0
    distal_candidates = contour_local[x_values <= np.quantile(x_values, 0.10)]
    if len(distal_candidates) == 0:
        distal_candidates = contour_local
    distances = np.linalg.norm(distal_candidates - base_center_local.reshape(1, 2), axis=1)
    distal_tip_local = distal_candidates[int(np.argmax(distances))]
    base_dorsal = contour_point_from_local(base_dorsal_local, center, basis, contour)
    base_ventral = contour_point_from_local(base_ventral_local, center, basis, contour)
    distal_tip = contour_point_from_local(distal_tip_local, center, basis, contour)
    posterior_notch = infer_posterior_notch(contour, base_dorsal, base_ventral, distal_tip)
    return np.asarray([base_dorsal, base_ventral, distal_tip, posterior_notch], dtype=np.float64)


def outer_margin_curvature(canonical_contour: np.ndarray) -> float:
    contour = np.asarray(canonical_contour, dtype=np.float64)
    if len(contour) < 8:
        return 0.0
    x_values = contour[:, 0]
    outer_mask = x_values >= np.quantile(x_values, 0.82)
    if int(outer_mask.sum()) < 6:
        outer_mask = x_values >= np.quantile(x_values, 0.72)
    prev_points = np.roll(contour, 1, axis=0)
    next_points = np.roll(contour, -1, axis=0)
    vec_prev = contour - prev_points
    vec_next = next_points - contour
    denom = np.linalg.norm(vec_prev, axis=1) * np.linalg.norm(vec_next, axis=1)
    denom = np.where(denom <= 1e-9, 1.0, denom)
    dot = np.sum(vec_prev * vec_next, axis=1) / denom
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.abs(np.arccos(dot))
    values = angles[outer_mask]
    if values.size == 0:
        return 0.0
    return float(values.max())


def compute_protrusion_features_from_aligned_mask(mask: np.ndarray) -> dict[str, float]:
    contour = extract_ordered_contour(mask, n_points=256)
    if len(contour) == 0:
        return {
            "protrusion_rate": 0.0,
            "protrusion_radius": 0.0,
            "protrusion_depth": 0.0,
            "left_margin_curvature_peak": 0.0,
        }
    contour_canonical = template_to_canonical(contour.astype(np.float64))
    ys, xs = np.nonzero(mask)
    pixels_canonical = template_to_canonical(np.column_stack([xs.astype(np.float64), ys.astype(np.float64)]))
    if len(pixels_canonical) == 0:
        return {
            "protrusion_rate": 0.0,
            "protrusion_radius": 0.0,
            "protrusion_depth": 0.0,
            "left_margin_curvature_peak": 0.0,
        }
    x_pixels = pixels_canonical[:, 0]
    x_contour = contour_canonical[:, 0]
    q85 = float(np.quantile(x_pixels, 0.85))
    q95 = float(np.quantile(x_contour, 0.95))
    q75 = float(np.quantile(x_contour, 0.75))
    protrusion_rate = float(np.mean(x_pixels >= q85))
    protrusion_radius = float(np.max(x_contour))
    protrusion_depth = float(max(0.0, q95 - q75))
    return {
        "protrusion_rate": protrusion_rate,
        "protrusion_radius": protrusion_radius,
        "protrusion_depth": protrusion_depth,
        "left_margin_curvature_peak": outer_margin_curvature(contour_canonical),
    }


def aligned_mask_feature_map(mask: np.ndarray, project_dir: Path) -> dict[str, float]:
    features = compute_shape_features(mask, project_dir)
    features.update(compute_protrusion_features_from_aligned_mask(mask))
    return features


def vector_from_feature_map(feature_map: dict[str, float]) -> np.ndarray:
    return np.asarray([safe_float(feature_map.get(name, 0.0), 0.0) for name in FEATURE_NAMES], dtype=np.float64)


def family_prior_cost_from_raw_mask(mask: np.ndarray, project_dir: Path) -> float:
    feature_map = compute_shape_features(mask, project_dir)
    iou = safe_float(feature_map.get("hindwing_template_iou", 0.0), 0.0)
    dice = safe_float(feature_map.get("hindwing_template_dice", 0.0), 0.0)
    diff = safe_float(feature_map.get("hindwing_template_diff", 1.0), 1.0)
    return float(0.45 * (1.0 - dice) + 0.35 * (1.0 - iou) + 0.20 * diff)


@lru_cache(maxsize=1)
def family_assets() -> dict[str, dict[str, object]]:
    hind_paths = {
        "base": DEFAULT_HINDWING_PROJECT_DIR / "template_system" / "base_template.json",
        "0.5": DEFAULT_HINDWING_PROJECT_DIR / "template_system" / "stage_template_0_5.json",
        "0.75": DEFAULT_HINDWING_PROJECT_DIR / "template_system" / "stage_template_0_75.json",
        "1.0": DEFAULT_HINDWING_PROJECT_DIR / "template_system" / "stage_template_1_0.json",
    }
    fore_paths_root = forewing_project_paths(DEFAULT_FOREWING_PROJECT_DIR)
    fore_paths = {
        "base": fore_paths_root["base_template_json"],
        "0.5": fore_paths_root["stage_template_0_5_json"],
        "0.75": fore_paths_root["stage_template_0_75_json"],
        "1.0": fore_paths_root["stage_template_1_0_json"],
    }
    outputs: dict[str, dict[str, object]] = {}
    for wing_type, paths in {"hindwing": hind_paths, "forewing": fore_paths}.items():
        outputs[wing_type] = {
            "base_template": load_template(Path(paths["base"])),
            "stage_templates": {label: load_template(Path(paths[label])) for label in STAGE_LABELS},
        }
    return outputs


def forewing_reference_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    root = DEFAULT_FOREWING_REFERENCE_ROOT
    if not root.exists():
        return rows
    for label in STAGE_LABELS:
        stage_dir = root / f"stage_{label.replace('.', '_')}"
        if not stage_dir.exists():
            continue
        for sample_dir in sorted(path for path in stage_dir.iterdir() if path.is_dir()):
            image_path = sample_dir / "standardized_dapi_gray.png"
            mask_path = sample_dir / "standardized_wing_mask.png"
            if not image_path.exists() or not mask_path.exists():
                continue
            rows.append(
                {
                    "id": sample_dir.name,
                    "stage": label,
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "source": "manual_standardized_forewing",
                }
            )
    return rows


def hindwing_reference_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv_rows(DEFAULT_HINDWING_REFERENCE_SUMMARY):
        stage = str(row.get("stage", "")).strip()
        status = str(row.get("status", "")).strip().lower()
        manual_status = str(row.get("manual_review_status", "")).strip().lower()
        image_path = str(row.get("review_aligned_gt100", "")).strip()
        mask_path = str(row.get("review_aligned_wing_mask", "")).strip()
        if stage not in STAGE_LABELS or status != "exported":
            continue
        if manual_status not in {"accepted", "adjusted"}:
            continue
        if not image_path or not mask_path:
            continue
        rows.append(
            {
                "id": str(row.get("id", "")).strip(),
                "stage": stage,
                "image_path": image_path,
                "mask_path": mask_path,
                "source": "manual_review_hindwing",
            }
        )
    return rows


@lru_cache(maxsize=1)
def reference_bank() -> dict[str, dict[str, list[ReferenceEntry]]]:
    families = family_assets()
    outputs: dict[str, dict[str, list[ReferenceEntry]]] = {wing_type: {label: [] for label in STAGE_LABELS} for wing_type in FAMILY_CONFIGS}
    family_rows = {
        "forewing": forewing_reference_rows(),
        "hindwing": hindwing_reference_rows(),
    }
    for wing_type, config in FAMILY_CONFIGS.items():
        project_dir = config.project_dir
        for row in family_rows[wing_type]:
            stage = str(row["stage"])
            mask = load_mask(Path(row["mask_path"]))
            features = vector_from_feature_map(aligned_mask_feature_map(mask, project_dir))
            outputs[wing_type][stage].append(
                ReferenceEntry(
                    ref_id=str(row["id"]),
                    stage=stage,
                    wing_type=wing_type,
                    image_path=Path(row["image_path"]).resolve(),
                    mask_path=Path(row["mask_path"]).resolve(),
                    source=str(row["source"]),
                    features=tuple(float(value) for value in features.tolist()),
                    mask=mask.astype(bool),
                )
            )
        stage_templates = families[wing_type]["stage_templates"]
        for stage in STAGE_LABELS:
            template_mask = np.asarray(stage_templates[stage]["_mask"], dtype=bool)
            features = vector_from_feature_map(aligned_mask_feature_map(template_mask, project_dir))
            outputs[wing_type][stage].append(
                ReferenceEntry(
                    ref_id=f"{wing_type}_template_{stage}",
                    stage=stage,
                    wing_type=wing_type,
                    image_path=Path(stage_templates[stage].get("_path", "")),
                    mask_path=Path(stage_templates[stage].get("mask_path", "")),
                    source="stage_template_mask",
                    features=tuple(float(value) for value in features.tolist()),
                    mask=template_mask,
                )
            )
    return outputs


@lru_cache(maxsize=1)
def global_feature_scales() -> np.ndarray:
    all_vectors: list[np.ndarray] = []
    for per_stage in reference_bank().values():
        for refs in per_stage.values():
            for ref in refs:
                all_vectors.append(np.asarray(ref.features, dtype=np.float64))
    if not all_vectors:
        return np.ones(len(FEATURE_NAMES), dtype=np.float64)
    matrix = np.vstack(all_vectors).astype(np.float64)
    scale = matrix.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return scale.astype(np.float64)


def align_sample_to_family(gray: np.ndarray, mask: np.ndarray, wing_type: str) -> dict[str, object]:
    assets = family_assets()[wing_type]
    base_template = assets["base_template"]
    landmarks = infer_landmarks_from_mask(mask)
    contour = extract_contour_points(mask)
    rotation, translation, scale = umeyama_similarity(landmarks, np.asarray(base_template["_landmarks"], dtype=np.float64))
    aligned_image = transform_image_similarity(gray, rotation, translation, scale)
    aligned_mask = transform_mask_similarity(mask, rotation, translation, scale)
    aligned_landmarks = apply_similarity(landmarks, rotation, translation, scale)
    aligned_contour = apply_similarity(contour, rotation, translation, scale)
    canonical_contour = template_to_canonical(aligned_contour)
    region_anchors = compute_region_anchors(canonical_contour)
    source_region_anchors = {
        name: np.asarray([[1536.0 - 1400.0 * point[0], 1024.0 + 1400.0 * point[1]]], dtype=np.float64)[0]
        for name, point in region_anchors.items()
    }
    source_controls = np.vstack(
        [
            aligned_landmarks,
            np.vstack(
                [
                    source_region_anchors[name]
                    for name in ["proximal", "distal_outer_margin", "posterior_notch", "dorsal_ventral_transition"]
                ]
            ),
        ]
    )
    aligned_feature_vector = vector_from_feature_map(aligned_mask_feature_map(aligned_mask, FAMILY_CONFIGS[wing_type].project_dir))
    return {
        "aligned_image": aligned_image,
        "aligned_mask": aligned_mask,
        "aligned_landmarks": aligned_landmarks,
        "aligned_contour": aligned_contour,
        "source_controls": source_controls,
        "feature_vector": aligned_feature_vector,
    }


def template_stage_cost(alignment: dict[str, object], wing_type: str, stage: str) -> float:
    stage_template = family_assets()[wing_type]["stage_templates"][stage]
    destination_controls = np.vstack(
        [
            np.asarray(stage_template["_landmarks"], dtype=np.float64),
            np.vstack(
                [
                    np.asarray(stage_template["_region_anchors"][name], dtype=np.float64)
                    for name in ["proximal", "distal_outer_margin", "posterior_notch", "dorsal_ventral_transition"]
                ]
            ),
        ]
    )
    _, tps_energy = tps_warp(
        np.asarray(alignment["aligned_image"], dtype=np.uint8),
        np.asarray(alignment["source_controls"], dtype=np.float64),
        destination_controls,
        order=1,
        energy_only=True,
    )
    metrics = build_registration_features(
        np.asarray(alignment["aligned_mask"], dtype=bool),
        np.asarray(alignment["aligned_contour"], dtype=np.float64),
        np.asarray(alignment["aligned_landmarks"], dtype=np.float64),
        stage_template,
    )
    metrics["tps_bending_energy"] = float(tps_energy)
    return float(
        0.25 * metrics["landmark_similarity_error"]
        + 0.25 * metrics["tps_bending_energy"]
        + 0.20 * metrics["radial_profile_l2"]
        + 0.15 * metrics["width_profile_l2"]
        + 0.10 * metrics["notch_offset"]
        + 0.05 * (1.0 - metrics["mask_iou_to_template"])
    )


def reference_distances(aligned_mask: np.ndarray, feature_vector: np.ndarray, wing_type: str, stage: str) -> dict[str, object]:
    refs = reference_bank()[wing_type][stage]
    if not refs:
        return {
            "feature_distance": 1e9,
            "nn_distance": 1.0,
            "topk_distance": 1.0,
            "reference_count": 0,
            "nearest_reference_id": "",
            "nearest_reference_source": "",
            "nearest_reference_path": "",
        }
    scale = global_feature_scales()
    ref_matrix = np.vstack([np.asarray(ref.features, dtype=np.float64) for ref in refs]).astype(np.float64)
    ref_mean = ref_matrix.mean(axis=0)
    feature_distance = float(np.sqrt(np.mean(((feature_vector - ref_mean) / scale) ** 2)))
    dice_rows = []
    for ref in refs:
        dice = dice_score(aligned_mask, np.asarray(ref.mask, dtype=bool))
        dice_rows.append((dice, ref))
    dice_rows.sort(key=lambda item: item[0], reverse=True)
    best_dice, best_ref = dice_rows[0]
    topk = dice_rows[: min(5, len(dice_rows))]
    topk_dice = float(np.mean([row[0] for row in topk])) if topk else float(best_dice)
    return {
        "feature_distance": feature_distance,
        "nn_distance": float(1.0 - best_dice),
        "topk_distance": float(1.0 - topk_dice),
        "reference_count": len(refs),
        "nearest_reference_id": best_ref.ref_id,
        "nearest_reference_source": best_ref.source,
        "nearest_reference_path": str(best_ref.mask_path),
    }


def mirror_gray_and_mask(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray_img = Image.fromarray(gray.astype(np.uint8), mode="L")
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mirrored_gray = np.asarray(ImageOps.mirror(gray_img), dtype=np.uint8)
    mirrored_mask = np.asarray(ImageOps.mirror(mask_img), dtype=np.uint8) > 127
    return mirrored_gray, mirrored_mask


def evaluate_variant(gray: np.ndarray, mask: np.ndarray, mirrored: bool) -> dict[str, object]:
    rows: list[CandidateScore] = []
    alignments: dict[str, dict[str, object] | None] = {}
    family_prior_costs = {
        wing_type: family_prior_cost_from_raw_mask(mask, config.project_dir)
        for wing_type, config in FAMILY_CONFIGS.items()
    }
    for wing_type in FAMILY_CONFIGS:
        try:
            alignments[wing_type] = align_sample_to_family(gray, mask, wing_type)
        except Exception:
            alignments[wing_type] = None
    for wing_type in FAMILY_CONFIGS:
        alignment = alignments[wing_type]
        for stage in STAGE_LABELS:
            if alignment is None:
                template_cost = math.inf
                ref_stats = reference_distances(np.zeros((2048, 2048), dtype=bool), np.zeros(len(FEATURE_NAMES), dtype=np.float64), wing_type, stage)
            else:
                template_cost = template_stage_cost(alignment, wing_type, stage)
                ref_stats = reference_distances(
                    np.asarray(alignment["aligned_mask"], dtype=bool),
                    np.asarray(alignment["feature_vector"], dtype=np.float64),
                    wing_type,
                    stage,
                )
            rows.append(
                CandidateScore(
                    wing_type=wing_type,
                    stage=stage,
                    template_cost=float(template_cost),
                    family_prior_cost=float(family_prior_costs[wing_type]),
                    feature_distance=float(ref_stats["feature_distance"]),
                    nn_distance=float(ref_stats["nn_distance"]),
                    topk_distance=float(ref_stats["topk_distance"]),
                    reference_count=int(ref_stats["reference_count"]),
                    nearest_reference_id=str(ref_stats["nearest_reference_id"]),
                    nearest_reference_source=str(ref_stats["nearest_reference_source"]),
                    nearest_reference_path=str(ref_stats["nearest_reference_path"]),
                )
            )
    template_norm = minmax_normalize([row.template_cost for row in rows])
    family_prior_norm = minmax_normalize([row.family_prior_cost for row in rows])
    feature_norm = minmax_normalize([row.feature_distance for row in rows])
    nn_norm = minmax_normalize([row.nn_distance for row in rows])
    weights = np.asarray([0.40, 0.25, 0.20, 0.15], dtype=np.float64)
    combined_raw: list[float] = []
    scored_rows: list[CandidateScore] = []
    for index, row in enumerate(rows):
        if not np.isfinite(row.template_cost):
            adjusted_weights = np.asarray([0.0, 0.45, 0.30, 0.25], dtype=np.float64)
        else:
            adjusted_weights = weights
        combined = float(
            adjusted_weights[0] * template_norm[index]
            + adjusted_weights[1] * family_prior_norm[index]
            + adjusted_weights[2] * feature_norm[index]
            + adjusted_weights[3] * nn_norm[index]
        )
        combined_raw.append(combined)
        scored_rows.append(
            CandidateScore(
                wing_type=row.wing_type,
                stage=row.stage,
                template_cost=row.template_cost,
                family_prior_cost=row.family_prior_cost,
                feature_distance=row.feature_distance,
                nn_distance=row.nn_distance,
                topk_distance=row.topk_distance,
                reference_count=row.reference_count,
                nearest_reference_id=row.nearest_reference_id,
                nearest_reference_source=row.nearest_reference_source,
                nearest_reference_path=row.nearest_reference_path,
                normalized_template_cost=float(template_norm[index]),
                normalized_feature_distance=float(feature_norm[index]),
                normalized_nn_distance=float(nn_norm[index]),
                combined_cost=combined,
            )
        )
    scored_rows = [
        CandidateScore(
            wing_type=row.wing_type,
            stage=row.stage,
            template_cost=row.template_cost,
            family_prior_cost=row.family_prior_cost,
            feature_distance=row.feature_distance,
            nn_distance=row.nn_distance,
            topk_distance=row.topk_distance,
            reference_count=row.reference_count,
            nearest_reference_id=row.nearest_reference_id,
            nearest_reference_source=row.nearest_reference_source,
            nearest_reference_path=row.nearest_reference_path,
            normalized_template_cost=row.normalized_template_cost,
            normalized_feature_distance=row.normalized_feature_distance,
            normalized_nn_distance=row.normalized_nn_distance,
            combined_cost=row.combined_cost,
            class_probability=0.0,
        )
        for row in scored_rows
    ]
    score_scale = max(float(np.std(combined_raw)), 0.06)
    class_probs = softmax(-np.asarray(combined_raw, dtype=np.float64) / score_scale)
    scored_rows = [
        CandidateScore(
            wing_type=row.wing_type,
            stage=row.stage,
            template_cost=row.template_cost,
            family_prior_cost=row.family_prior_cost,
            feature_distance=row.feature_distance,
            nn_distance=row.nn_distance,
            topk_distance=row.topk_distance,
            reference_count=row.reference_count,
            nearest_reference_id=row.nearest_reference_id,
            nearest_reference_source=row.nearest_reference_source,
            nearest_reference_path=row.nearest_reference_path,
            normalized_template_cost=row.normalized_template_cost,
            normalized_feature_distance=row.normalized_feature_distance,
            normalized_nn_distance=row.normalized_nn_distance,
            combined_cost=row.combined_cost,
            class_probability=float(class_probs[index]),
        )
        for index, row in enumerate(scored_rows)
    ]
    ranked = sorted(scored_rows, key=lambda row: row.combined_cost)
    return {
        "candidates": scored_rows,
        "ranked": ranked,
        "alignments": alignments,
        "mirrored": mirrored,
    }


def select_best_variant(gray: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    normal = evaluate_variant(gray, mask, mirrored=False)
    mirrored_gray, mirrored_mask = mirror_gray_and_mask(gray, mask)
    mirrored = evaluate_variant(mirrored_gray, mirrored_mask, mirrored=True)
    if not mirrored["ranked"]:
        return normal
    if not normal["ranked"]:
        return mirrored
    return normal if normal["ranked"][0].combined_cost <= mirrored["ranked"][0].combined_cost else mirrored


def top_feature_summary(alignment: dict[str, object] | None, wing_type: str) -> list[dict[str, object]]:
    if alignment is None:
        return []
    feature_vector = np.asarray(alignment["feature_vector"], dtype=np.float64)
    feature_map = {name: float(feature_vector[index]) for index, name in enumerate(FEATURE_NAMES)}
    ordered = [
        "protrusion_rate",
        "protrusion_radius",
        "protrusion_depth",
        "left_margin_curvature_peak",
        "notch_depth",
        "solidity",
        "circularity",
        "width_std",
        "radial_std",
    ]
    rows = []
    for name in ordered:
        if name not in feature_map:
            continue
        rows.append({"name": name, "value": float(feature_map[name]), "wing_type_space": wing_type})
    return rows


def classify_single_image(
    image_path: Path,
    mask_path: Path | None = None,
    item_id_override: str = "",
    stage_profile: str = "deployment_balanced",
    wing_conf_threshold: float = 0.62,
    stage_conf_threshold: float = 0.48,
    wing_margin_threshold: float = 0.06,
    stage_margin_threshold: float = 0.04,
    nn_distance_threshold: float = 0.42,
    strong_stage_override_wing_conf_min: float = 0.95,
    forewing_far_override_stage_conf_min: float = 0.999,
    forewing_far_override_stage_margin_min: float = 0.999,
    hindwing_far_override_stage_conf_min: float = 0.999,
    hindwing_far_override_stage_margin_min: float = 0.999,
    wing_type_override: str = "",
) -> dict[str, object]:
    item_id = str(item_id_override).strip() or derive_item_id(image_path)
    gray = load_gray(image_path)
    if mask_path is not None:
        raw_mask = load_mask(mask_path)
        mask = extract_wing_mask(raw_mask.astype(np.uint8) * 255)
    else:
        preprocess = extract_best_wing_mask(gray)
        mask = np.asarray(preprocess["mask"], dtype=bool)
    wing_bundle = load_wing_model_bundle()
    wing_model_pred = ""
    wing_model_conf = 0.0
    wing_model_hind_probability = 0.5
    if wing_bundle is not None:
        if "bundle_kind" in wing_bundle:
            wing_model_pred, wing_model_conf, wing_model_hind_probability = apply_aligned_wing_bundle(
                wing_bundle,
                item_id=item_id,
                image_path=image_path,
                mask_path=mask_path,
            )
        else:
            wing_feature_row = build_wing_feature_row(
                item_id=item_id,
                image_path=str(image_path.resolve()),
                project_dir=Path(str(wing_bundle.get("project_dir", "")) or str(DEFAULT_HINDWING_PROJECT_DIR)).resolve(),
                source_group="dapi_classifier",
                modality="dapi",
                wing_type_label="",
                stage_label="",
                is_other_stage=0,
                split_group=f"inference::{item_id.lower()}",
            )
            wing_model_hind_probability = float(apply_binary_probability_model(wing_bundle, [wing_feature_row])[0])
            wing_model_pred = "hindwing" if wing_model_hind_probability >= 0.5 else "forewing"
            wing_model_conf = float(max(wing_model_hind_probability, 1.0 - wing_model_hind_probability))
    best_variant = select_best_variant(gray, mask)
    ranked: list[CandidateScore] = list(best_variant["ranked"])
    if not ranked:
        raise ValueError("No valid candidate scores were produced.")
    best = ranked[0]
    requested_wing_type = str(wing_type_override).strip().lower()
    if requested_wing_type not in {"forewing", "hindwing"}:
        requested_wing_type = ""
    chosen_wing_type = requested_wing_type or wing_model_pred or best.wing_type
    same_family = sorted([row for row in ranked if row.wing_type == chosen_wing_type], key=lambda row: row.combined_cost)
    other_family = sorted([row for row in ranked if row.wing_type != chosen_wing_type], key=lambda row: row.combined_cost)
    if same_family:
        best = same_family[0]
    else:
        chosen_wing_type = best.wing_type
        same_family = [row for row in ranked if row.wing_type == chosen_wing_type]
        other_family = [row for row in ranked if row.wing_type != chosen_wing_type]
    second_stage = same_family[1] if len(same_family) > 1 else same_family[0]
    second_family = other_family[0] if other_family else best
    wing_margin = float(second_family.combined_cost - best.combined_cost)
    if requested_wing_type and wing_model_pred:
        wing_prob = float(
            wing_model_hind_probability
            if requested_wing_type == "hindwing"
            else 1.0 - wing_model_hind_probability
        )
    else:
        wing_prob = wing_model_conf if wing_model_pred else float(sum(row.class_probability for row in ranked if row.wing_type == chosen_wing_type))
    family_prob_total = max(wing_prob, 1e-9)
    family_stage_prob_total = max(float(sum(row.class_probability for row in same_family)), 1e-9)
    stage_prob = float(best.class_probability / family_stage_prob_total)
    family_rows = [row for row in ranked if row.wing_type == chosen_wing_type]
    family_stage_probs = np.asarray([row.class_probability for row in family_rows], dtype=np.float64)
    family_stage_probs = family_stage_probs / max(float(family_stage_probs.sum()), 1e-9)
    maturity_score = float(
        sum(stage_numeric(row.stage) * float(prob) for row, prob in zip(family_rows, family_stage_probs))
    )
    stage_model_payload: dict[str, object] | None = None
    stage_model_pred = ""
    stage_model_conf = 0.0
    stage_model_margin = 0.0
    stage_model_decision = ""
    stage_model_pred_raw = ""
    stage_model_conf_raw = 0.0
    ordinal_stage_pred = ""
    ordinal_stage_conf = 0.0
    fused_stage_pred = ""
    fused_stage_conf = 0.0
    fused_stage_margin = 0.0
    stage_profile_name = str(stage_profile).strip() or "deployment_balanced"
    bundle_profile_reference_nn_max = nn_distance_threshold
    bundle_far_reference_override = False
    ordinal_conflict = False
    stage_bundle: dict[str, object] | None = None
    if chosen_wing_type == "hindwing":
        stage_bundle = load_hindwing_stage_model_bundle()
    elif chosen_wing_type == "forewing":
        stage_bundle = load_forewing_stage_model_bundle()
    if stage_bundle is not None:
        if chosen_wing_type == "hindwing":
            stage_model_payload = apply_hindwing_stage_bundle(
                stage_bundle,
                item_id=item_id,
                image_path=image_path,
                mask_path=mask_path,
                stage_profile=stage_profile_name,
            )
        else:
            stage_model_payload = apply_forewing_stage_bundle(
                stage_bundle,
                item_id=item_id,
                image_path=image_path,
                mask_path=mask_path,
                stage_profile=stage_profile_name,
            )
        stage_model_pred = str(stage_model_payload.get("stage_pred", "")).strip()
        stage_model_conf = float(stage_model_payload.get("stage_conf", 0.0))
        stage_model_margin = float(stage_model_payload.get("stage_margin", 0.0))
        stage_model_decision = str(stage_model_payload.get("decision", "")).strip()
        stage_model_pred_raw = str(stage_model_payload.get("stage_model_pred_raw", stage_model_pred)).strip()
        stage_model_conf_raw = float(stage_model_payload.get("stage_model_conf_raw", stage_model_conf))
        ordinal_stage_pred = str(stage_model_payload.get("ordinal_stage_pred", "")).strip()
        ordinal_stage_conf = float(stage_model_payload.get("ordinal_stage_conf", 0.0))
        fused_stage_pred = str(stage_model_payload.get("fused_stage_pred", stage_model_pred)).strip()
        fused_stage_conf = float(stage_model_payload.get("fused_stage_conf", stage_model_conf))
        fused_stage_margin = float(stage_model_payload.get("fused_stage_margin", stage_model_margin))
        stage_profile_name = str(stage_model_payload.get("stage_profile", stage_profile_name)).strip() or stage_profile_name
        bundle_far_reference_override = bool(stage_model_payload.get("far_reference_override_applied", False))
        ordinal_conflict = bool(stage_model_payload.get("ordinal_conflict", False))
        if str(stage_bundle.get("bundle_kind", "")).strip().lower() == "stage_fusion_v1":
            selected_profile = dict(dict(stage_bundle.get("profile_settings", {})).get(stage_profile_name, {}))
            bundle_profile_reference_nn_max = float(selected_profile.get("reference_nn_max", nn_distance_threshold))
        matching_stage = [row for row in same_family if row.stage == stage_model_pred]
        if matching_stage:
            best = matching_stage[0]
        others = [row for row in same_family if row.stage != best.stage]
        second_stage = others[0] if others else best
        stage_prob = stage_model_conf
        stage_margin = stage_model_margin
        probability_map = dict(stage_model_payload.get("probabilities", {}))
        maturity_score = float(
            sum(stage_numeric(stage_label) * float(probability_map.get(stage_label, 0.0)) for stage_label in STAGE_LABELS)
        )
    else:
        stage_margin = float(second_stage.combined_cost - best.combined_cost)
    chosen_alignment = best_variant["alignments"].get(chosen_wing_type)
    stage_model_auto_override = bool(stage_model_decision.startswith(f"auto_{chosen_wing_type}_"))
    strong_stage_override = bool(
        chosen_wing_type in {"forewing", "hindwing"}
        and wing_model_pred == chosen_wing_type
        and wing_model_conf >= strong_stage_override_wing_conf_min
        and stage_model_auto_override
    )
    forewing_far_reference_override = bool(
        chosen_wing_type == "forewing"
        and strong_stage_override
        and stage_model_conf >= forewing_far_override_stage_conf_min
        and stage_model_margin >= forewing_far_override_stage_margin_min
    )
    hindwing_far_reference_override = bool(
        chosen_wing_type == "hindwing"
        and strong_stage_override
        and stage_model_conf >= hindwing_far_override_stage_conf_min
        and stage_model_margin >= hindwing_far_override_stage_margin_min
    )
    bundle_far_reference_override = bool(bundle_far_reference_override and strong_stage_override)
    review_reasons: list[str] = []
    if chosen_alignment is None:
        review_reasons.append("family_alignment_failed")
    if wing_prob < wing_conf_threshold:
        review_reasons.append("low_wing_type_conf")
    if wing_model_pred and wing_model_pred != ranked[0].wing_type and not strong_stage_override:
        review_reasons.append("wing_model_overrode_template_family")
    if requested_wing_type and wing_model_pred and requested_wing_type != wing_model_pred:
        review_reasons.append("canonical_family_disagrees_with_full_alignment")
    if stage_model_pred and stage_model_pred != same_family[0].stage and not stage_model_auto_override:
        review_reasons.append("stage_model_overrode_template_stage")
    if stage_prob < stage_conf_threshold and not stage_model_auto_override:
        review_reasons.append("low_stage_conf")
    if wing_margin < wing_margin_threshold and not strong_stage_override:
        review_reasons.append("small_wing_margin")
    if stage_margin < stage_margin_threshold and not stage_model_auto_override:
        review_reasons.append("small_stage_margin")
    far_threshold = float(bundle_profile_reference_nn_max if stage_bundle is not None else nn_distance_threshold)
    if best.nn_distance > far_threshold and not forewing_far_reference_override and not hindwing_far_reference_override and not bundle_far_reference_override:
        review_reasons.append("far_from_reference_bank")
    if ordinal_conflict and not stage_model_auto_override:
        review_reasons.append("ordinal_conflict")
    if not np.isfinite(best.template_cost):
        review_reasons.append("template_registration_failed")
    decision = f"auto_{best.wing_type}_{best.stage}" if not review_reasons else "uncertain_review"
    return {
        "id": item_id,
        "image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()) if mask_path is not None else "",
        "wing_type_pred": best.wing_type,
        "wing_model_pred": wing_model_pred,
        "wing_model_conf": wing_model_conf,
        "wing_model_hind_probability": wing_model_hind_probability,
        "wing_type_override": requested_wing_type,
        "wing_type_conf": wing_prob,
        "stage_model_pred": stage_model_pred,
        "stage_model_conf": stage_model_conf,
        "stage_model_margin": stage_model_margin,
        "stage_model_decision": stage_model_decision,
        "stage_model_pred_raw": stage_model_pred_raw,
        "stage_model_conf_raw": stage_model_conf_raw,
        "ordinal_stage_pred": ordinal_stage_pred,
        "ordinal_stage_conf": ordinal_stage_conf,
        "fused_stage_pred": fused_stage_pred,
        "fused_stage_conf": fused_stage_conf,
        "fused_stage_margin": fused_stage_margin,
        "stage_profile": stage_profile_name,
        "bundle_profile_reference_nn_max": float(bundle_profile_reference_nn_max),
        "bundle_far_reference_override_applied": bundle_far_reference_override,
        "ordinal_conflict": ordinal_conflict,
        "best_p_model_candidate_name": str(stage_model_payload.get("best_p_model_candidate_name", "")) if stage_model_payload else "",
        "best_p_model_family": str(stage_model_payload.get("best_p_model_family", "")) if stage_model_payload else "",
        "early05_aux_score": float(stage_model_payload.get("early05_aux_score", 0.0)) if stage_model_payload else 0.0,
        "early05_rescue_applied": bool(stage_model_payload.get("early05_rescue_applied", False)) if stage_model_payload else False,
        "p_model_probabilities": dict(stage_model_payload.get("p_model_probabilities", {})) if stage_model_payload else {},
        "template_stage_probabilities": dict(stage_model_payload.get("template_stage_probabilities", {})) if stage_model_payload else {},
        "reference_stage_probabilities": dict(stage_model_payload.get("reference_stage_probabilities", {})) if stage_model_payload else {},
        "strong_stage_override": strong_stage_override,
        "forewing_far_reference_override_applied": forewing_far_reference_override,
        "hindwing_far_reference_override_applied": hindwing_far_reference_override,
        "stage_pred": best.stage,
        "stage_conf": stage_prob,
        "maturity_score": maturity_score,
        "maturity_offset": float(maturity_score - stage_numeric(best.stage)),
        "best_template": f"{best.wing_type}:{best.stage}",
        "second_template": f"{chosen_wing_type}:{second_stage.stage}",
        "template_margin": float(second_stage.template_cost - best.template_cost) if np.isfinite(best.template_cost) and np.isfinite(second_stage.template_cost) else 0.0,
        "combined_margin": stage_margin,
        "wing_margin": wing_margin,
        "best_reference_id": best.nearest_reference_id,
        "best_reference_source": best.nearest_reference_source,
        "best_reference_path": best.nearest_reference_path,
        "best_reference_count": best.reference_count,
        "reference_nn_distance": best.nn_distance,
        "reference_topk_distance": best.topk_distance,
        "feature_distance": best.feature_distance,
        "template_cost": best.template_cost,
        "decision": decision,
        "review_reason": ",".join(review_reasons),
        "strong_stage_override_wing_conf_min": float(strong_stage_override_wing_conf_min),
        "forewing_far_override_stage_conf_min": float(forewing_far_override_stage_conf_min),
        "forewing_far_override_stage_margin_min": float(forewing_far_override_stage_margin_min),
        "hindwing_far_override_stage_conf_min": float(hindwing_far_override_stage_conf_min),
        "hindwing_far_override_stage_margin_min": float(hindwing_far_override_stage_margin_min),
        "mirrored_input_selected": bool(best_variant["mirrored"]),
        "top_features": top_feature_summary(chosen_alignment, chosen_wing_type),
        "all_candidates": [
            {
                "wing_type": row.wing_type,
                "stage": row.stage,
                "template_cost": row.template_cost,
                "family_prior_cost": row.family_prior_cost,
                "feature_distance": row.feature_distance,
                "nn_distance": row.nn_distance,
                "topk_distance": row.topk_distance,
                "combined_cost": row.combined_cost,
                "class_probability": row.class_probability,
                "nearest_reference_id": row.nearest_reference_id,
                "nearest_reference_source": row.nearest_reference_source,
            }
            for row in ranked
        ],
        "_variant": best_variant,
        "_gray": gray,
        "_mask": mask,
    }


def preview_image_from_array(array: np.ndarray, title: str, size: int = 240) -> Image.Image:
    if array.ndim == 2:
        image = Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
    else:
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    image = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size + 28), (18, 18, 18))
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), title, fill="white")
    return canvas


def overlay_masks(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    left_bool = np.asarray(left_mask, dtype=bool)
    right_bool = np.asarray(right_mask, dtype=bool)
    canvas = np.zeros((*left_bool.shape, 3), dtype=np.uint8)
    canvas[left_bool] = np.asarray([85, 180, 255], dtype=np.uint8)
    canvas[right_bool] = np.asarray([255, 140, 95], dtype=np.uint8)
    canvas[np.logical_and(left_bool, right_bool)] = np.asarray([255, 255, 255], dtype=np.uint8)
    return canvas


def write_review_panel(output_path: Path, payload: dict[str, object]) -> None:
    best = payload["all_candidates"][0]
    wing_type = str(best["wing_type"])
    alignment = payload["_variant"]["alignments"].get(wing_type)
    if alignment is None:
        return
    aligned_mask = np.asarray(alignment["aligned_mask"], dtype=bool)
    template_mask = np.asarray(family_assets()[wing_type]["stage_templates"][str(best["stage"])]["_mask"], dtype=bool)
    reference_mask = load_mask(Path(str(payload["best_reference_path"])))
    panels = [
        preview_image_from_array(np.asarray(payload["_gray"], dtype=np.uint8), "Raw image"),
        preview_image_from_array(np.asarray(payload["_mask"], dtype=np.uint8) * 255, "Raw mask"),
        preview_image_from_array(np.asarray(aligned_mask, dtype=np.uint8) * 255, "Aligned mask"),
        preview_image_from_array(overlay_masks(aligned_mask, template_mask), "Mask vs template"),
        preview_image_from_array(np.asarray(reference_mask, dtype=np.uint8) * 255, "Nearest reference"),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels) + 130
    canvas = Image.new("RGB", (width, height), (12, 12, 12))
    x_offset = 0
    for panel in panels:
        canvas.paste(panel, (x_offset, 0))
        x_offset += panel.width
    draw = ImageDraw.Draw(canvas)
    summary_lines = [
        f"Prediction: {payload['wing_type_pred']} / stage {payload['stage_pred']}",
        f"Decision: {payload['decision']}",
        f"Wing conf: {safe_float(payload['wing_type_conf']):.3f}    Stage conf: {safe_float(payload['stage_conf']):.3f}",
        f"Stage model: {payload['stage_model_pred'] or 'n/a'} ({payload['stage_model_decision'] or 'n/a'})    Margin: {safe_float(payload['stage_model_margin']):.3f}",
        f"Raw stage: {payload.get('stage_model_pred_raw', 'n/a')}    Ordinal: {payload.get('ordinal_stage_pred', 'n/a')}    Fused: {payload.get('fused_stage_pred', 'n/a')}",
        (
            f"Strong override: {bool(payload.get('strong_stage_override', False))}    "
            f"Forewing far override: {bool(payload.get('forewing_far_reference_override_applied', False))}    "
            f"Hindwing far override: {bool(payload.get('hindwing_far_reference_override_applied', False))}    "
            f"Bundle far override: {bool(payload.get('bundle_far_reference_override_applied', False))}"
        ),
        f"Stage profile: {payload.get('stage_profile', 'deployment_balanced')}    Ordinal conflict: {bool(payload.get('ordinal_conflict', False))}",
        f"Reference: {payload['best_reference_id']} ({payload['best_reference_source']})",
        f"Template margin: {safe_float(payload['template_margin']):.4f}    Combined margin: {safe_float(payload['combined_margin']):.4f}",
        f"Review reason: {payload['review_reason'] or 'none'}",
    ]
    y = max(panel.height for panel in panels) + 10
    for line in summary_lines:
        draw.text((16, y), line, fill="white")
        y += 18
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def prediction_row(payload: dict[str, object]) -> dict[str, object]:
    return {
        "id": payload["id"],
        "image_path": payload["image_path"],
        "mask_path": payload["mask_path"],
        "wing_type_pred": payload["wing_type_pred"],
        "wing_model_pred": payload["wing_model_pred"],
        "wing_model_conf": payload["wing_model_conf"],
        "wing_model_hind_probability": payload["wing_model_hind_probability"],
        "wing_type_conf": payload["wing_type_conf"],
        "stage_model_pred": payload["stage_model_pred"],
        "stage_model_conf": payload["stage_model_conf"],
        "stage_model_margin": payload["stage_model_margin"],
        "stage_model_decision": payload["stage_model_decision"],
        "stage_model_pred_raw": payload.get("stage_model_pred_raw", ""),
        "stage_model_conf_raw": payload.get("stage_model_conf_raw", 0.0),
        "ordinal_stage_pred": payload.get("ordinal_stage_pred", ""),
        "ordinal_stage_conf": payload.get("ordinal_stage_conf", 0.0),
        "fused_stage_pred": payload.get("fused_stage_pred", ""),
        "fused_stage_conf": payload.get("fused_stage_conf", 0.0),
        "fused_stage_margin": payload.get("fused_stage_margin", 0.0),
        "stage_profile": payload.get("stage_profile", ""),
        "bundle_profile_reference_nn_max": payload.get("bundle_profile_reference_nn_max", 0.0),
        "bundle_far_reference_override_applied": payload.get("bundle_far_reference_override_applied", False),
        "ordinal_conflict": payload.get("ordinal_conflict", False),
        "best_p_model_candidate_name": payload.get("best_p_model_candidate_name", ""),
        "best_p_model_family": payload.get("best_p_model_family", ""),
        "early05_aux_score": payload.get("early05_aux_score", 0.0),
        "early05_rescue_applied": payload.get("early05_rescue_applied", False),
        "p_model_probabilities": json.dumps(payload.get("p_model_probabilities", {}), ensure_ascii=False),
        "template_stage_probabilities": json.dumps(payload.get("template_stage_probabilities", {}), ensure_ascii=False),
        "reference_stage_probabilities": json.dumps(payload.get("reference_stage_probabilities", {}), ensure_ascii=False),
        "strong_stage_override": payload["strong_stage_override"],
        "forewing_far_reference_override_applied": payload["forewing_far_reference_override_applied"],
        "hindwing_far_reference_override_applied": payload["hindwing_far_reference_override_applied"],
        "stage_pred": payload["stage_pred"],
        "stage_conf": payload["stage_conf"],
        "maturity_score": payload["maturity_score"],
        "maturity_offset": payload["maturity_offset"],
        "best_template": payload["best_template"],
        "second_template": payload["second_template"],
        "template_margin": payload["template_margin"],
        "combined_margin": payload["combined_margin"],
        "wing_margin": payload["wing_margin"],
        "best_reference_id": payload["best_reference_id"],
        "best_reference_source": payload["best_reference_source"],
        "best_reference_path": payload["best_reference_path"],
        "reference_nn_distance": payload["reference_nn_distance"],
        "reference_topk_distance": payload["reference_topk_distance"],
        "feature_distance": payload["feature_distance"],
        "template_cost": payload["template_cost"],
        "decision": payload["decision"],
        "review_reason": payload["review_reason"],
        "strong_stage_override_wing_conf_min": payload["strong_stage_override_wing_conf_min"],
        "forewing_far_override_stage_conf_min": payload["forewing_far_override_stage_conf_min"],
        "forewing_far_override_stage_margin_min": payload["forewing_far_override_stage_margin_min"],
        "hindwing_far_override_stage_conf_min": payload["hindwing_far_override_stage_conf_min"],
        "hindwing_far_override_stage_margin_min": payload["hindwing_far_override_stage_margin_min"],
        "mirrored_input_selected": payload["mirrored_input_selected"],
    }


def batch_image_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def manifest_targets(manifest_csv: Path) -> list[tuple[str, Path, Path | None]]:
    rows = read_csv_rows(manifest_csv)
    targets: list[tuple[str, Path, Path | None]] = []
    for row in rows:
        image_path = str(row.get("image_path", "")).strip()
        if not image_path:
            continue
        item_id = str(row.get("id", "")).strip() or derive_item_id(Path(image_path))
        mask_raw = str(row.get("mask_path", "")).strip()
        targets.append((item_id, Path(image_path).resolve(), Path(mask_raw).resolve() if mask_raw else None))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify DAPI wing type and stage using templates plus manually aligned reference banks.")
    parser.add_argument("--image", default="")
    parser.add_argument("--mask", default="")
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--manifest-csv", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-review-panels", action="store_true")
    parser.add_argument("--stage-profile", default="deployment_balanced")
    parser.add_argument("--wing-conf-threshold", type=float, default=0.62)
    parser.add_argument("--stage-conf-threshold", type=float, default=0.48)
    parser.add_argument("--wing-margin-threshold", type=float, default=0.06)
    parser.add_argument("--stage-margin-threshold", type=float, default=0.04)
    parser.add_argument("--nn-distance-threshold", type=float, default=0.42)
    parser.add_argument("--strong-stage-override-wing-conf-min", type=float, default=0.95)
    parser.add_argument("--forewing-far-override-stage-conf-min", type=float, default=0.999)
    parser.add_argument("--forewing-far-override-stage-margin-min", type=float, default=0.999)
    parser.add_argument("--hindwing-far-override-stage-conf-min", type=float, default=0.999)
    parser.add_argument("--hindwing-far-override-stage-margin-min", type=float, default=0.999)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, Path, Path | None]] = []
    if args.manifest_csv:
        targets = manifest_targets(Path(args.manifest_csv).resolve())
    elif args.image:
        image_path = Path(args.image).resolve()
        mask_path = Path(args.mask).resolve() if args.mask else None
        targets.append((derive_item_id(image_path), image_path, mask_path))
    elif args.input_dir:
        for image_path in batch_image_paths(Path(args.input_dir).resolve()):
            targets.append((derive_item_id(image_path), image_path, None))
    else:
        raise ValueError("Pass --manifest-csv, --image, or --input-dir.")

    rows: list[dict[str, object]] = []
    for item_id, image_path, mask_path in targets:
        item_dir = output_dir / safe_output_name(str(item_id))
        prediction_json_path = item_dir / "prediction.json"
        if args.resume and prediction_json_path.exists():
            payload = load_json(prediction_json_path)
        else:
            payload = classify_single_image(
                image_path=image_path,
                mask_path=mask_path,
                item_id_override=item_id,
                stage_profile=str(args.stage_profile),
                wing_conf_threshold=float(args.wing_conf_threshold),
                stage_conf_threshold=float(args.stage_conf_threshold),
                wing_margin_threshold=float(args.wing_margin_threshold),
                stage_margin_threshold=float(args.stage_margin_threshold),
                nn_distance_threshold=float(args.nn_distance_threshold),
                strong_stage_override_wing_conf_min=float(args.strong_stage_override_wing_conf_min),
                forewing_far_override_stage_conf_min=float(args.forewing_far_override_stage_conf_min),
                forewing_far_override_stage_margin_min=float(args.forewing_far_override_stage_margin_min),
                hindwing_far_override_stage_conf_min=float(args.hindwing_far_override_stage_conf_min),
                hindwing_far_override_stage_margin_min=float(args.hindwing_far_override_stage_margin_min),
            )
            item_dir = output_dir / safe_output_name(str(payload["id"]))
            prediction_json_path = item_dir / "prediction.json"
            serializable = {key: value for key, value in payload.items() if not key.startswith("_")}
            save_json(prediction_json_path, serializable)
            if not args.skip_review_panels:
                write_review_panel(item_dir / "review_panel.png", payload)
        rows.append(prediction_row(payload))

    if rows:
        write_csv_rows(output_dir / "predictions.csv", list(rows[0].keys()), rows)
    print(f"Prediction outputs: {output_dir}")


if __name__ == "__main__":
    main()
