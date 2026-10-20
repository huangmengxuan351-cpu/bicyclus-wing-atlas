from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from forewing_stage_template_utils import DEFAULT_PROJECT_DIR as DEFAULT_FOREWING_PROJECT_DIR
from pose_landmark_utils import DEFAULT_PROJECT_DIR as DEFAULT_HINDWING_PROJECT_DIR
from sample_identity_utils import canonical_original_id, strict_group_key
from shape_priority_hindwing_utils import (
    coarse_template_similarity,
    compute_shape_features,
    derive_split_group,
    extract_best_wing_mask,
    finalize_mask,
    infer_modality,
    load_gray_image,
    resize_mask_crop,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASK_VECTOR_SIZES = (32, 48, 64)
METADATA_KEYS = {
    "id",
    "image_path",
    "mask_path",
    "source_group",
    "modality",
    "wing_type_label",
    "stage_label",
    "original_id",
    "strict_group",
    "split_group",
    "status",
    "manual_review_status",
    "alignment_source",
    "mask_source",
}


def load_binary_mask(mask_path: str | Path) -> np.ndarray:
    with Image.open(mask_path) as image:
        raw = np.asarray(image.convert("L"), dtype=np.uint8) > 127
    return finalize_mask(raw)


def prepare_gray_and_mask(
    image_path: str | Path,
    mask_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    gray = load_gray_image(image_path)
    if mask_path and Path(mask_path).exists():
        return gray, load_binary_mask(mask_path), "provided_mask"
    preprocess = extract_best_wing_mask(gray)
    return gray, np.asarray(preprocess["mask"], dtype=bool), "extracted_mask"


def _rename_template_features(features: dict[str, float], family_name: str) -> dict[str, float]:
    return {
        f"{family_name}_template_iou": float(features.get("hindwing_template_iou", 0.0)),
        f"{family_name}_template_dice": float(features.get("hindwing_template_dice", 0.0)),
        f"{family_name}_template_diff": float(features.get("hindwing_template_diff", 1.0)),
    }


def compute_dual_family_features(
    mask: np.ndarray,
    forewing_project_dir: str | Path = DEFAULT_FOREWING_PROJECT_DIR,
    hindwing_project_dir: str | Path = DEFAULT_HINDWING_PROJECT_DIR,
) -> dict[str, float]:
    hind_dir = Path(hindwing_project_dir).resolve()
    fore_dir = Path(forewing_project_dir).resolve()

    base = compute_shape_features(mask, hind_dir)
    for key in ["hindwing_template_iou", "hindwing_template_dice", "hindwing_template_diff"]:
        base.pop(key, None)

    hind_template = _rename_template_features(coarse_template_similarity(mask, hind_dir), "hindwing")
    fore_template = _rename_template_features(coarse_template_similarity(mask, fore_dir), "forewing")
    return {
        **{key: float(value) for key, value in base.items()},
        **hind_template,
        **fore_template,
        "family_template_iou_gap": float(hind_template["hindwing_template_iou"] - fore_template["forewing_template_iou"]),
        "family_template_dice_gap": float(hind_template["hindwing_template_dice"] - fore_template["forewing_template_dice"]),
        "family_template_diff_gap": float(fore_template["forewing_template_diff"] - hind_template["hindwing_template_diff"]),
    }


def add_mask_vector_features(row: dict[str, object], mask: np.ndarray, mask_sizes: list[int] | tuple[int, ...]) -> None:
    for size in mask_sizes:
        resized = resize_mask_crop(mask, size=int(size)).astype(np.float32)
        flattened = resized.reshape(-1)
        prefix = f"mask{int(size)}_px_"
        for index, value in enumerate(flattened.tolist()):
            row[f"{prefix}{index:04d}"] = float(value)


def build_aligned_wing_feature_row(
    item_id: str,
    image_path: str | Path,
    mask_path: str | Path | None,
    source_group: str,
    original_id: str = "",
    wing_type_label: str = "",
    stage_label: str = "",
    modality: str = "",
    strict_group: str = "",
    split_group: str = "",
    status: str = "",
    manual_review_status: str = "",
    alignment_source: str = "",
    mask_sizes: list[int] | tuple[int, ...] = DEFAULT_MASK_VECTOR_SIZES,
    include_mask_vectors: bool = True,
    forewing_project_dir: str | Path = DEFAULT_FOREWING_PROJECT_DIR,
    hindwing_project_dir: str | Path = DEFAULT_HINDWING_PROJECT_DIR,
    gray_override: np.ndarray | None = None,
    mask_override: np.ndarray | None = None,
    mask_source_override: str = "",
) -> dict[str, object]:
    resolved_image = str(Path(image_path).resolve())
    resolved_mask = str(Path(mask_path).resolve()) if mask_path else ""
    if gray_override is not None and mask_override is not None:
        gray = np.asarray(gray_override, dtype=np.uint8)
        mask = finalize_mask(np.asarray(mask_override, dtype=bool))
        mask_source = str(mask_source_override or "override_mask")
    else:
        gray, mask, mask_source = prepare_gray_and_mask(resolved_image, resolved_mask or None)
    row: dict[str, object] = {
        "id": item_id,
        "original_id": canonical_original_id(original_id, item_id, resolved_image),
        "image_path": resolved_image,
        "mask_path": resolved_mask,
        "source_group": source_group,
        "modality": modality or infer_modality(resolved_image),
        "wing_type_label": wing_type_label,
        "stage_label": stage_label,
        "strict_group": strict_group or strict_group_key(original_id, item_id, resolved_image),
        "split_group": split_group or derive_split_group(item_id, resolved_image, source_group),
        "status": status,
        "manual_review_status": manual_review_status,
        "alignment_source": alignment_source,
        "mask_source": mask_source,
        "mask_area_pixels": int(mask.sum()),
        "image_height": int(gray.shape[0]),
        "image_width": int(gray.shape[1]),
    }
    row.update(compute_dual_family_features(mask, forewing_project_dir=forewing_project_dir, hindwing_project_dir=hindwing_project_dir))
    if include_mask_vectors:
        add_mask_vector_features(row, mask, mask_sizes)
    return row


def numeric_feature_names(rows: list[dict[str, object]], prefixes: list[str] | None = None) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in METADATA_KEYS or key in seen:
                continue
            if prefixes and not any(key.startswith(prefix) for prefix in prefixes):
                continue
            try:
                float(value)  # type: ignore[arg-type]
            except Exception:
                continue
            seen.add(key)
            selected.append(key)
    return selected


def row_matrix(rows: list[dict[str, object]], feature_names: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for feature_index, feature_name in enumerate(feature_names):
            try:
                matrix[row_index, feature_index] = float(row.get(feature_name, 0.0))  # type: ignore[arg-type]
            except Exception:
                matrix[row_index, feature_index] = 0.0
    return matrix


def family_prototype_from_rows(rows: list[dict[str, object]], wing_type_label: str, mask_size: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for row in rows:
        if str(row.get("wing_type_label", "")).strip().lower() != wing_type_label:
            continue
        prefix = f"mask{int(mask_size)}_px_"
        values = [float(value) for key, value in row.items() if key.startswith(prefix)]
        if values:
            vectors.append(np.asarray(values, dtype=np.float64))
    if not vectors:
        return np.zeros((int(mask_size) * int(mask_size),), dtype=np.float64)
    return np.mean(np.stack(vectors, axis=0), axis=0)


def prototype_hind_probability(vector: np.ndarray, forewing_proto: np.ndarray, hindwing_proto: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    forewing_proto = np.asarray(forewing_proto, dtype=np.float64)
    hindwing_proto = np.asarray(hindwing_proto, dtype=np.float64)
    fore_cost = float(np.mean((vector - forewing_proto) ** 2))
    hind_cost = float(np.mean((vector - hindwing_proto) ** 2))
    scale = max(abs(hind_cost - fore_cost), 1e-6, 0.01)
    hind_logit = (fore_cost - hind_cost) / scale
    return float(1.0 / (1.0 + math.exp(-hind_logit)))
