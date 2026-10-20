from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from aligned_wing_classifier_utils import (
    DEFAULT_MASK_VECTOR_SIZES,
    add_mask_vector_features,
    load_binary_mask,
    prepare_gray_and_mask,
)
from forewing_stage_template_utils import DEFAULT_PROJECT_DIR as DEFAULT_FOREWING_PROJECT_DIR
from hindwing_stage_classifier_utils import row_matrix, stage_probabilities_from_bundle
from sample_identity_utils import canonical_original_id, strict_group_key
from shape_priority_hindwing_utils import (
    compute_shape_features,
    extract_contour_points,
    infer_modality,
    resize_mask_crop,
    width_profile_without_landmarks,
)
from stage_upgrade_utils import apply_stage_fusion_bundle
from stage_template_geometry import compute_radial_profile, fill_profile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_LABELS = ("0.5", "0.75", "1.0")
DEFAULT_FOREWING_STAGE_TEMPLATE_MASKS = {
    "0.5": ROOT / "output_results" / "standard_forewing_templates_stalign_tps_multisample_v2" / "stage_0_5" / "stage_standard_mask.png",
    "0.75": ROOT
    / "output_results"
    / "standard_forewing_templates_stalign_tps_multisample_v2_stage_0_75_usable_only_constrained"
    / "stage_0_75"
    / "stage_standard_mask.png",
    "1.0": ROOT / "output_results" / "standard_forewing_templates_stalign_tps_multisample_v2" / "stage_1_0" / "stage_standard_mask.png",
}
METADATA_KEYS = {
    "id",
    "image_path",
    "mask_path",
    "source_group",
    "modality",
    "gene_family",
    "sample_family",
    "stage_label",
    "original_id",
    "strict_group",
    "split_group",
    "status",
    "manual_review_status",
    "alignment_source",
    "mask_source",
}

KNOWN_PREFIXES = (
    "stage_0_5::",
    "stage_0_75::",
    "stage_1_0::",
    "dll_",
    "dpp_",
    "omb_",
    "sal_",
    "rho3__",
    "rho__",
    "rho3_",
    "rho_",
    "bls_",
    "vein_",
)


def canonical_stage_sample_key(item_id: str, image_path: str | Path) -> str:
    raw = str(item_id or "").strip()
    if not raw:
        raw = Path(image_path).stem
    raw = raw.strip()
    raw = re.sub(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", "", raw, flags=re.IGNORECASE)
    lowered = raw.lower()
    for prefix in KNOWN_PREFIXES:
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            break
    lowered = re.sub(r"_(488|514|546|647)hcr$", "", lowered)
    lowered = re.sub(r"_dapi$", "", lowered)
    lowered = lowered.strip("_")
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or Path(image_path).stem.lower()


def strict_stage_split_group(item_id: str, image_path: str | Path) -> str:
    key = canonical_stage_sample_key(item_id, image_path)
    return f"forewing_stage::{key}"


@lru_cache(maxsize=8)
def load_stage_template_mask(stage_label: str) -> np.ndarray:
    mask_path = DEFAULT_FOREWING_STAGE_TEMPLATE_MASKS[str(stage_label)]
    return load_binary_mask(mask_path)


@lru_cache(maxsize=8)
def load_stage_template_profiles(stage_label: str) -> dict[str, np.ndarray]:
    mask = load_stage_template_mask(stage_label)
    contour = extract_contour_points(mask)
    if contour.shape[0] == 0:
        return {
            "radial": np.zeros(24, dtype=np.float64),
            "width": np.zeros(24, dtype=np.float64),
        }
    center = contour.mean(axis=0)
    radial = fill_profile(compute_radial_profile(contour.astype(np.float64), center.astype(np.float64)))
    width = fill_profile(width_profile_without_landmarks(mask, bins=24))
    return {
        "radial": np.asarray(radial, dtype=np.float64),
        "width": np.asarray(width, dtype=np.float64),
    }


def _mask_overlap_features(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_bool = np.asarray(left, dtype=bool)
    right_bool = np.asarray(right, dtype=bool)
    union = float(np.logical_or(left_bool, right_bool).sum())
    intersection = float(np.logical_and(left_bool, right_bool).sum())
    denom = float(left_bool.sum() + right_bool.sum())
    iou = 0.0 if union <= 0.0 else intersection / union
    dice = 0.0 if denom <= 0.0 else (2.0 * intersection) / denom
    diff = float(np.mean(np.logical_xor(left_bool, right_bool)))
    return {"iou": float(iou), "dice": float(dice), "diff": float(diff)}


def compute_stage_template_features(mask: np.ndarray, stage_labels: list[str] | tuple[str, ...] = DEFAULT_STAGE_LABELS) -> dict[str, float]:
    features: dict[str, float] = {}
    sample_160 = resize_mask_crop(mask, size=160)
    contour = extract_contour_points(mask)
    center = contour.mean(axis=0) if contour.shape[0] else np.asarray([0.0, 0.0], dtype=np.float64)
    sample_radial = (
        np.asarray(fill_profile(compute_radial_profile(contour.astype(np.float64), center.astype(np.float64))), dtype=np.float64)
        if contour.shape[0]
        else np.zeros(24, dtype=np.float64)
    )
    sample_width = np.asarray(fill_profile(width_profile_without_landmarks(mask, bins=24)), dtype=np.float64)

    best_stage = ""
    best_score = -1.0
    second_score = -1.0
    for stage_label in stage_labels:
        stage_key = str(stage_label).replace(".", "_")
        template_mask = load_stage_template_mask(stage_label)
        template_160 = resize_mask_crop(template_mask, size=160)
        overlap = _mask_overlap_features(sample_160, template_160)
        profiles = load_stage_template_profiles(stage_label)
        radial_l2 = float(np.linalg.norm(sample_radial - profiles["radial"]))
        width_l2 = float(np.linalg.norm(sample_width - profiles["width"]))
        features[f"stage_iou_{stage_key}"] = float(overlap["iou"])
        features[f"stage_dice_{stage_key}"] = float(overlap["dice"])
        features[f"stage_diff_{stage_key}"] = float(overlap["diff"])
        features[f"stage_radial_l2_{stage_key}"] = radial_l2
        features[f"stage_width_l2_{stage_key}"] = width_l2
        score = float(overlap["dice"] - 0.15 * overlap["diff"])
        if score > best_score:
            second_score = best_score
            best_score = score
            best_stage = str(stage_label)
        elif score > second_score:
            second_score = score
    features["best_template_stage"] = best_stage
    features["best_template_score"] = float(best_score)
    features["best_template_margin"] = float(best_score - max(second_score, -1.0))
    return features


def build_forewing_stage_feature_row(
    item_id: str,
    image_path: str | Path,
    mask_path: str | Path | None,
    source_group: str,
    original_id: str = "",
    stage_label: str = "",
    modality: str = "",
    gene_family: str = "",
    sample_family: str = "forewing",
    strict_group: str = "",
    split_group: str = "",
    status: str = "",
    manual_review_status: str = "",
    alignment_source: str = "",
    mask_sizes: list[int] | tuple[int, ...] = DEFAULT_MASK_VECTOR_SIZES,
    include_mask_vectors: bool = True,
    project_dir: str | Path = DEFAULT_FOREWING_PROJECT_DIR,
    gray_override: np.ndarray | None = None,
    mask_override: np.ndarray | None = None,
    mask_source_override: str = "",
) -> dict[str, object]:
    resolved_image = str(Path(image_path).resolve())
    resolved_mask = str(Path(mask_path).resolve()) if mask_path else ""
    if gray_override is not None and mask_override is not None:
        gray = np.asarray(gray_override, dtype=np.uint8)
        mask = np.asarray(mask_override, dtype=bool)
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
        "gene_family": gene_family,
        "sample_family": sample_family or "forewing",
        "stage_label": stage_label,
        "strict_group": strict_group or strict_group_key(original_id, item_id, resolved_image),
        "split_group": split_group or strict_group or strict_group_key(original_id, item_id, resolved_image),
        "status": status,
        "manual_review_status": manual_review_status,
        "alignment_source": alignment_source,
        "mask_source": mask_source,
        "mask_area_pixels": int(mask.sum()),
        "image_height": int(gray.shape[0]),
        "image_width": int(gray.shape[1]),
    }
    row.update({f"shape_{key}": float(value) for key, value in compute_shape_features(mask, project_dir).items()})
    row.update(compute_stage_template_features(mask))
    if include_mask_vectors:
        add_mask_vector_features(row, mask, mask_sizes)
    return row


def apply_forewing_stage_bundle(
    bundle: dict[str, object],
    item_id: str,
    image_path: str | Path,
    mask_path: str | Path | None,
    stage_profile: str = "",
    gray_override: np.ndarray | None = None,
    mask_override: np.ndarray | None = None,
    mask_source_override: str = "",
) -> dict[str, object]:
    feature_row = build_forewing_stage_feature_row(
        item_id=item_id,
        image_path=image_path,
        mask_path=mask_path,
        source_group="dapi_classifier",
        original_id=str(item_id),
        stage_label="",
        modality="dapi",
        gene_family="",
        sample_family="forewing",
        strict_group=f"inference::{str(item_id).lower()}",
        split_group=f"inference::{str(item_id).lower()}",
        include_mask_vectors=True,
        gray_override=gray_override,
        mask_override=mask_override,
        mask_source_override=mask_source_override,
    )
    if str(bundle.get("bundle_kind", "")).strip().lower() == "stage_fusion_v1":
        payload = apply_stage_fusion_bundle(bundle, feature_row, row_matrix, stage_profile=stage_profile)
        return {
            "stage_pred": payload["fused_stage_pred"],
            "stage_conf": payload["fused_stage_conf"],
            "stage_margin": payload["fused_stage_margin"],
            "decision": payload["decision"],
            "probabilities": dict(payload["probabilities"]),
            "feature_row": feature_row,
            **payload,
        }
    stage_labels = [str(label) for label in bundle.get("stage_labels", DEFAULT_STAGE_LABELS)]
    probabilities = np.asarray(stage_probabilities_from_bundle(bundle, feature_row), dtype=np.float64)
    probabilities = probabilities / max(float(probabilities.sum()), 1e-9)
    sorted_prob = np.sort(probabilities)
    pred_index = int(np.argmax(probabilities))
    thresholds = dict(bundle.get("auto_thresholds", {}))
    best_prob = float(probabilities[pred_index])
    margin = float(sorted_prob[-1] - sorted_prob[-2]) if sorted_prob.size >= 2 else best_prob
    decision = "uncertain_review"
    if best_prob >= float(thresholds.get("prob_min", 0.0)) and margin >= float(thresholds.get("margin_min", 0.0)):
        decision = f"auto_forewing_{stage_labels[pred_index]}"
    return {
        "stage_pred": stage_labels[pred_index],
        "stage_conf": best_prob,
        "stage_margin": margin,
        "decision": decision,
        "probabilities": {label: float(probabilities[index]) for index, label in enumerate(stage_labels)},
        "feature_row": feature_row,
    }
