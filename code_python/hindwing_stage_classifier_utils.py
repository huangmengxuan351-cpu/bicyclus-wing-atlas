from __future__ import annotations

import math
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
from pose_landmark_utils import DEFAULT_PROJECT_DIR as DEFAULT_HINDWING_PROJECT_DIR
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
DEFAULT_STAGE_TEMPLATE_ROOT = ROOT / "output_results" / "stage_standard_vein_dapi_atlas"
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


def strict_stage_split_group(item_id: str, image_path: str | Path) -> str:
    key = str(item_id or Path(image_path).stem).strip().lower()
    key = re.sub(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", "", key)
    key = re.sub(r"[^a-z0-9]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    if not key:
        key = Path(image_path).stem.lower()
    return f"hindwing_stage::{key}"


@lru_cache(maxsize=8)
def load_stage_template_mask(stage_label: str, template_root: str) -> np.ndarray:
    stage_key = f"stage_{str(stage_label).replace('.', '_')}"
    mask_path = Path(template_root) / stage_key / "recommended" / "atlas_wing_mask.png"
    return load_binary_mask(mask_path)


@lru_cache(maxsize=8)
def load_stage_template_profiles(stage_label: str, template_root: str) -> dict[str, np.ndarray]:
    mask = load_stage_template_mask(stage_label, template_root)
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
    return {
        "iou": float(iou),
        "dice": float(dice),
        "diff": float(diff),
    }


def compute_stage_template_features(
    mask: np.ndarray,
    stage_labels: list[str] | tuple[str, ...] = DEFAULT_STAGE_LABELS,
    template_root: str | Path = DEFAULT_STAGE_TEMPLATE_ROOT,
) -> dict[str, float]:
    template_root = str(Path(template_root).resolve())
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
        template_mask = load_stage_template_mask(stage_label, template_root)
        template_160 = resize_mask_crop(template_mask, size=160)
        overlap = _mask_overlap_features(sample_160, template_160)
        profiles = load_stage_template_profiles(stage_label, template_root)
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


def build_hindwing_stage_feature_row(
    item_id: str,
    image_path: str | Path,
    mask_path: str | Path | None,
    source_group: str,
    original_id: str = "",
    stage_label: str = "",
    modality: str = "",
    gene_family: str = "",
    sample_family: str = "",
    strict_group: str = "",
    split_group: str = "",
    status: str = "",
    manual_review_status: str = "",
    alignment_source: str = "",
    mask_sizes: list[int] | tuple[int, ...] = DEFAULT_MASK_VECTOR_SIZES,
    include_mask_vectors: bool = True,
    project_dir: str | Path = DEFAULT_HINDWING_PROJECT_DIR,
    template_root: str | Path = DEFAULT_STAGE_TEMPLATE_ROOT,
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
        "sample_family": sample_family,
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
    row.update(compute_stage_template_features(mask, template_root=template_root))
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


def stage_prototype_from_rows(rows: list[dict[str, object]], stage_label: str, mask_size: int) -> np.ndarray:
    prefix = f"mask{int(mask_size)}_px_"
    vectors: list[np.ndarray] = []
    for row in rows:
        if str(row.get("stage_label", "")).strip() != str(stage_label):
            continue
        values = [float(value) for key, value in row.items() if key.startswith(prefix)]
        if values:
            vectors.append(np.asarray(values, dtype=np.float64))
    if not vectors:
        return np.zeros((int(mask_size) * int(mask_size),), dtype=np.float64)
    return np.mean(np.stack(vectors, axis=0), axis=0)


def prototype_stage_probabilities(vector: np.ndarray, prototypes: dict[str, np.ndarray]) -> dict[str, float]:
    vector = np.asarray(vector, dtype=np.float64)
    labels = list(prototypes.keys())
    costs = np.asarray(
        [float(np.mean((vector - np.asarray(prototypes[label], dtype=np.float64)) ** 2)) for label in labels],
        dtype=np.float64,
    )
    if costs.size == 0:
        return {}
    shifted = costs - float(costs.min())
    scale = max(float(np.std(shifted)), 1e-6, 0.01)
    logits = -(shifted / scale)
    logits -= float(logits.max())
    exp = np.exp(logits)
    prob = exp / max(float(exp.sum()), 1e-12)
    return {label: float(prob[index]) for index, label in enumerate(labels)}


def stage_feature_summary(row: dict[str, object]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for name in [
        "shape_notch_depth",
        "shape_solidity",
        "shape_circularity",
        "shape_radial_std",
        "shape_width_std",
        "best_template_margin",
    ]:
        if name not in row:
            continue
        try:
            summary.append({"name": name, "value": float(row[name])})
        except Exception:
            continue
    return summary


def fill_probability_matrix(pred_prob: np.ndarray, stage_labels: list[str] | tuple[str, ...]) -> np.ndarray:
    labels = [str(label) for label in stage_labels]
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((pred_prob.shape[0], len(labels)), dtype=np.float64)
    for column_index, label in enumerate(labels):
        if label not in label_to_index:
            continue
        matrix[:, label_to_index[label]] = pred_prob[:, column_index]
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return matrix / row_sums


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


def stage_probabilities_from_bundle(bundle: dict[str, object], feature_row: dict[str, object]) -> np.ndarray:
    stage_labels = [str(label) for label in bundle.get("stage_labels", DEFAULT_STAGE_LABELS)]
    kind = str(bundle.get("bundle_kind", "")).strip().lower()
    if kind == "logistic_multiclass":
        feature_names = list(bundle["feature_names"])
        pred_prob = np.asarray(bundle["model"].predict_proba(row_matrix([feature_row], feature_names)), dtype=np.float64)
        classes = [str(value) for value in getattr(bundle["model"], "classes_", stage_labels)]
        class_prob = np.zeros((pred_prob.shape[0], len(stage_labels)), dtype=np.float64)
        for column_index, label in enumerate(classes):
            if label not in stage_labels:
                continue
            class_prob[:, stage_labels.index(label)] = pred_prob[:, column_index]
        row_sums = class_prob.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0.0] = 1.0
        return class_prob[0] / row_sums[0]
    if kind == "hist_multiclass":
        feature_names = list(bundle["feature_names"])
        pred_prob = np.asarray(bundle["model"].predict_proba(row_matrix([feature_row], feature_names)), dtype=np.float64)
        classes = [str(value) for value in getattr(bundle["model"], "classes_", stage_labels)]
        class_prob = np.zeros((pred_prob.shape[0], len(stage_labels)), dtype=np.float64)
        for column_index, label in enumerate(classes):
            if label not in stage_labels:
                continue
            class_prob[:, stage_labels.index(label)] = pred_prob[:, column_index]
        row_sums = class_prob.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0.0] = 1.0
        return class_prob[0] / row_sums[0]
    if kind == "ordinal_logistic":
        feature_names = list(bundle["feature_names"])
        matrix = row_matrix([feature_row], feature_names)
        scores: dict[str, float] = {}
        for threshold_name in ["ge_075", "ge_100"]:
            payload = dict(bundle[threshold_name])
            if payload.get("type") == "model":
                scores[threshold_name] = float(payload["model"].predict_proba(matrix)[:, 1][0])
            else:
                scores[threshold_name] = float(payload.get("value", 0.0))
        p_ge_075 = float(np.clip(scores["ge_075"], 1e-6, 1.0 - 1e-6))
        p_ge_100 = float(np.clip(min(scores["ge_100"], p_ge_075), 1e-6, 1.0 - 1e-6))
        prob = np.asarray([1.0 - p_ge_075, p_ge_075 - p_ge_100, p_ge_100], dtype=np.float64)
        return prob / max(float(prob.sum()), 1e-9)
    if kind == "mask_prototype":
        mask_size = int(bundle["mask_size"])
        prefix = f"mask{mask_size}_px_"
        vector = np.asarray([float(value) for key, value in feature_row.items() if key.startswith(prefix)], dtype=np.float64)
        prob_map = prototype_stage_probabilities(
            vector,
            {label: np.asarray(values, dtype=np.float64) for label, values in bundle["stage_prototypes"].items()},
        )
        return np.asarray([prob_map.get(label, 0.0) for label in stage_labels], dtype=np.float64)
    if kind == "template_rule":
        score_kind = str(bundle.get("score_kind", "diff"))
        if score_kind == "diff":
            scores = np.asarray(
                [-float(feature_row.get(f"stage_diff_{label.replace('.', '_')}", 1.0)) for label in stage_labels],
                dtype=np.float64,
            )
        elif score_kind == "dice":
            scores = np.asarray(
                [float(feature_row.get(f"stage_dice_{label.replace('.', '_')}", 0.0)) for label in stage_labels],
                dtype=np.float64,
            )
        else:
            scores = np.asarray(
                [-float(feature_row.get(f"stage_width_l2_{label.replace('.', '_')}", 0.0)) for label in stage_labels],
                dtype=np.float64,
            )
        return softmax(scores)
    if kind == "stage_fusion_v1":
        payload = apply_stage_fusion_bundle(bundle, feature_row, row_matrix)
        return np.asarray([float(payload["probabilities"].get(label, 0.0)) for label in stage_labels], dtype=np.float64)
    raise ValueError(f"Unsupported hindwing stage bundle kind: {kind}")


def apply_hindwing_stage_bundle(
    bundle: dict[str, object],
    item_id: str,
    image_path: str | Path,
    mask_path: str | Path | None,
    stage_profile: str = "",
    gray_override: np.ndarray | None = None,
    mask_override: np.ndarray | None = None,
    mask_source_override: str = "",
) -> dict[str, object]:
    feature_row = build_hindwing_stage_feature_row(
        item_id=item_id,
        image_path=image_path,
        mask_path=mask_path,
        source_group="dapi_classifier",
        original_id=str(item_id),
        stage_label="",
        modality="dapi",
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
    if (
        best_prob >= float(thresholds.get("prob_min", 0.0))
        and margin >= float(thresholds.get("margin_min", 0.0))
    ):
        decision = f"auto_hindwing_{stage_labels[pred_index]}"
    return {
        "stage_pred": stage_labels[pred_index],
        "stage_conf": best_prob,
        "stage_margin": margin,
        "decision": decision,
        "probabilities": {label: float(probabilities[index]) for index, label in enumerate(stage_labels)},
        "feature_row": feature_row,
    }
