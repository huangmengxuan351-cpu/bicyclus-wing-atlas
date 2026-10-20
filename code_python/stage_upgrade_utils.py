from __future__ import annotations

import json
import math
import pickle
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aligned_wing_classifier_utils import load_binary_mask
from pose_landmark_utils import read_csv_rows, write_csv_rows
from sample_identity_utils import canonical_original_id, group_value
from shape_priority_hindwing_utils import extract_best_wing_mask, load_gray_image


STAGE_LABELS = ["0.5", "0.75", "1.0"]
STAGE_TO_INDEX = {label: index for index, label in enumerate(STAGE_LABELS)}
FUSION_WEIGHT_GRID = [
    (0.90, 0.05, 0.03, 0.02),
    (0.80, 0.10, 0.05, 0.05),
    (0.70, 0.15, 0.10, 0.05),
    (0.55, 0.15, 0.15, 0.15),
    (0.50, 0.20, 0.15, 0.15),
    (0.45, 0.20, 0.20, 0.15),
]
TAU_GRID = [0.5, 1.0]
HARD_WEIGHT_CONFIG = {
    "base": 1.0,
    "stage_0_5_multiplier": 1.75,
    "forewing_multiplier": 1.15,
    "misclassified_multiplier": 1.40,
    "low_margin_multiplier": 1.25,
    "weight_cap": 3.0,
}
FOREWING_HARD_WEIGHT_CONFIG = {
    "base": 1.0,
    "stage_0_5_multiplier": 2.0,
    "misclassified_multiplier": 1.5,
    "stage_0_5_to_0_75_multiplier": 1.25,
    "low_margin_multiplier": 1.25,
    "manual_standardized_stage_0_5_multiplier": 1.10,
    "weight_cap": 3.5,
    "low_margin_quantile": 0.30,
    "maturity_low": 0.62,
    "maturity_high": 0.82,
}
EARLY05_AUX_WEIGHT_GRID = [0.0, 0.10, 0.15, 0.20]
EARLY05_RESCUE_DELTA_GRID = [0.08, 0.12, 0.16]
EARLY05_RESCUE_FACTOR_GRID = [1.0, 1.08, 1.12, 1.18]
CALIBRATION_BIN_COUNT = 10
STAGE_NUMERIC_VALUES = np.asarray([0.5, 0.75, 1.0], dtype=np.float64)


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def _safe_group_label(value: object, empty_fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    return text if text else empty_fallback


def _group_accuracy_rows(
    output_rows: list[dict[str, object]],
    fold_field: str,
    label_field: str,
) -> list[dict[str, object]]:
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in output_rows:
        buckets[str(row.get(fold_field, ""))].append(row)
    summary_rows: list[dict[str, object]] = []
    for group_name, rows in sorted(buckets.items()):
        correct = sum(1 for row in rows if str(row.get("correct", "")) == "True")
        summary_rows.append(
            {
                label_field: group_name,
                "n": int(len(rows)),
                "accuracy": float(correct / max(1, len(rows))),
            }
        )
    return summary_rows


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    output = np.asarray(mask, dtype=bool)
    for _ in range(max(1, int(iterations))):
        padded = np.pad(output, 1, mode="constant", constant_values=False)
        neighborhoods = [padded[dy : dy + output.shape[0], dx : dx + output.shape[1]] for dy in range(3) for dx in range(3)]
        output = np.logical_or.reduce(neighborhoods)
    return output


def _binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    output = np.asarray(mask, dtype=bool)
    for _ in range(max(1, int(iterations))):
        padded = np.pad(output, 1, mode="constant", constant_values=False)
        neighborhoods = [padded[dy : dy + output.shape[0], dx : dx + output.shape[1]] for dy in range(3) for dx in range(3)]
        output = np.logical_and.reduce(neighborhoods)
    return output


def _rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    image = Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255), mode="L")
    rotated = image.rotate(float(angle_deg), resample=Image.Resampling.BILINEAR, fillcolor=0)
    return np.asarray(rotated, dtype=np.uint8) > 127


def _perturb_gray(gray: np.ndarray, brightness: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    image = Image.fromarray(np.asarray(gray, dtype=np.uint8), mode="L")
    if not math.isclose(float(brightness), 1.0):
        image = ImageEnhance.Brightness(image).enhance(float(brightness))
    if not math.isclose(float(contrast), 1.0):
        image = ImageEnhance.Contrast(image).enhance(float(contrast))
    return np.asarray(image, dtype=np.uint8)


def _dropout_mask(mask: np.ndarray, seed_text: str, drop_fraction: float = 0.10) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    coords = np.argwhere(output)
    if coords.size == 0:
        return output
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    height = max(1, int(y1 - y0 + 1))
    width = max(1, int(x1 - x0 + 1))
    side = max(1, int(round(math.sqrt(float(drop_fraction) * height * width))))
    digest = hashlib.sha1(str(seed_text).encode("utf-8")).hexdigest()
    seed_a = int(digest[:8], 16)
    seed_b = int(digest[8:16], 16)
    top = int(y0 + (seed_a % max(1, height - side + 1)))
    left = int(x0 + (seed_b % max(1, width - side + 1)))
    output[top : top + side, left : left + side] = False
    return output


def save_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def accuracy_score(truth: list[str], pred: list[str]) -> float:
    return float(sum(t == p for t, p in zip(truth, pred)) / len(truth)) if truth else 0.0


def macro_f1_score(truth: list[str], pred: list[str]) -> float:
    if not truth:
        return 0.0
    return float(f1_score(truth, pred, labels=STAGE_LABELS, average="macro", zero_division=0))


def stage_numeric(label: str) -> float:
    return {"0.5": 0.5, "0.75": 0.75, "1.0": 1.0}.get(str(label), 0.0)


def hard_weight_config_for_family(family_name: str) -> dict[str, float]:
    if str(family_name).strip().lower() == "forewing":
        return dict(FOREWING_HARD_WEIGHT_CONFIG)
    return dict(HARD_WEIGHT_CONFIG)


def model_variant_specs(
    shape_feature_names: list[str],
    template_feature_names: list[str],
    mask32_feature_names: list[str],
) -> dict[str, list[str]]:
    return {
        "logistic_shape_only": list(shape_feature_names),
        "logistic_mask32_only": list(mask32_feature_names),
        "logistic_shape_mask32": list(shape_feature_names) + list(mask32_feature_names),
        "logistic_shape_template": list(shape_feature_names) + list(template_feature_names),
        "logistic_shape_template_mask32": list(shape_feature_names) + list(template_feature_names) + list(mask32_feature_names),
        "hist_shape_only": list(shape_feature_names),
        "hist_mask32_only": list(mask32_feature_names),
        "hist_shape_mask32": list(shape_feature_names) + list(mask32_feature_names),
        "hist_shape_template": list(shape_feature_names) + list(template_feature_names),
        "hist_shape_template_mask32": list(shape_feature_names) + list(template_feature_names) + list(mask32_feature_names),
    }


def candidate_preference_score(candidate_name: str, family_name: str) -> int:
    if str(family_name).strip().lower() != "forewing":
        return 0
    ordered = [
        "logistic_shape_template_mask32",
        "logistic_shape_mask32",
        "logistic_shape_template",
        "hist_shape_template",
        "hist_shape_template_mask32",
        "logistic_shape_only",
        "hist_shape_only",
        "hist_shape_mask32",
        "logistic_mask32_only",
        "hist_mask32_only",
    ]
    return max(0, len(ordered) - ordered.index(candidate_name)) if candidate_name in ordered else 0


def feature_set_label(candidate_name: str) -> str:
    lowered = str(candidate_name).strip().lower()
    if "shape_template_mask32" in lowered:
        return "shape+template+mask"
    if "shape_template" in lowered:
        return "shape+template"
    if "shape_mask32" in lowered:
        return "shape+mask"
    if "mask32_only" in lowered:
        return "mask only"
    if "shape_only" in lowered:
        return "shape only"
    return lowered


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


def softmax_from_costs(costs: np.ndarray, tau: float) -> np.ndarray:
    costs = np.asarray(costs, dtype=np.float64)
    tau = max(float(tau), 1e-6)
    return softmax(-(costs / tau))


def build_logistic_pipeline(C: float, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(C),
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=int(random_state),
                ),
            ),
        ]
    )


def build_hist_model(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=300,
        min_samples_leaf=4,
        l2_regularization=0.02,
        random_state=int(random_state),
    )


def fit_pipeline_with_optional_weights(model: Pipeline, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> Pipeline:
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, model__sample_weight=np.asarray(sample_weight, dtype=np.float64))
    return model


def fill_probability_matrix(pred_prob: np.ndarray, classes: list[str]) -> np.ndarray:
    matrix = np.zeros((pred_prob.shape[0], len(STAGE_LABELS)), dtype=np.float64)
    for column_index, label in enumerate(classes):
        if label in STAGE_TO_INDEX:
            matrix[:, STAGE_TO_INDEX[label]] = pred_prob[:, column_index]
    sums = matrix.sum(axis=1, keepdims=True)
    sums[sums <= 0.0] = 1.0
    return matrix / sums


def groups_and_labels(rows: list[dict[str, object]]) -> tuple[np.ndarray, list[str], np.ndarray]:
    groups = np.asarray([group_value(row) for row in rows], dtype=object)
    truth = [str(row.get("stage_label", "")) for row in rows]
    y_ord = np.asarray([STAGE_TO_INDEX[label] for label in truth], dtype=np.int32)
    return groups, truth, y_ord


def fit_multiclass_logistic(
    rows: list[dict[str, object]],
    feature_names: list[str],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    C: float,
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    X = row_matrix_fn(rows, feature_names)
    y = np.asarray([str(row.get("stage_label", "")) for row in rows], dtype=object)
    pipeline = build_logistic_pipeline(C=C, random_state=random_state)
    return fit_pipeline_with_optional_weights(pipeline, X, y, sample_weight)


def fit_hist_classifier(
    rows: list[dict[str, object]],
    feature_names: list[str],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> HistGradientBoostingClassifier:
    X = row_matrix_fn(rows, feature_names)
    y = np.asarray([str(row.get("stage_label", "")) for row in rows], dtype=object)
    model = build_hist_model(random_state)
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=np.asarray(sample_weight, dtype=np.float64))
    return model


def fit_binary_logistic(
    rows: list[dict[str, object]],
    feature_names: list[str],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    positive_fn: Callable[[dict[str, object]], bool],
    C: float,
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> Pipeline | None:
    X = row_matrix_fn(rows, feature_names)
    y = np.asarray([1 if positive_fn(row) else 0 for row in rows], dtype=np.int32)
    if len(set(y.tolist())) < 2:
        return None
    model = build_logistic_pipeline(C=C, random_state=random_state)
    return fit_pipeline_with_optional_weights(model, X, y, sample_weight)


def train_multiclass_variant(
    variant_name: str,
    rows: list[dict[str, object]],
    feature_names: list[str],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    random_state: int,
    logistic_c: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, object]:
    if str(variant_name).startswith("logistic_"):
        model = fit_multiclass_logistic(
            rows,
            feature_names=feature_names,
            row_matrix_fn=row_matrix_fn,
            C=logistic_c,
            random_state=random_state,
            sample_weight=sample_weight,
        )
        return {
            "bundle_kind": "logistic_multiclass",
            "candidate": variant_name,
            "feature_names": list(feature_names),
            "stage_labels": list(STAGE_LABELS),
            "model": model,
        }
    model = fit_hist_classifier(
        rows,
        feature_names=feature_names,
        row_matrix_fn=row_matrix_fn,
        random_state=random_state,
        sample_weight=sample_weight,
    )
    return {
        "bundle_kind": "hist_multiclass",
        "candidate": variant_name,
        "feature_names": list(feature_names),
        "stage_labels": list(STAGE_LABELS),
        "model": model,
    }


def multiclass_variant_probabilities(
    bundle: dict[str, object],
    rows: list[dict[str, object]],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
) -> np.ndarray:
    feature_names = list(bundle["feature_names"])
    pred_prob = np.asarray(bundle["model"].predict_proba(row_matrix_fn(rows, feature_names)), dtype=np.float64)
    classes = [str(value) for value in getattr(bundle["model"], "classes_", STAGE_LABELS)]
    return fill_probability_matrix(pred_prob, classes)


def evaluate_feature_model_cv(
    rows: list[dict[str, object]],
    feature_names: list[str],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    builder: Callable[[int], Pipeline],
    random_state: int,
) -> dict[str, object]:
    groups, truth, _ = groups_and_labels(rows)
    X = row_matrix_fn(rows, feature_names)
    y = np.asarray(truth, dtype=object)
    splitter = GroupKFold(n_splits=max(2, min(5, len(sorted(set(groups.tolist()))))))
    oof_prob = np.zeros((len(rows), len(STAGE_LABELS)), dtype=np.float64)
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups=groups)):
        model = builder(random_state + fold_index + 1)
        model.fit(X[train_idx], y[train_idx])
        pred_prob = np.asarray(model.predict_proba(X[val_idx]), dtype=np.float64)
        classes = [str(value) for value in getattr(model, "classes_", STAGE_LABELS)]
        oof_prob[val_idx] = fill_probability_matrix(pred_prob, classes)
    pred, best_prob, margin = probs_to_pred_and_margin(oof_prob)
    return {
        "accuracy": accuracy_score(truth, pred),
        "macro_f1": macro_f1_score(truth, pred),
        "oof_probability": oof_prob,
        "predicted_labels": pred,
        "best_probability": best_prob,
        "margin": margin,
    }


def evaluate_hist_model_cv(
    rows: list[dict[str, object]],
    feature_names: list[str],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    random_state: int,
) -> dict[str, object]:
    return evaluate_feature_model_cv(
        rows,
        feature_names=feature_names,
        row_matrix_fn=row_matrix_fn,
        builder=lambda state: build_hist_model(state),
        random_state=random_state,
    )


def _constant_binary_probability(y_binary: np.ndarray) -> float:
    return float(np.mean(y_binary.astype(np.float64))) if y_binary.size else 0.0


def fit_ordinal_models(
    rows: list[dict[str, object]],
    feature_names: list[str],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    C: float,
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> dict[str, object]:
    X = row_matrix_fn(rows, feature_names)
    _, _, y_ord = groups_and_labels(rows)
    bundles: dict[str, object] = {"feature_names": list(feature_names), "stage_labels": list(STAGE_LABELS)}
    for offset, threshold_name in enumerate(["ge_075", "ge_100"]):
        threshold = 1 if threshold_name == "ge_075" else 2
        y_binary = (y_ord >= threshold).astype(np.int32)
        if len(set(y_binary.tolist())) >= 2:
            model = build_logistic_pipeline(C=C, random_state=random_state + offset + 1)
            model = fit_pipeline_with_optional_weights(model, X, y_binary, sample_weight)
            bundles[threshold_name] = {"type": "model", "model": model}
        else:
            bundles[threshold_name] = {"type": "constant", "value": _constant_binary_probability(y_binary)}
    return bundles


def ordinal_probabilities_from_models(
    bundle: dict[str, object],
    rows: list[dict[str, object]],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
) -> np.ndarray:
    feature_names = list(bundle["feature_names"])
    matrix = row_matrix_fn(rows, feature_names)
    scores: dict[str, np.ndarray] = {}
    for threshold_name in ["ge_075", "ge_100"]:
        payload = dict(bundle[threshold_name])
        if payload.get("type") == "model":
            scores[threshold_name] = np.asarray(payload["model"].predict_proba(matrix)[:, 1], dtype=np.float64)
        else:
            scores[threshold_name] = np.full((len(rows),), float(payload.get("value", 0.0)), dtype=np.float64)
    p_ge_075 = np.clip(scores["ge_075"], 1e-6, 1.0 - 1e-6)
    p_ge_100 = np.clip(np.minimum(scores["ge_100"], p_ge_075), 1e-6, 1.0 - 1e-6)
    prob = np.column_stack([1.0 - p_ge_075, p_ge_075 - p_ge_100, p_ge_100]).astype(np.float64)
    sums = prob.sum(axis=1, keepdims=True)
    sums[sums <= 0.0] = 1.0
    return prob / sums


def choose_auto_thresholds(
    truth: list[str],
    pred: list[str],
    best_prob: np.ndarray,
    margin: np.ndarray,
    target_precision: float,
) -> dict[str, float]:
    best = {
        "prob_min": 1.01,
        "margin_min": 1.01,
        "accepted_precision": 1.0,
        "accepted_coverage": 0.0,
    }
    prob_grid = sorted({0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, *best_prob.tolist()})
    margin_grid = sorted({0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, *margin.tolist()})
    for prob_min in prob_grid:
        for margin_min in margin_grid:
            accepted = [index for index in range(len(truth)) if best_prob[index] >= prob_min and margin[index] >= margin_min]
            if not accepted:
                continue
            accepted_truth = [truth[index] for index in accepted]
            accepted_pred = [pred[index] for index in accepted]
            precision = accuracy_score(accepted_truth, accepted_pred)
            coverage = float(len(accepted) / max(1, len(truth)))
            if precision + 1e-9 < target_precision:
                continue
            if (
                coverage > float(best["accepted_coverage"]) + 1e-9
                or (
                    math.isclose(coverage, float(best["accepted_coverage"]), abs_tol=1e-9)
                    and precision > float(best["accepted_precision"]) + 1e-9
                )
            ):
                best = {
                    "prob_min": float(prob_min),
                    "margin_min": float(margin_min),
                    "accepted_precision": float(precision),
                    "accepted_coverage": float(coverage),
                }
    if best["prob_min"] > 1.0:
        return {"prob_min": 1.1, "margin_min": 1.1, "accepted_precision": 1.0, "accepted_coverage": 0.0}
    return best


def accepted_metrics(
    truth: list[str],
    pred: list[str],
    best_prob: np.ndarray,
    margin: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, float]:
    accepted = [
        index
        for index in range(len(truth))
        if best_prob[index] >= float(thresholds.get("prob_min", 0.0))
        and margin[index] >= float(thresholds.get("margin_min", 0.0))
    ]
    if not accepted:
        return {"accepted_n": 0.0, "coverage": 0.0, "precision": 0.0}
    accepted_truth = [truth[index] for index in accepted]
    accepted_pred = [pred[index] for index in accepted]
    return {
        "accepted_n": float(len(accepted)),
        "coverage": float(len(accepted) / max(1, len(truth))),
        "precision": accuracy_score(accepted_truth, accepted_pred),
    }


def compute_reference_bank(
    rows: list[dict[str, object]],
    template_feature_names: list[str],
    reference_feature_names: list[str],
    mask_feature_names: list[str],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
) -> dict[str, object]:
    template_matrix = row_matrix_fn(rows, template_feature_names)
    reference_matrix = row_matrix_fn(rows, reference_feature_names)
    mask_matrix = row_matrix_fn(rows, mask_feature_names)
    reference_scale = np.std(reference_matrix, axis=0) if len(rows) else np.ones((len(reference_feature_names),), dtype=np.float64)
    reference_scale = np.where(reference_scale <= 1e-6, 1.0, reference_scale)
    template_scale = np.std(template_matrix, axis=0) if len(rows) else np.ones((len(template_feature_names),), dtype=np.float64)
    template_scale = np.where(template_scale <= 1e-6, 1.0, template_scale)
    mask_scale = np.std(mask_matrix, axis=0) if len(rows) else np.ones((len(mask_feature_names),), dtype=np.float64)
    mask_scale = np.where(mask_scale <= 1e-6, 1.0, mask_scale)

    banks: dict[str, object] = {
        "stage_labels": list(STAGE_LABELS),
        "template_feature_names": list(template_feature_names),
        "reference_feature_names": list(reference_feature_names),
        "mask_feature_names": list(mask_feature_names),
        "reference_scale": reference_scale,
        "template_scale": template_scale,
        "mask_scale": mask_scale,
        "per_stage": {},
    }
    for label in STAGE_LABELS:
        stage_indices = [index for index, row in enumerate(rows) if str(row.get("stage_label", "")) == label]
        if not stage_indices:
            banks["per_stage"][label] = {
                "template_centroid": np.zeros((len(template_feature_names),), dtype=np.float64),
                "reference_centroid": np.zeros((len(reference_feature_names),), dtype=np.float64),
                "mask_vectors": np.zeros((0, len(mask_feature_names)), dtype=np.float64),
                "reference_vectors": np.zeros((0, len(reference_feature_names)), dtype=np.float64),
                "reference_ids": [],
            }
            continue
        banks["per_stage"][label] = {
            "template_centroid": template_matrix[stage_indices].mean(axis=0),
            "reference_centroid": reference_matrix[stage_indices].mean(axis=0),
            "mask_vectors": mask_matrix[stage_indices],
            "reference_vectors": reference_matrix[stage_indices],
            "reference_ids": [str(rows[index].get("id", "")) for index in stage_indices],
        }
    return banks


def reference_bank_posteriors(
    feature_row: dict[str, object],
    bank: dict[str, object],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    tau_template: float,
    tau_reference: float,
) -> dict[str, object]:
    template_feature_names = list(bank["template_feature_names"])
    reference_feature_names = list(bank["reference_feature_names"])
    mask_feature_names = list(bank["mask_feature_names"])
    template_vec = row_matrix_fn([feature_row], template_feature_names)[0]
    reference_vec = row_matrix_fn([feature_row], reference_feature_names)[0]
    mask_vec = row_matrix_fn([feature_row], mask_feature_names)[0]
    reference_scale = np.asarray(bank["reference_scale"], dtype=np.float64)
    template_scale = np.asarray(bank["template_scale"], dtype=np.float64)
    mask_scale = np.asarray(bank["mask_scale"], dtype=np.float64)

    template_costs: list[float] = []
    reference_costs: list[float] = []
    nn_distances: list[float] = []
    topk_distances: list[float] = []
    feature_distances: list[float] = []
    nearest_ids: list[str] = []
    for label in STAGE_LABELS:
        stage_payload = dict(bank["per_stage"][label])
        template_centroid = np.asarray(stage_payload["template_centroid"], dtype=np.float64)
        reference_centroid = np.asarray(stage_payload["reference_centroid"], dtype=np.float64)
        reference_vectors = np.asarray(stage_payload["reference_vectors"], dtype=np.float64)
        mask_vectors = np.asarray(stage_payload["mask_vectors"], dtype=np.float64)
        template_cost = float(np.sqrt(np.mean(((template_vec - template_centroid) / template_scale) ** 2)))
        feature_distance = float(np.sqrt(np.mean(((reference_vec - reference_centroid) / reference_scale) ** 2)))
        if len(mask_vectors) == 0:
            nn_distance = 1e6
            topk_distance = 1e6
            nearest_id = ""
        else:
            mask_distances = np.sqrt(np.mean(((mask_vectors - mask_vec.reshape(1, -1)) / mask_scale.reshape(1, -1)) ** 2, axis=1))
            order = np.argsort(mask_distances)
            nn_distance = float(mask_distances[order[0]])
            topk_distance = float(np.mean(mask_distances[order[: min(5, len(order))]]))
            ids = list(stage_payload["reference_ids"])
            nearest_id = ids[int(order[0])] if ids else ""
        composite_reference_cost = 0.50 * feature_distance + 0.30 * nn_distance + 0.20 * topk_distance
        template_costs.append(template_cost)
        reference_costs.append(composite_reference_cost)
        nn_distances.append(nn_distance)
        topk_distances.append(topk_distance)
        feature_distances.append(feature_distance)
        nearest_ids.append(nearest_id)
    p_template = softmax_from_costs(np.asarray(template_costs, dtype=np.float64), tau_template)
    p_reference = softmax_from_costs(np.asarray(reference_costs, dtype=np.float64), tau_reference)
    best_index = int(np.argmin(reference_costs)) if reference_costs else 0
    return {
        "template_probabilities": {label: float(p_template[index]) for index, label in enumerate(STAGE_LABELS)},
        "reference_probabilities": {label: float(p_reference[index]) for index, label in enumerate(STAGE_LABELS)},
        "template_costs": {label: float(template_costs[index]) for index, label in enumerate(STAGE_LABELS)},
        "reference_costs": {label: float(reference_costs[index]) for index, label in enumerate(STAGE_LABELS)},
        "reference_feature_distances": {label: float(feature_distances[index]) for index, label in enumerate(STAGE_LABELS)},
        "reference_nn_distances": {label: float(nn_distances[index]) for index, label in enumerate(STAGE_LABELS)},
        "reference_topk_distances": {label: float(topk_distances[index]) for index, label in enumerate(STAGE_LABELS)},
        "best_reference_stage": STAGE_LABELS[best_index],
        "best_reference_id": nearest_ids[best_index],
        "best_reference_distance": float(nn_distances[best_index]),
        "p_template": p_template,
        "p_reference": p_reference,
    }


def blend_ordinal_probabilities(prob_a: np.ndarray, prob_b: np.ndarray) -> np.ndarray:
    probs = 0.5 * np.asarray(prob_a, dtype=np.float64) + 0.5 * np.asarray(prob_b, dtype=np.float64)
    sums = probs.sum(axis=1, keepdims=True)
    sums[sums <= 0.0] = 1.0
    return probs / sums


def fuse_probabilities(
    p_model: np.ndarray,
    p_ordinal: np.ndarray,
    p_template: np.ndarray,
    p_reference: np.ndarray,
    fusion_weights: tuple[float, float, float, float],
) -> np.ndarray:
    w_model, w_ordinal, w_template, w_reference = [float(value) for value in fusion_weights]
    fused = (
        w_model * np.asarray(p_model, dtype=np.float64)
        + w_ordinal * np.asarray(p_ordinal, dtype=np.float64)
        + w_template * np.asarray(p_template, dtype=np.float64)
        + w_reference * np.asarray(p_reference, dtype=np.float64)
    )
    sums = fused.sum(axis=1, keepdims=True)
    sums[sums <= 0.0] = 1.0
    return fused / sums


def probs_to_pred_and_margin(prob: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    pred = [STAGE_LABELS[int(np.argmax(row))] for row in prob]
    best_prob = np.max(prob, axis=1)
    sorted_prob = np.sort(prob, axis=1)
    margin = sorted_prob[:, -1] - sorted_prob[:, -2] if prob.shape[1] >= 2 else best_prob
    return pred, best_prob, margin


def maturity_from_probabilities(prob: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(prob, dtype=np.float64) * STAGE_NUMERIC_VALUES.reshape(1, -1), axis=1)


def run_weighted_inner_baseline(
    rows: list[dict[str, object]],
    family_name: str,
    feature_names: list[str],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    random_state: int,
) -> dict[str, object]:
    del family_name
    groups, truth, _ = groups_and_labels(rows)
    X = row_matrix_fn(rows, feature_names)
    unique_groups = sorted(set(groups.tolist()))
    splitter = GroupKFold(n_splits=max(2, min(5, len(unique_groups))))
    oof_prob = np.zeros((len(rows), len(STAGE_LABELS)), dtype=np.float64)
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X, truth, groups=groups)):
        train_rows = [rows[index] for index in train_idx.tolist()]
        model = fit_hist_classifier(
            train_rows,
            feature_names=feature_names,
            row_matrix_fn=row_matrix_fn,
            random_state=random_state + fold_index + 1,
            sample_weight=None,
        )
        pred_prob = np.asarray(model.predict_proba(X[val_idx]), dtype=np.float64)
        classes = [str(value) for value in getattr(model, "classes_", STAGE_LABELS)]
        oof_prob[val_idx] = fill_probability_matrix(pred_prob, classes)
    pred, _, margin = probs_to_pred_and_margin(oof_prob)
    return {
        "predicted_labels": pred,
        "margin": margin,
        "probabilities": oof_prob,
        "maturity_score": maturity_from_probabilities(oof_prob),
    }


def compute_hard_case_payload(
    rows: list[dict[str, object]],
    family_name: str,
    baseline_feature_names: list[str],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    random_state: int,
) -> dict[str, object]:
    baseline = run_weighted_inner_baseline(
        rows,
        family_name=family_name,
        feature_names=baseline_feature_names,
        row_matrix_fn=row_matrix_fn,
        random_state=random_state,
    )
    baseline_pred = list(baseline["predicted_labels"])
    baseline_margin = np.asarray(baseline["margin"], dtype=np.float64)
    baseline_prob = np.asarray(baseline["probabilities"], dtype=np.float64)
    baseline_maturity = np.asarray(baseline["maturity_score"], dtype=np.float64)
    config = hard_weight_config_for_family(family_name)
    low_margin_quantile = float(config.get("low_margin_quantile", 0.25))
    margin_05_075 = baseline_prob[:, STAGE_TO_INDEX["0.5"]] - baseline_prob[:, STAGE_TO_INDEX["0.75"]]
    margin_threshold = float(np.quantile(margin_05_075, low_margin_quantile)) if margin_05_075.size else 0.0
    weights = np.full((len(rows),), float(config["base"]), dtype=np.float64)
    hard_flags: list[dict[str, bool]] = []
    for index, row in enumerate(rows):
        stage_label = str(row.get("stage_label", ""))
        source_group = str(row.get("source_group", ""))
        misclassified = baseline_pred[index] != stage_label
        low_margin = bool(margin_05_075[index] <= margin_threshold)
        maturity_band = bool(
            stage_label == "0.5"
            and float(config.get("maturity_low", -1.0)) <= float(baseline_maturity[index]) <= float(config.get("maturity_high", 2.0))
        )
        stage_0_5_to_0_75 = bool(stage_label == "0.5" and baseline_pred[index] == "0.75")
        if stage_label == "0.5":
            weights[index] *= float(config["stage_0_5_multiplier"])
            if source_group == "manual_standardized_forewing_dapi" and "manual_standardized_stage_0_5_multiplier" in config:
                weights[index] *= float(config["manual_standardized_stage_0_5_multiplier"])
        if str(family_name).strip().lower() == "forewing" and "forewing_multiplier" in config:
            weights[index] *= float(config["forewing_multiplier"])
        if misclassified:
            weights[index] *= float(config["misclassified_multiplier"])
        if stage_0_5_to_0_75 and "stage_0_5_to_0_75_multiplier" in config:
            weights[index] *= float(config["stage_0_5_to_0_75_multiplier"])
        if low_margin or maturity_band:
            weights[index] *= float(config["low_margin_multiplier"])
        hard_flags.append(
            {
                "misclassified": misclassified,
                "low_margin": low_margin,
                "maturity_band": maturity_band,
                "stage_0_5_to_0_75": stage_0_5_to_0_75,
            }
        )
    return {
        "weights": np.clip(weights, 0.0, float(config["weight_cap"])),
        "baseline_pred": baseline_pred,
        "baseline_margin": baseline_margin,
        "baseline_prob": baseline_prob,
        "baseline_maturity_score": baseline_maturity,
        "baseline_margin_05_075": margin_05_075,
        "hard_flags": hard_flags,
        "config": config,
    }


def compute_hard_case_sample_weights(
    rows: list[dict[str, object]],
    family_name: str,
    baseline_feature_names: list[str],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    random_state: int,
) -> np.ndarray:
    payload = compute_hard_case_payload(
        rows,
        family_name=family_name,
        baseline_feature_names=baseline_feature_names,
        row_matrix_fn=row_matrix_fn,
        random_state=random_state,
    )
    return np.asarray(payload["weights"], dtype=np.float64)


def evaluate_model_variants_cv(
    rows: list[dict[str, object]],
    model_variant_features: dict[str, list[str]],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    random_state: int,
    logistic_c: float,
    family_name: str,
    sample_weight: np.ndarray | None = None,
) -> list[dict[str, object]]:
    groups, truth, _ = groups_and_labels(rows)
    unique_groups = sorted(set(groups.tolist()))
    splitter = GroupKFold(n_splits=max(2, min(5, len(unique_groups))))
    variant_probs = {
        name: np.zeros((len(rows), len(STAGE_LABELS)), dtype=np.float64)
        for name in model_variant_features
    }
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(np.zeros((len(rows), 1)), truth, groups=groups)):
        train_rows = [rows[index] for index in train_idx.tolist()]
        val_rows = [rows[index] for index in val_idx.tolist()]
        train_weights = np.asarray(sample_weight[train_idx], dtype=np.float64) if sample_weight is not None else None
        for offset, (variant_name, variant_features) in enumerate(model_variant_features.items()):
            bundle = train_multiclass_variant(
                variant_name,
                train_rows,
                feature_names=variant_features,
                row_matrix_fn=row_matrix_fn,
                random_state=random_state + fold_index * 100 + offset + 1,
                logistic_c=logistic_c,
                sample_weight=train_weights,
            )
            variant_probs[variant_name][val_idx] = multiclass_variant_probabilities(bundle, val_rows, row_matrix_fn)
    metrics: list[dict[str, object]] = []
    for variant_name, variant_prob in variant_probs.items():
        pred, best_prob, margin = probs_to_pred_and_margin(variant_prob)
        metrics.append(
            {
                "candidate": variant_name,
                "family_name": family_name,
                "accuracy": accuracy_score(truth, pred),
                "macro_f1": macro_f1_score(truth, pred),
                "predicted_labels": pred,
                "best_probability": best_prob,
                "margin": margin,
                "probabilities": variant_prob,
                "feature_names": list(model_variant_features[variant_name]),
                "preference_score": candidate_preference_score(variant_name, family_name),
            }
        )
    metrics.sort(
        key=lambda row: (
            float(row["accuracy"]),
            float(row["macro_f1"]),
            int(row.get("preference_score", 0)),
        ),
        reverse=True,
    )
    return metrics


def augment_model_probabilities(p_model: np.ndarray, early05_prob: np.ndarray, aux_weight: float) -> np.ndarray:
    aux_weight = float(aux_weight)
    if aux_weight <= 0.0:
        return np.asarray(p_model, dtype=np.float64)
    p_model = np.asarray(p_model, dtype=np.float64)
    early05_prob = np.clip(np.asarray(early05_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    updated = np.asarray(p_model, dtype=np.float64).copy()
    updated[:, 0] = np.clip((1.0 - aux_weight) * p_model[:, 0] + aux_weight * early05_prob, 1e-6, 1.0 - 1e-6)
    tail = np.asarray(p_model[:, 1:], dtype=np.float64)
    tail_sum = np.sum(tail, axis=1, keepdims=True)
    tail_sum[tail_sum <= 0.0] = 1.0
    updated[:, 1:] = ((1.0 - updated[:, 0]).reshape(-1, 1) * (tail / tail_sum)).astype(np.float64)
    sums = updated.sum(axis=1, keepdims=True)
    sums[sums <= 0.0] = 1.0
    return updated / sums


def apply_early05_rescue(
    p_model: np.ndarray,
    p_ordinal: np.ndarray,
    p_template: np.ndarray,
    p_reference: np.ndarray,
    p_fused: np.ndarray,
    early05_prob: np.ndarray,
    delta_model: float,
    rescue_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    rescued = np.asarray(p_fused, dtype=np.float64).copy()
    early05_prob = np.asarray(early05_prob, dtype=np.float64)
    maturity_score = maturity_from_probabilities(rescued)
    apply_flags = (
        (np.argmax(p_model, axis=1) == STAGE_TO_INDEX["0.5"])
        & (np.argmax(p_ordinal, axis=1) == STAGE_TO_INDEX["0.75"])
        & ((p_model[:, STAGE_TO_INDEX["0.5"]] - p_model[:, STAGE_TO_INDEX["0.75"]]) >= float(delta_model))
        & (early05_prob >= 0.50)
        & (p_template[:, STAGE_TO_INDEX["1.0"]] < 0.42)
        & (p_reference[:, STAGE_TO_INDEX["1.0"]] < 0.42)
        & (maturity_score <= 0.78)
    )
    if float(rescue_factor) > 1.0:
        rescued[apply_flags, STAGE_TO_INDEX["0.5"]] *= float(rescue_factor)
        sums = rescued.sum(axis=1, keepdims=True)
        sums[sums <= 0.0] = 1.0
        rescued = rescued / sums
    return rescued, apply_flags.astype(bool)

def evaluate_components_oof(
    rows: list[dict[str, object]],
    shape_feature_names: list[str],
    template_feature_names: list[str],
    mask32_feature_names: list[str],
    mask64_feature_names: list[str],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    random_state: int,
    family_name: str,
    logistic_c: float = 1.0,
) -> dict[str, object]:
    groups, truth, _ = groups_and_labels(rows)
    unique_groups = sorted(set(groups.tolist()))
    splitter = GroupKFold(n_splits=max(2, min(5, len(unique_groups))))
    model_variant_features = model_variant_specs(shape_feature_names, template_feature_names, mask32_feature_names)
    model_variant_probs = {
        name: np.zeros((len(rows), len(STAGE_LABELS)), dtype=np.float64)
        for name in model_variant_features
    }
    selected_model_prob = np.zeros((len(rows), len(STAGE_LABELS)), dtype=np.float64)
    selected_model_candidate = np.asarray([""] * len(rows), dtype=object)
    ord_template_prob = np.zeros((len(rows), len(STAGE_LABELS)), dtype=np.float64)
    ord_shape_prob = np.zeros((len(rows), len(STAGE_LABELS)), dtype=np.float64)
    early05_prob = np.zeros((len(rows),), dtype=np.float64)
    p_template_grid: dict[tuple[float, float], np.ndarray] = {}
    p_reference_grid: dict[tuple[float, float], np.ndarray] = {}
    template_cost_grid: dict[tuple[float, float], list[dict[str, float]]] = {}
    reference_cost_grid: dict[tuple[float, float], list[dict[str, float]]] = {}
    reference_best_distance = np.zeros((len(rows),), dtype=np.float64)
    hard_case_payload = compute_hard_case_payload(
        rows,
        family_name=family_name,
        baseline_feature_names=shape_feature_names + template_feature_names,
        row_matrix_fn=row_matrix_fn,
        random_state=random_state + 5,
    )
    hard_weights = np.asarray(hard_case_payload["weights"], dtype=np.float64)
    for tau_template in TAU_GRID:
        for tau_reference in TAU_GRID:
            p_template_grid[(tau_template, tau_reference)] = np.zeros((len(rows), len(STAGE_LABELS)), dtype=np.float64)
            p_reference_grid[(tau_template, tau_reference)] = np.zeros((len(rows), len(STAGE_LABELS)), dtype=np.float64)
            template_cost_grid[(tau_template, tau_reference)] = [{} for _ in rows]
            reference_cost_grid[(tau_template, tau_reference)] = [{} for _ in rows]

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(np.zeros((len(rows), 1)), truth, groups=groups)):
        train_rows = [rows[index] for index in train_idx.tolist()]
        val_rows = [rows[index] for index in val_idx.tolist()]
        train_weights = hard_weights[train_idx]
        inner_variant_metrics = evaluate_model_variants_cv(
            train_rows,
            model_variant_features=model_variant_features,
            row_matrix_fn=row_matrix_fn,
            random_state=random_state + 40 + fold_index * 17,
            logistic_c=logistic_c,
            family_name=family_name,
            sample_weight=train_weights,
        )
        chosen_variant = str(inner_variant_metrics[0]["candidate"])

        for offset, (variant_name, variant_features) in enumerate(model_variant_features.items()):
            model_bundle = train_multiclass_variant(
                variant_name,
                train_rows,
                feature_names=variant_features,
                row_matrix_fn=row_matrix_fn,
                random_state=random_state + 100 + fold_index * 20 + offset,
                logistic_c=logistic_c,
                sample_weight=train_weights,
            )
            model_prob = multiclass_variant_probabilities(model_bundle, val_rows, row_matrix_fn)
            model_variant_probs[variant_name][val_idx] = model_prob
            if variant_name == chosen_variant:
                selected_model_prob[val_idx] = model_prob
                selected_model_candidate[val_idx] = variant_name

        ordinal_template_bundle = fit_ordinal_models(
            train_rows,
            feature_names=template_feature_names + mask64_feature_names,
            row_matrix_fn=row_matrix_fn,
            C=logistic_c,
            random_state=random_state + 200 + fold_index * 10,
            sample_weight=train_weights,
        )
        ord_template_prob[val_idx] = ordinal_probabilities_from_models(ordinal_template_bundle, val_rows, row_matrix_fn)

        ordinal_shape_bundle = fit_ordinal_models(
            train_rows,
            feature_names=shape_feature_names + template_feature_names + mask32_feature_names,
            row_matrix_fn=row_matrix_fn,
            C=logistic_c,
            random_state=random_state + 300 + fold_index * 10,
            sample_weight=train_weights,
        )
        ord_shape_prob[val_idx] = ordinal_probabilities_from_models(ordinal_shape_bundle, val_rows, row_matrix_fn)

        early05_bundle = fit_binary_logistic(
            train_rows,
            feature_names=shape_feature_names + template_feature_names + mask32_feature_names,
            row_matrix_fn=row_matrix_fn,
            positive_fn=lambda row: str(row.get("stage_label", "")) == "0.5",
            C=logistic_c,
            random_state=random_state + 350 + fold_index * 10,
            sample_weight=train_weights,
        )
        if early05_bundle is None:
            early05_prob[val_idx] = 0.5
        else:
            early05_prob[val_idx] = np.asarray(
                early05_bundle.predict_proba(row_matrix_fn(val_rows, shape_feature_names + template_feature_names + mask32_feature_names))[:, 1],
                dtype=np.float64,
            )

        bank = compute_reference_bank(
            train_rows,
            template_feature_names=template_feature_names,
            reference_feature_names=shape_feature_names + template_feature_names,
            mask_feature_names=mask32_feature_names,
            row_matrix_fn=row_matrix_fn,
        )
        for offset, row_index in enumerate(val_idx.tolist()):
            feature_row = rows[row_index]
            raw_eval = reference_bank_posteriors(
                feature_row,
                bank=bank,
                row_matrix_fn=row_matrix_fn,
                tau_template=1.0,
                tau_reference=1.0,
            )
            reference_best_distance[row_index] = float(raw_eval["best_reference_distance"])
            for tau_template in TAU_GRID:
                for tau_reference in TAU_GRID:
                    payload = reference_bank_posteriors(
                        feature_row,
                        bank=bank,
                        row_matrix_fn=row_matrix_fn,
                        tau_template=tau_template,
                        tau_reference=tau_reference,
                    )
                    key = (tau_template, tau_reference)
                    p_template_grid[key][row_index] = np.asarray(payload["p_template"], dtype=np.float64)
                    p_reference_grid[key][row_index] = np.asarray(payload["p_reference"], dtype=np.float64)
                    template_cost_grid[key][row_index] = dict(payload["template_costs"])
                    reference_cost_grid[key][row_index] = dict(payload["reference_costs"])

    model_variant_metrics: list[dict[str, object]] = []
    for variant_name, variant_prob in model_variant_probs.items():
        variant_pred, variant_best_prob, variant_margin = probs_to_pred_and_margin(variant_prob)
        model_variant_metrics.append(
            {
                "candidate": variant_name,
                "accuracy": accuracy_score(truth, variant_pred),
                "macro_f1": macro_f1_score(truth, variant_pred),
                "predicted_labels": variant_pred,
                "best_probability": variant_best_prob,
                "margin": variant_margin,
                "probabilities": variant_prob,
                "feature_names": model_variant_features[variant_name],
                "preference_score": candidate_preference_score(variant_name, family_name),
            }
        )
    model_variant_metrics.sort(
        key=lambda row: (
            float(row["accuracy"]),
            float(row["macro_f1"]),
            int(row.get("preference_score", 0)),
        ),
        reverse=True,
    )
    best_model_variant = model_variant_metrics[0]
    base_model_prob = np.asarray(best_model_variant["probabilities"], dtype=np.float64)
    selected_model_candidate = np.asarray([str(best_model_variant["candidate"])] * len(rows), dtype=object)
    selected_model_pred, _, _ = probs_to_pred_and_margin(base_model_prob)
    ordinal_blend = blend_ordinal_probabilities(ord_template_prob, ord_shape_prob)
    best_fusion: dict[str, object] = {}
    fusion_candidates: list[dict[str, object]] = []
    for aux_weight in EARLY05_AUX_WEIGHT_GRID if str(family_name).strip().lower() == "forewing" else [0.0]:
        model_prob = augment_model_probabilities(base_model_prob, early05_prob, aux_weight)
        for weights in FUSION_WEIGHT_GRID:
            for tau_template in TAU_GRID:
                for tau_reference in TAU_GRID:
                    key = (tau_template, tau_reference)
                    base_fused_prob = fuse_probabilities(
                        model_prob,
                        ordinal_blend,
                        p_template_grid[key],
                        p_reference_grid[key],
                        fusion_weights=weights,
                    )
                    if str(family_name).strip().lower() == "forewing":
                        for delta_model in EARLY05_RESCUE_DELTA_GRID:
                            for rescue_factor in EARLY05_RESCUE_FACTOR_GRID:
                                fused_prob, rescue_applied = apply_early05_rescue(
                                    model_prob,
                                    ordinal_blend,
                                    p_template_grid[key],
                                    p_reference_grid[key],
                                    base_fused_prob,
                                    early05_prob,
                                    delta_model=delta_model,
                                    rescue_factor=rescue_factor,
                                )
                                fused_pred, _, _ = probs_to_pred_and_margin(fused_prob)
                                fusion_candidates.append(
                                    {
                                        "fusion_weights": tuple(float(value) for value in weights),
                                        "tau_template": float(tau_template),
                                        "tau_reference": float(tau_reference),
                                        "aux_weight": float(aux_weight),
                                        "delta_model": float(delta_model),
                                        "rescue_factor": float(rescue_factor),
                                        "accuracy": accuracy_score(truth, fused_pred),
                                        "macro_f1": macro_f1_score(truth, fused_pred),
                                        "probabilities": fused_prob,
                                        "model_probabilities": model_prob,
                                        "rescue_applied": rescue_applied,
                                    }
                                )
                    else:
                        fused_pred, _, _ = probs_to_pred_and_margin(base_fused_prob)
                        fusion_candidates.append(
                            {
                                "fusion_weights": tuple(float(value) for value in weights),
                                "tau_template": float(tau_template),
                                "tau_reference": float(tau_reference),
                                "aux_weight": 0.0,
                                "delta_model": 0.0,
                                "rescue_factor": 1.0,
                                "accuracy": accuracy_score(truth, fused_pred),
                                "macro_f1": macro_f1_score(truth, fused_pred),
                                "probabilities": base_fused_prob,
                                "model_probabilities": model_prob,
                                "rescue_applied": np.zeros((len(rows),), dtype=bool),
                            }
                        )
    fusion_candidates.sort(key=lambda row: (float(row["accuracy"]), float(row["macro_f1"])), reverse=True)
    best_fusion = fusion_candidates[0]

    return {
        "truth": truth,
        "groups": groups,
        "hard_weights": hard_weights,
        "hard_case_payload": hard_case_payload,
        "model_variant_metrics": model_variant_metrics,
        "hist_variant_metrics": model_variant_metrics,
        "best_model_variant": best_model_variant,
        "best_hist_variant": best_model_variant,
        "selected_model_candidate": selected_model_candidate,
        "model_prob": np.asarray(best_fusion["model_probabilities"], dtype=np.float64),
        "hist_prob": np.asarray(best_fusion["model_probabilities"], dtype=np.float64),
        "ord_template_prob": ord_template_prob,
        "ord_shape_prob": ord_shape_prob,
        "ordinal_blend_prob": ordinal_blend,
        "early05_prob": early05_prob,
        "p_template_grid": p_template_grid,
        "p_reference_grid": p_reference_grid,
        "template_cost_grid": template_cost_grid,
        "reference_cost_grid": reference_cost_grid,
        "reference_best_distance": reference_best_distance,
        "best_fusion": best_fusion,
        "selected_model_pred": selected_model_pred,
    }


def save_stage_visualization(
    output_path: Path,
    family_name: str,
    leaderboard_rows: list[dict[str, object]],
    truth: list[str],
    pred: list[str],
) -> None:
    names = [str(row["candidate"]) for row in leaderboard_rows]
    accuracy_values = [float(row["accuracy"]) for row in leaderboard_rows]
    f1_values = [float(row["macro_f1"]) for row in leaderboard_rows]
    matrix = confusion_matrix(truth, pred, labels=STAGE_LABELS)

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_f1 = fig.add_subplot(gs[0, 1])
    ax_conf = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    ax_acc.barh(names, accuracy_values, color="#4E79A7")
    ax_acc.set_xlim(0.0, 1.0)
    ax_acc.set_title("Strict OOF Accuracy")

    ax_f1.barh(names, f1_values, color="#59A14F")
    ax_f1.set_xlim(0.0, 1.0)
    ax_f1.set_title("Strict OOF Macro F1")

    ax_conf.imshow(matrix, cmap="Blues")
    ax_conf.set_title("Best Candidate Confusion")
    ax_conf.set_xticks(range(len(STAGE_LABELS)), labels=STAGE_LABELS)
    ax_conf.set_yticks(range(len(STAGE_LABELS)), labels=STAGE_LABELS)
    ax_conf.set_xlabel("Predicted")
    ax_conf.set_ylabel("True")
    max_value = int(matrix.max()) if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            color = "white" if max_value and matrix[row_index, column_index] >= max_value * 0.5 else "black"
            ax_conf.text(column_index, row_index, str(int(matrix[row_index, column_index])), ha="center", va="center", color=color)

    ax_text.axis("off")
    best = leaderboard_rows[0] if leaderboard_rows else {}
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                f"family: {family_name}",
                f"best_candidate: {best.get('candidate', '')}",
                f"best_accuracy: {float(best.get('accuracy', 0.0)):.3f}",
                f"best_macro_f1: {float(best.get('macro_f1', 0.0)):.3f}",
                f"rows: {len(truth)}",
            ]
        ),
        ha="left",
        va="top",
        family="monospace",
        fontsize=12,
    )

    fig.suptitle(f"{family_name.title()} Stage Upgrade Benchmark", fontsize=16)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_risk_coverage_curve(output_path: Path, oof_rows: list[dict[str, object]], family_name: str) -> None:
    scores = np.asarray([float(row.get("oof_fused_conf", 0.0)) for row in oof_rows], dtype=np.float64)
    truth = [str(row.get("stage_label", "")) for row in oof_rows]
    pred = [str(row.get("oof_fused_pred", "")) for row in oof_rows]
    order = np.argsort(scores)[::-1]
    coverage_points: list[float] = []
    risk_points: list[float] = []
    for count in range(1, len(order) + 1):
        accepted = order[:count].tolist()
        accepted_truth = [truth[index] for index in accepted]
        accepted_pred = [pred[index] for index in accepted]
        coverage = float(count / max(1, len(order)))
        accuracy = accuracy_score(accepted_truth, accepted_pred)
        coverage_points.append(coverage)
        risk_points.append(1.0 - accuracy)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(coverage_points, risk_points, color="#2F6DB3", linewidth=2.0)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (1 - accepted accuracy)")
    ax.set_title(f"{family_name.title()} Risk-Coverage")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_calibration_summary(
    truth: list[str],
    probabilities: np.ndarray,
    pred: list[str],
    best_prob: np.ndarray,
    margin: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, object]:
    prob = np.asarray(probabilities, dtype=np.float64)
    label_index = [STAGE_TO_INDEX[label] for label in truth]
    one_hot = np.zeros_like(prob)
    for row_index, stage_index in enumerate(label_index):
        one_hot[row_index, stage_index] = 1.0
    brier = float(np.mean(np.sum((prob - one_hot) ** 2, axis=1))) if len(prob) else 0.0
    ece = 0.0
    bin_rows: list[dict[str, object]] = []
    ece_bins = np.minimum(CALIBRATION_BIN_COUNT - 1, np.floor(np.clip(best_prob, 0.0, 0.999999) * CALIBRATION_BIN_COUNT).astype(int))
    for bin_index in range(CALIBRATION_BIN_COUNT):
        members = np.where(ece_bins == bin_index)[0]
        if len(members) == 0:
            continue
        bin_truth = [truth[index] for index in members.tolist()]
        bin_pred = [pred[index] for index in members.tolist()]
        bin_acc = accuracy_score(bin_truth, bin_pred)
        bin_conf = float(np.mean(best_prob[members]))
        bin_frac = float(len(members) / max(1, len(truth)))
        ece += abs(bin_acc - bin_conf) * bin_frac
        bin_rows.append(
            {
                "bin_index": int(bin_index),
                "bin_start": float(bin_index / CALIBRATION_BIN_COUNT),
                "bin_end": float((bin_index + 1) / CALIBRATION_BIN_COUNT),
                "n": int(len(members)),
                "accuracy": float(bin_acc),
                "mean_confidence": float(bin_conf),
            }
        )
    selective = accepted_metrics(truth, pred, best_prob, margin, thresholds)
    return {
        "ece": float(ece),
        "brier_score": brier,
        "selective_accuracy": float(selective["precision"]),
        "accepted_only_precision": float(selective["precision"]),
        "accepted_only_coverage": float(selective["coverage"]),
        "bins": bin_rows,
        "ece_bin_assignments": ece_bins.astype(int).tolist(),
    }


def summarize_error_stratification(
    oof_rows: list[dict[str, object]],
    thresholds: dict[str, float],
) -> list[dict[str, object]]:
    summary: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"n": 0.0, "correct": 0.0})
    for row in oof_rows:
        conf = float(row.get("oof_fused_conf", 0.0))
        margin = float(row.get("oof_fused_margin", 0.0))
        accepted = conf >= float(thresholds.get("prob_min", 0.0)) and margin >= float(thresholds.get("margin_min", 0.0))
        groups = {
            "stage": str(row.get("stage_label", "")),
            "source_group": str(row.get("source_group", "")),
            "strict_group": str(row.get("strict_group", "")),
            "accepted": "accepted" if accepted else "review",
            "stage_bucket": "0.5" if str(row.get("stage_label", "")) == "0.5" else "non_0.5",
        }
        if str(row.get("gene_family", "")).strip():
            groups["gene_family"] = str(row.get("gene_family", ""))
        for dimension, value in groups.items():
            key = (dimension, value)
            summary[key]["n"] += 1.0
            summary[key]["correct"] += 1.0 if str(row.get("oof_correct", "")) == "True" else 0.0
    rows: list[dict[str, object]] = []
    for (dimension, value), payload in sorted(summary.items()):
        n = float(payload["n"])
        correct = float(payload["correct"])
        rows.append(
            {
                "dimension": dimension,
                "value": value,
                "n": int(n),
                "accuracy": float(correct / max(1.0, n)),
                "error_rate": float(1.0 - (correct / max(1.0, n))),
            }
        )
    return rows


def build_ablation_rows(
    candidate_runs: list[dict[str, object]],
    best_components: dict[str, object],
    best_fused_run: dict[str, object],
    truth: list[str],
) -> list[dict[str, object]]:
    def choose_best(predicate: Callable[[dict[str, object]], bool]) -> dict[str, object] | None:
        matches = [row for row in candidate_runs if predicate(row)]
        if not matches:
            return None
        matches.sort(key=lambda row: (float(row["accuracy"]), float(row["macro_f1"])), reverse=True)
        return matches[0]

    def prob_metrics(name: str, prob: np.ndarray, detail: str) -> dict[str, object]:
        pred, _, _ = probs_to_pred_and_margin(prob)
        return {
            "ablation_name": name,
            "candidate": detail,
            "accuracy": accuracy_score(truth, pred),
            "macro_f1": macro_f1_score(truth, pred),
        }

    rows: list[dict[str, object]] = []
    category_map = {
        "shape only": lambda row: str(row.get("candidate", "")).endswith("shape_only"),
        "mask only": lambda row: str(row.get("candidate", "")).endswith("mask32_only"),
        "shape + mask": lambda row: str(row.get("candidate", "")) in {"logistic_shape_mask32", "hist_shape_mask32"},
        "shape + template": lambda row: str(row.get("candidate", "")) in {"logistic_shape_template", "hist_shape_template"},
        "shape + template + mask": lambda row: str(row.get("candidate", "")) in {"logistic_shape_template_mask32", "hist_shape_template_mask32"},
    }
    for ablation_name, predicate in category_map.items():
        selected = choose_best(predicate)
        if selected is None:
            continue
        rows.append(
            {
                "ablation_name": ablation_name,
                "candidate": str(selected["candidate"]),
                "accuracy": float(selected["accuracy"]),
                "macro_f1": float(selected["macro_f1"]),
            }
        )
    reference_best: dict[str, object] | None = None
    template_best: dict[str, object] | None = None
    for key, prob in dict(best_components["p_reference_grid"]).items():
        metrics = prob_metrics("reference bank only", np.asarray(prob, dtype=np.float64), f"reference_tau_t{key[0]}_r{key[1]}")
        if reference_best is None or (float(metrics["accuracy"]), float(metrics["macro_f1"])) > (float(reference_best["accuracy"]), float(reference_best["macro_f1"])):
            reference_best = metrics
    for key, prob in dict(best_components["p_template_grid"]).items():
        metrics = prob_metrics("template feature only", np.asarray(prob, dtype=np.float64), f"template_tau_t{key[0]}_r{key[1]}")
        if template_best is None or (float(metrics["accuracy"]), float(metrics["macro_f1"])) > (float(template_best["accuracy"]), float(template_best["macro_f1"])):
            template_best = metrics
    if reference_best is not None:
        rows.append(reference_best)
    if template_best is not None:
        rows.append(template_best)
    single_stage = choose_best(lambda row: str(row.get("candidate", "")).startswith("logistic_") or str(row.get("candidate", "")).startswith("hist_"))
    if single_stage is not None:
        rows.append(
            {
                "ablation_name": "single-stage direct prediction",
                "candidate": str(single_stage["candidate"]),
                "accuracy": float(single_stage["accuracy"]),
                "macro_f1": float(single_stage["macro_f1"]),
            }
        )
    rows.append(
        {
            "ablation_name": "two-stage pipeline",
            "candidate": str(best_fused_run["candidate"]),
            "accuracy": float(best_fused_run["accuracy"]),
            "macro_f1": float(best_fused_run["macro_f1"]),
        }
    )
    return rows


def build_stage_bundle_from_rows(
    feature_rows: list[dict[str, object]],
    family_name: str,
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    stage_feature_summary_fn: Callable[[dict[str, object]], list[dict[str, object]]],
    shape_feature_names: list[str],
    template_feature_names: list[str],
    mask32_feature_names: list[str],
    mask64_feature_names: list[str],
    logistic_c: float,
    random_state: int,
    best_model_candidate_name: str,
    fusion_weights: tuple[float, float, float, float],
    tau_template: float,
    tau_reference: float,
    research_thresholds: dict[str, float],
    deployment_thresholds: dict[str, float],
    profile_settings: dict[str, dict[str, float]],
    strict_protocol_summary: dict[str, object],
    best_candidate_name: str,
    best_single_candidate_name: str,
    early05_aux_weight: float,
    rescue_delta_model: float,
    rescue_factor: float,
) -> dict[str, object]:
    model_variants = model_variant_specs(shape_feature_names, template_feature_names, mask32_feature_names)
    hard_payload = compute_hard_case_payload(
        feature_rows,
        family_name=family_name,
        baseline_feature_names=shape_feature_names + template_feature_names,
        row_matrix_fn=row_matrix_fn,
        random_state=random_state + 1,
    )
    train_weights = np.asarray(hard_payload["weights"], dtype=np.float64)
    selected_feature_names = list(model_variants[best_model_candidate_name])
    model_component = train_multiclass_variant(
        best_model_candidate_name,
        feature_rows,
        feature_names=selected_feature_names,
        row_matrix_fn=row_matrix_fn,
        random_state=random_state + 10,
        logistic_c=logistic_c,
        sample_weight=train_weights,
    )
    ordinal_template_bundle = fit_ordinal_models(
        feature_rows,
        feature_names=template_feature_names + mask64_feature_names,
        row_matrix_fn=row_matrix_fn,
        C=logistic_c,
        random_state=random_state + 20,
        sample_weight=train_weights,
    )
    ordinal_shape_bundle = fit_ordinal_models(
        feature_rows,
        feature_names=shape_feature_names + template_feature_names + mask32_feature_names,
        row_matrix_fn=row_matrix_fn,
        C=logistic_c,
        random_state=random_state + 30,
        sample_weight=train_weights,
    )
    early05_component = fit_binary_logistic(
        feature_rows,
        feature_names=shape_feature_names + template_feature_names + mask32_feature_names,
        row_matrix_fn=row_matrix_fn,
        positive_fn=lambda row: str(row.get("stage_label", "")) == "0.5",
        C=logistic_c,
        random_state=random_state + 40,
        sample_weight=train_weights,
    )
    reference_bank = compute_reference_bank(
        feature_rows,
        template_feature_names=template_feature_names,
        reference_feature_names=shape_feature_names + template_feature_names,
        mask_feature_names=mask32_feature_names,
        row_matrix_fn=row_matrix_fn,
    )
    reference_best_distance: list[float] = []
    for row in feature_rows:
        payload = reference_bank_posteriors(row, reference_bank, row_matrix_fn, tau_template=tau_template, tau_reference=tau_reference)
        reference_best_distance.append(float(payload["best_reference_distance"]))
    model_family = "logistic" if best_model_candidate_name.startswith("logistic_") else "hist"
    return {
        "bundle_kind": "stage_fusion_v1",
        "family_name": family_name,
        "stage_labels": list(STAGE_LABELS),
        "feature_names": selected_feature_names,
        "model_feature_names": selected_feature_names,
        "ordinal_template_feature_names": template_feature_names + mask64_feature_names,
        "ordinal_shape_feature_names": shape_feature_names + template_feature_names + mask32_feature_names,
        "template_feature_names": template_feature_names,
        "reference_feature_names": shape_feature_names + template_feature_names,
        "reference_mask_feature_names": mask32_feature_names,
        "best_candidate_name": best_candidate_name,
        "best_single_candidate_name": best_single_candidate_name,
        "best_p_model_candidate_name": best_model_candidate_name,
        "best_p_model_family": model_family,
        "best_p_model_feature_set": feature_set_label(best_model_candidate_name),
        "fusion_weights": tuple(float(value) for value in fusion_weights),
        "tau_template": float(tau_template),
        "tau_reference": float(tau_reference),
        "ordinal_enabled": True,
        "hard_weight_config": dict(hard_weight_config_for_family(family_name)),
        "hard_weight_config_forewing": dict(FOREWING_HARD_WEIGHT_CONFIG) if family_name == "forewing" else {},
        "strict_protocol_summary": dict(strict_protocol_summary),
        "research_profile_name": "research_strict",
        "default_stage_profile": "deployment_balanced",
        "auto_thresholds_research": dict(research_thresholds),
        "auto_thresholds_deployment": dict(deployment_thresholds),
        "profile_settings": dict(profile_settings),
        "reference_bank_metadata": {
            "rows": len(feature_rows),
            "counts_by_stage": dict(Counter(str(row.get("stage_label", "")) for row in feature_rows)),
            "template_feature_count": len(template_feature_names),
            "reference_feature_count": len(shape_feature_names + template_feature_names),
            "mask_feature_count": len(mask32_feature_names),
            "best_model_variant": best_model_candidate_name,
        },
        "model_component": model_component,
        "hist_component": model_component,
        "ordinal_template_component": {
            "bundle_kind": "ordinal_logistic",
            **ordinal_template_bundle,
        },
        "ordinal_shape_component": {
            "bundle_kind": "ordinal_logistic",
            **ordinal_shape_bundle,
        },
        "early05_aux_enabled": bool(family_name == "forewing" and early05_component is not None and float(early05_aux_weight) > 0.0),
        "early05_aux_weight": float(early05_aux_weight),
        "early05_aux_feature_set": "shape+template+mask",
        "early05_aux_feature_names": list(shape_feature_names + template_feature_names + mask32_feature_names),
        "early05_aux_component": early05_component,
        "forewing_early05_rescue_enabled": bool(family_name == "forewing" and float(rescue_factor) > 1.0),
        "forewing_early05_rescue_delta_model": float(rescue_delta_model),
        "forewing_early05_rescue_factor": float(rescue_factor),
        "reference_bank": reference_bank,
        "full_training_reference_best_distance_quantiles": {
            "q90": float(np.quantile(reference_best_distance, 0.90)) if reference_best_distance else 0.0,
            "q97": float(np.quantile(reference_best_distance, 0.97)) if reference_best_distance else 0.0,
        },
        "top_feature_names": [row["name"] for row in stage_feature_summary_fn(feature_rows[0])] if feature_rows else [],
    }


def run_group_holdout_eval(
    feature_rows: list[dict[str, object]],
    final_bundle: dict[str, object],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    stage_feature_summary_fn: Callable[[dict[str, object]], list[dict[str, object]]],
    shape_feature_names: list[str],
    template_feature_names: list[str],
    mask32_feature_names: list[str],
    mask64_feature_names: list[str],
    logistic_c: float,
    random_state: int,
    group_field: str,
    output_field: str,
    min_group_size: int = 1,
) -> list[dict[str, object]]:
    groups = sorted(
        {
            str(row.get(group_field, "")).strip()
            for row in feature_rows
            if str(row.get(group_field, "")).strip()
        }
    )
    output_rows: list[dict[str, object]] = []
    for fold_index, held_out_group in enumerate(groups):
        train_rows = [row for row in feature_rows if str(row.get(group_field, "")) != held_out_group]
        val_rows = [row for row in feature_rows if str(row.get(group_field, "")) == held_out_group]
        if len(val_rows) < max(1, int(min_group_size)):
            continue
        if not train_rows or not val_rows:
            continue
        if len({str(row.get("stage_label", "")) for row in train_rows}) < len(STAGE_LABELS):
            continue
        holdout_bundle = build_stage_bundle_from_rows(
            train_rows,
            family_name=str(final_bundle.get("family_name", "")),
            row_matrix_fn=row_matrix_fn,
            stage_feature_summary_fn=stage_feature_summary_fn,
            shape_feature_names=shape_feature_names,
            template_feature_names=template_feature_names,
            mask32_feature_names=mask32_feature_names,
            mask64_feature_names=mask64_feature_names,
            logistic_c=logistic_c,
            random_state=random_state + 500 + fold_index * 50,
            best_model_candidate_name=str(final_bundle.get("best_p_model_candidate_name", final_bundle.get("best_single_candidate_name", "hist_shape_template"))),
            fusion_weights=tuple(final_bundle.get("fusion_weights", FUSION_WEIGHT_GRID[0])),
            tau_template=float(final_bundle.get("tau_template", 1.0)),
            tau_reference=float(final_bundle.get("tau_reference", 1.0)),
            research_thresholds=dict(final_bundle.get("auto_thresholds_research", {})),
            deployment_thresholds=dict(final_bundle.get("auto_thresholds_deployment", {})),
            profile_settings=dict(final_bundle.get("profile_settings", {})),
            strict_protocol_summary=dict(final_bundle.get("strict_protocol_summary", {})),
            best_candidate_name=str(final_bundle.get("best_candidate_name", "")),
            best_single_candidate_name=str(final_bundle.get("best_single_candidate_name", "")),
            early05_aux_weight=float(final_bundle.get("early05_aux_weight", 0.0)),
            rescue_delta_model=float(final_bundle.get("forewing_early05_rescue_delta_model", 0.0)),
            rescue_factor=float(final_bundle.get("forewing_early05_rescue_factor", 1.0)),
        )
        for row in val_rows:
            payload = apply_stage_fusion_bundle(holdout_bundle, row, row_matrix_fn, stage_profile="research_strict")
            output_rows.append(
                {
                    output_field: held_out_group,
                    "id": row.get("id", ""),
                    "original_id": row.get("original_id", ""),
                    "source_group": row.get("source_group", ""),
                    "modality": row.get("modality", ""),
                    "gene_family": row.get("gene_family", ""),
                    "stage_label": row.get("stage_label", ""),
                    "predicted_stage": payload["fused_stage_pred"],
                    "predicted_conf": payload["fused_stage_conf"],
                    "predicted_margin": payload["fused_stage_margin"],
                    "correct": str(str(row.get("stage_label", "")) == str(payload["fused_stage_pred"])),
                    "best_p_model_candidate_name": payload.get("best_p_model_candidate_name", holdout_bundle.get("best_p_model_candidate_name", "")),
                }
            )
    return output_rows


def run_source_group_holdout_eval(
    feature_rows: list[dict[str, object]],
    final_bundle: dict[str, object],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    stage_feature_summary_fn: Callable[[dict[str, object]], list[dict[str, object]]],
    shape_feature_names: list[str],
    template_feature_names: list[str],
    mask32_feature_names: list[str],
    mask64_feature_names: list[str],
    logistic_c: float,
    random_state: int,
) -> list[dict[str, object]]:
    return run_group_holdout_eval(
        feature_rows=feature_rows,
        final_bundle=final_bundle,
        row_matrix_fn=row_matrix_fn,
        stage_feature_summary_fn=stage_feature_summary_fn,
        shape_feature_names=shape_feature_names,
        template_feature_names=template_feature_names,
        mask32_feature_names=mask32_feature_names,
        mask64_feature_names=mask64_feature_names,
        logistic_c=logistic_c,
        random_state=random_state,
        group_field="source_group",
        output_field="source_group_holdout_fold",
        min_group_size=1,
    )


def run_human_in_loop_simulation(
    oof_rows: list[dict[str, object]],
    profile_settings: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    truth = [str(row.get("stage_label", "")) for row in oof_rows]
    pred = [str(row.get("oof_fused_pred", "")) for row in oof_rows]
    conf = np.asarray([float(row.get("oof_fused_conf", 0.0)) for row in oof_rows], dtype=np.float64)
    margin = np.asarray([float(row.get("oof_fused_margin", 0.0)) for row in oof_rows], dtype=np.float64)
    reference_distance = np.asarray([float(row.get("oof_reference_best_distance", 0.0)) for row in oof_rows], dtype=np.float64)
    ordinal_pred = [str(row.get("oof_ordinal_shape_pred", row.get("oof_ordinal_template_pred", ""))) for row in oof_rows]
    results: list[dict[str, object]] = []

    def accepted_indices(thresholds: dict[str, float], allow_reference_override: bool) -> list[int]:
        accepted: list[int] = []
        for index, row in enumerate(oof_rows):
            ordinal_conflict = abs(STAGE_TO_INDEX.get(pred[index], 0) - STAGE_TO_INDEX.get(ordinal_pred[index], 0)) >= 2
            distance_ok = reference_distance[index] <= float(thresholds.get("reference_nn_max", 1e9))
            far_override = bool(
                allow_reference_override
                and not distance_ok
                and conf[index] >= float(thresholds.get("far_override_conf_min", 1.1))
                and margin[index] >= float(thresholds.get("far_override_margin_min", 1.1))
            )
            if (
                conf[index] >= float(thresholds.get("prob_min", 0.0))
                and margin[index] >= float(thresholds.get("margin_min", 0.0))
                and not ordinal_conflict
                and (distance_ok or far_override)
            ):
                accepted.append(index)
        return accepted

    strategies = {
        "all_auto": {"prob_min": 0.0, "margin_min": 0.0, "reference_nn_max": 1e9, "far_override_conf_min": 0.0, "far_override_margin_min": 0.0},
        "research_strict": dict(profile_settings.get("research_strict", {})),
        "deployment_balanced": dict(profile_settings.get("deployment_balanced", {})),
    }
    high_precision = dict(profile_settings.get("deployment_balanced", {}))
    high_precision["prob_min"] = max(float(high_precision.get("prob_min", 0.0)), 0.80)
    high_precision["margin_min"] = max(float(high_precision.get("margin_min", 0.0)), 0.15)
    strategies["high_precision_auto"] = high_precision

    all_auto_accuracy = accuracy_score(truth, pred)
    for name, thresholds in strategies.items():
        accepted = accepted_indices(thresholds, allow_reference_override=bool(thresholds.get("allow_far_override", False)))
        review = [index for index in range(len(oof_rows)) if index not in set(accepted)]
        accepted_truth = [truth[index] for index in accepted]
        accepted_pred = [pred[index] for index in accepted]
        auto_precision = accuracy_score(accepted_truth, accepted_pred) if accepted else 0.0
        system_accuracy_if_reviewed = float((sum(1 for index in accepted if truth[index] == pred[index]) + len(review)) / max(1, len(oof_rows)))
        results.append(
            {
                "strategy": name,
                "total_n": int(len(oof_rows)),
                "auto_n": int(len(accepted)),
                "review_n": int(len(review)),
                "auto_coverage": float(len(accepted) / max(1, len(oof_rows))),
                "auto_precision": float(auto_precision),
                "auto_accuracy": float(auto_precision),
                "system_accuracy_if_review_correct": system_accuracy_if_reviewed,
                "manual_load_fraction": float(len(review) / max(1, len(oof_rows))),
                "manual_reduction_vs_all_manual": float(len(accepted) / max(1, len(oof_rows))),
                "error_reduction_vs_all_auto": float(system_accuracy_if_reviewed - all_auto_accuracy),
            }
        )
    return results


def run_robustness_eval(
    feature_rows: list[dict[str, object]],
    final_bundle: dict[str, object],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    build_feature_row_fn: Callable[..., dict[str, object]],
    mask_sizes: list[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    perturbations = [
        ("mask_erode_1", "mask_only"),
        ("mask_dilate_1", "mask_only"),
        ("rotate_5deg", "mask_only"),
        ("mirror_horizontal", "mask_only"),
        ("local_dropout_10pct", "mask_only"),
        ("brightness_0_85_extract", "image_extract"),
        ("brightness_1_15_extract", "image_extract"),
        ("contrast_0_85_extract", "image_extract"),
        ("contrast_1_15_extract", "image_extract"),
    ]
    prediction_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for base_row in feature_rows:
        original_payload = apply_stage_fusion_bundle(final_bundle, base_row, row_matrix_fn, stage_profile="research_strict")
        gray = load_gray_image(str(base_row.get("image_path", "")))
        mask_path = str(base_row.get("mask_path", "")).strip()
        if not mask_path:
            continue
        mask = load_binary_mask(mask_path)
        for perturbation_name, perturbation_family in perturbations:
            perturbed_gray = np.asarray(gray, dtype=np.uint8)
            perturbed_mask = np.asarray(mask, dtype=bool)
            mask_source = f"robust::{perturbation_name}"
            if perturbation_name == "mask_erode_1":
                perturbed_mask = _binary_erode(perturbed_mask, iterations=1)
            elif perturbation_name == "mask_dilate_1":
                perturbed_mask = _binary_dilate(perturbed_mask, iterations=1)
            elif perturbation_name == "rotate_5deg":
                perturbed_mask = _rotate_mask(perturbed_mask, 5.0)
            elif perturbation_name == "mirror_horizontal":
                perturbed_mask = np.fliplr(perturbed_mask)
            elif perturbation_name == "local_dropout_10pct":
                perturbed_mask = _dropout_mask(perturbed_mask, seed_text=f"{base_row.get('id', '')}:{perturbation_name}", drop_fraction=0.10)
            elif perturbation_name == "brightness_0_85_extract":
                perturbed_gray = _perturb_gray(gray, brightness=0.85, contrast=1.0)
                perturbed_mask = np.asarray(extract_best_wing_mask(perturbed_gray)["mask"], dtype=bool)
                mask_source = "robust::brightness_extract"
            elif perturbation_name == "brightness_1_15_extract":
                perturbed_gray = _perturb_gray(gray, brightness=1.15, contrast=1.0)
                perturbed_mask = np.asarray(extract_best_wing_mask(perturbed_gray)["mask"], dtype=bool)
                mask_source = "robust::brightness_extract"
            elif perturbation_name == "contrast_0_85_extract":
                perturbed_gray = _perturb_gray(gray, brightness=1.0, contrast=0.85)
                perturbed_mask = np.asarray(extract_best_wing_mask(perturbed_gray)["mask"], dtype=bool)
                mask_source = "robust::contrast_extract"
            elif perturbation_name == "contrast_1_15_extract":
                perturbed_gray = _perturb_gray(gray, brightness=1.0, contrast=1.15)
                perturbed_mask = np.asarray(extract_best_wing_mask(perturbed_gray)["mask"], dtype=bool)
                mask_source = "robust::contrast_extract"
            perturbed_feature_row = build_feature_row_fn(
                item_id=str(base_row.get("id", "")),
                original_id=str(base_row.get("original_id", "")),
                image_path=str(base_row.get("image_path", "")),
                mask_path=str(base_row.get("mask_path", "")),
                source_group=str(base_row.get("source_group", "")),
                stage_label=str(base_row.get("stage_label", "")),
                modality=str(base_row.get("modality", "")),
                gene_family=str(base_row.get("gene_family", "")),
                sample_family=str(base_row.get("sample_family", "")),
                strict_group=str(base_row.get("strict_group", "")),
                split_group=str(base_row.get("split_group", "")),
                status=str(base_row.get("status", "")),
                manual_review_status=str(base_row.get("manual_review_status", "")),
                alignment_source=str(base_row.get("alignment_source", "")),
                mask_sizes=mask_sizes,
                include_mask_vectors=True,
                gray_override=perturbed_gray,
                mask_override=perturbed_mask,
                mask_source_override=mask_source,
            )
            perturbed_payload = apply_stage_fusion_bundle(final_bundle, perturbed_feature_row, row_matrix_fn, stage_profile="research_strict")
            prediction_rows.append(
                {
                    "id": base_row.get("id", ""),
                    "source_group": base_row.get("source_group", ""),
                    "modality": base_row.get("modality", ""),
                    "stage_label": base_row.get("stage_label", ""),
                    "perturbation": perturbation_name,
                    "perturbation_family": perturbation_family,
                    "base_pred": original_payload["fused_stage_pred"],
                    "perturbed_pred": perturbed_payload["fused_stage_pred"],
                    "prediction_changed": str(original_payload["fused_stage_pred"] != perturbed_payload["fused_stage_pred"]),
                    "base_conf": float(original_payload["fused_stage_conf"]),
                    "perturbed_conf": float(perturbed_payload["fused_stage_conf"]),
                    "confidence_delta": float(perturbed_payload["fused_stage_conf"] - original_payload["fused_stage_conf"]),
                    "base_maturity": float(original_payload["maturity_score"]),
                    "perturbed_maturity": float(perturbed_payload["maturity_score"]),
                    "maturity_shift": float(abs(perturbed_payload["maturity_score"] - original_payload["maturity_score"])),
                    "base_decision": original_payload["decision"],
                    "perturbed_decision": perturbed_payload["decision"],
                    "review_changed": str(original_payload["decision"] != perturbed_payload["decision"]),
                    "base_correct": str(original_payload["fused_stage_pred"] == str(base_row.get("stage_label", ""))),
                    "perturbed_correct": str(perturbed_payload["fused_stage_pred"] == str(base_row.get("stage_label", ""))),
                }
            )
    by_perturbation: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        by_perturbation[str(row.get("perturbation", ""))].append(row)
    for perturbation_name, rows in sorted(by_perturbation.items()):
        prediction_consistency = float(sum(1 for row in rows if str(row.get("prediction_changed", "")) == "False") / max(1, len(rows)))
        review_rate = float(sum(1 for row in rows if str(row.get("perturbed_decision", "")) == "uncertain_review") / max(1, len(rows)))
        perturbed_accuracy = float(sum(1 for row in rows if str(row.get("perturbed_correct", "")) == "True") / max(1, len(rows)))
        summary_rows.append(
            {
                "perturbation": perturbation_name,
                "perturbation_family": str(rows[0].get("perturbation_family", "")),
                "n": int(len(rows)),
                "prediction_consistency": prediction_consistency,
                "perturbed_accuracy": perturbed_accuracy,
                "mean_confidence_delta": float(np.mean([float(row.get("confidence_delta", 0.0)) for row in rows])),
                "mean_maturity_shift": float(np.mean([float(row.get("maturity_shift", 0.0)) for row in rows])),
                "review_rate": review_rate,
            }
        )
    return prediction_rows, summary_rows


def train_stage_family_upgrade(
    family_name: str,
    manifest_csv: str | Path,
    output_dir: str | Path,
    build_feature_row_fn: Callable[..., dict[str, object]],
    numeric_feature_names_fn: Callable[[list[dict[str, object]], list[str] | None], list[str]],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    stage_feature_summary_fn: Callable[[dict[str, object]], list[dict[str, object]]],
    mask_sizes: list[int],
    logistic_c_grid: list[float],
    random_state: int,
    target_auto_precision: float,
    research_auto_precision: float = 0.95,
    allowed_modalities: list[str] | None = None,
    run_robustness: bool = True,
) -> dict[str, object]:
    manifest_csv = Path(manifest_csv).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [
        row
        for row in read_csv_rows(manifest_csv)
        if str(row.get("stage_label", "")).strip() in STAGE_LABELS
        and str(row.get("image_path", "")).strip()
        and str(row.get("mask_path", "")).strip()
    ]
    if allowed_modalities:
        allowed = {str(value).strip().lower() for value in allowed_modalities if str(value).strip()}
        manifest_rows = [row for row in manifest_rows if str(row.get("modality", "")).strip().lower() in allowed]

    feature_rows = [
        build_feature_row_fn(
            item_id=str(row.get("id", "")).strip() or Path(str(row["image_path"])).stem,
            original_id=str(row.get("original_id", "")).strip() or canonical_original_id("", row.get("id", ""), str(row.get("image_path", ""))),
            image_path=str(row["image_path"]),
            mask_path=str(row["mask_path"]),
            source_group=str(row.get("source_group", "")),
            stage_label=str(row.get("stage_label", "")).strip(),
            modality=str(row.get("modality", "")).strip(),
            gene_family=str(row.get("gene_family", "")).strip(),
            sample_family=str(row.get("sample_family", family_name)).strip() or family_name,
            strict_group=str(row.get("strict_group", "")).strip(),
            split_group=str(row.get("split_group", "")).strip(),
            status=str(row.get("status", "")).strip(),
            manual_review_status=str(row.get("manual_review_status", "")).strip(),
            alignment_source=str(row.get("alignment_source", "")).strip(),
            mask_sizes=mask_sizes,
            include_mask_vectors=True,
        )
        for row in manifest_rows
    ]

    all_numeric_feature_names = list(numeric_feature_names_fn(feature_rows, None))
    template_feature_names = [name for name in all_numeric_feature_names if name.startswith("stage_") or name in {"best_template_score", "best_template_margin"}]
    shape_feature_names = [name for name in all_numeric_feature_names if name not in template_feature_names and not name.startswith("mask")]
    mask32_feature_names = numeric_feature_names_fn(feature_rows, ["mask32_px_"])
    mask64_feature_names = numeric_feature_names_fn(feature_rows, ["mask64_px_"])

    candidate_runs: list[dict[str, object]] = []
    truth = [str(row.get("stage_label", "")) for row in feature_rows]
    for c_value in logistic_c_grid or [1.0]:
        component_oof = evaluate_components_oof(
            feature_rows,
            shape_feature_names=shape_feature_names,
            template_feature_names=template_feature_names,
            mask32_feature_names=mask32_feature_names,
            mask64_feature_names=mask64_feature_names,
            row_matrix_fn=row_matrix_fn,
            random_state=random_state + int(round(float(c_value) * 100)),
            family_name=family_name,
            logistic_c=float(c_value),
        )
        for metric in list(component_oof["model_variant_metrics"]):
            candidate_runs.append(
                {
                    "candidate": str(metric["candidate"]),
                    "C": float(c_value),
                    "accuracy": float(metric["accuracy"]),
                    "macro_f1": float(metric["macro_f1"]),
                    "predicted_labels": list(metric["predicted_labels"]),
                    "best_probability": np.asarray(metric["best_probability"], dtype=np.float64),
                    "margin": np.asarray(metric["margin"], dtype=np.float64),
                    "probabilities": np.asarray(metric["probabilities"], dtype=np.float64),
                    "components": component_oof,
                    "feature_set": feature_set_label(str(metric["candidate"])),
                    "model_family": "logistic" if str(metric["candidate"]).startswith("logistic_") else "hist",
                }
            )
        ord_template_pred, ord_template_best_prob, ord_template_margin = probs_to_pred_and_margin(component_oof["ord_template_prob"])
        ord_shape_pred, ord_shape_best_prob, ord_shape_margin = probs_to_pred_and_margin(component_oof["ord_shape_prob"])
        best_fusion = dict(component_oof["best_fusion"])
        fused_prob = np.asarray(best_fusion["probabilities"], dtype=np.float64)
        fused_pred, fused_best_prob, fused_margin = probs_to_pred_and_margin(fused_prob)
        fusion_suffix = (
            f"w{best_fusion['fusion_weights'][0]:.2f}_{best_fusion['fusion_weights'][1]:.2f}_"
            f"{best_fusion['fusion_weights'][2]:.2f}_{best_fusion['fusion_weights'][3]:.2f}_"
            f"tt{float(best_fusion['tau_template']):.2f}_tr{float(best_fusion['tau_reference']):.2f}_"
            f"aux{float(best_fusion.get('aux_weight', 0.0)):.2f}_d{float(best_fusion.get('delta_model', 0.0)):.2f}_"
            f"r{float(best_fusion.get('rescue_factor', 1.0)):.2f}"
        )
        candidate_runs.extend(
            [
                {
                    "candidate": "ordinal_template_mask64",
                    "C": float(c_value),
                    "accuracy": accuracy_score(truth, ord_template_pred),
                    "macro_f1": macro_f1_score(truth, ord_template_pred),
                    "predicted_labels": ord_template_pred,
                    "best_probability": ord_template_best_prob,
                    "margin": ord_template_margin,
                    "probabilities": np.asarray(component_oof["ord_template_prob"], dtype=np.float64),
                    "components": component_oof,
                },
                {
                    "candidate": "ordinal_shape_template_mask32",
                    "C": float(c_value),
                    "accuracy": accuracy_score(truth, ord_shape_pred),
                    "macro_f1": macro_f1_score(truth, ord_shape_pred),
                    "predicted_labels": ord_shape_pred,
                    "best_probability": ord_shape_best_prob,
                    "margin": ord_shape_margin,
                    "probabilities": np.asarray(component_oof["ord_shape_prob"], dtype=np.float64),
                    "components": component_oof,
                },
                {
                    "candidate": f"fused_{fusion_suffix}",
                    "family_candidate": "fused",
                    "C": float(c_value),
                    "fusion_weights": best_fusion["fusion_weights"],
                    "tau_template": best_fusion["tau_template"],
                    "tau_reference": best_fusion["tau_reference"],
                    "aux_weight": best_fusion.get("aux_weight", 0.0),
                    "delta_model": best_fusion.get("delta_model", 0.0),
                    "rescue_factor": best_fusion.get("rescue_factor", 1.0),
                    "accuracy": accuracy_score(truth, fused_pred),
                    "macro_f1": macro_f1_score(truth, fused_pred),
                    "predicted_labels": fused_pred,
                    "best_probability": fused_best_prob,
                    "margin": fused_margin,
                    "probabilities": fused_prob,
                    "components": component_oof,
                },
            ]
        )

    candidate_runs.sort(key=lambda row: (float(row["accuracy"]), float(row["macro_f1"])), reverse=True)
    best_overall = candidate_runs[0]
    fused_runs = [row for row in candidate_runs if str(row.get("family_candidate", "")) == "fused"]
    best_run = fused_runs[0] if fused_runs else best_overall
    best_components = dict(best_run["components"])
    best_fusion = dict(best_components["best_fusion"])
    fused_prob = np.asarray(best_run["probabilities"], dtype=np.float64)
    fused_pred = list(best_run["predicted_labels"])
    fused_best_prob = np.asarray(best_run["best_probability"], dtype=np.float64)
    fused_margin = np.asarray(best_run["margin"], dtype=np.float64)

    research_thresholds = choose_auto_thresholds(truth, fused_pred, fused_best_prob, fused_margin, target_precision=research_auto_precision)
    deployment_thresholds = choose_auto_thresholds(truth, fused_pred, fused_best_prob, fused_margin, target_precision=target_auto_precision)
    correct_reference_distances = [
        float(best_components["reference_best_distance"][index])
        for index in range(len(truth))
        if fused_pred[index] == truth[index]
    ]
    research_ref_max = float(np.quantile(correct_reference_distances, 0.90)) if correct_reference_distances else 1e6
    deployment_ref_max = float(np.quantile(correct_reference_distances, 0.97)) if correct_reference_distances else 1e6
    strict_protocol_summary = {
        "group_field": "strict_group preferred, then split_group",
        "outer_protocol": "group-held-out OOF",
        "reference_bank_used": True,
        "template_bank_rebuilt_per_fold": True,
        "hard_case_reweighting_enabled": True,
        "source_group_holdout_enabled": True,
    }
    profile_settings = {
        "research_strict": {
            **research_thresholds,
            "reference_nn_max": float(research_ref_max),
            "maturity_offset_max": 0.20,
            "allow_far_override": False,
            "far_override_conf_min": 1.1,
            "far_override_margin_min": 1.1,
        },
        "deployment_balanced": {
            **deployment_thresholds,
            "reference_nn_max": float(deployment_ref_max),
            "maturity_offset_max": 0.20,
            "allow_far_override": True,
            "far_override_conf_min": float(max(deployment_thresholds.get("prob_min", 0.0) + 0.10, 0.80)),
            "far_override_margin_min": float(max(deployment_thresholds.get("margin_min", 0.0) + 0.05, 0.10)),
        },
    }
    best_model_candidate_name = str(best_components["best_model_variant"]["candidate"])
    best_single_candidate_name = max(
        [row for row in candidate_runs if not str(row.get("candidate", "")).startswith("fused_")],
        key=lambda row: (float(row["accuracy"]), float(row["macro_f1"])),
    )["candidate"]
    final_bundle = build_stage_bundle_from_rows(
        feature_rows,
        family_name=family_name,
        row_matrix_fn=row_matrix_fn,
        stage_feature_summary_fn=stage_feature_summary_fn,
        shape_feature_names=shape_feature_names,
        template_feature_names=template_feature_names,
        mask32_feature_names=mask32_feature_names,
        mask64_feature_names=mask64_feature_names,
        logistic_c=float(best_run.get("C", 1.0)),
        random_state=random_state + 700,
        best_model_candidate_name=best_model_candidate_name,
        fusion_weights=tuple(best_fusion["fusion_weights"]),
        tau_template=float(best_fusion["tau_template"]),
        tau_reference=float(best_fusion["tau_reference"]),
        research_thresholds=research_thresholds,
        deployment_thresholds=deployment_thresholds,
        profile_settings=profile_settings,
        strict_protocol_summary=strict_protocol_summary,
        best_candidate_name=str(best_run["candidate"]),
        best_single_candidate_name=str(best_single_candidate_name),
        early05_aux_weight=float(best_fusion.get("aux_weight", 0.0)),
        rescue_delta_model=float(best_fusion.get("delta_model", 0.0)),
        rescue_factor=float(best_fusion.get("rescue_factor", 1.0)),
    )

    p_model_pred, p_model_best_prob, _ = probs_to_pred_and_margin(best_components["model_prob"])
    ord_template_pred, _, _ = probs_to_pred_and_margin(best_components["ord_template_prob"])
    ord_shape_pred, _, _ = probs_to_pred_and_margin(best_components["ord_shape_prob"])
    ordinal_blend = np.asarray(best_components["ordinal_blend_prob"], dtype=np.float64)
    calibration_summary = compute_calibration_summary(truth, fused_prob, fused_pred, fused_best_prob, fused_margin, deployment_thresholds)
    ece_bins = list(calibration_summary["ece_bin_assignments"])
    oof_rows: list[dict[str, object]] = []
    rescue_flags = np.asarray(best_fusion.get("rescue_applied", np.zeros((len(feature_rows),), dtype=bool)), dtype=bool)
    for index, row in enumerate(feature_rows):
        maturity_score = float(np.sum(np.asarray(fused_prob[index], dtype=np.float64) * STAGE_NUMERIC_VALUES))
        oof_rows.append(
            {
                "id": row["id"],
                "original_id": row.get("original_id", ""),
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "source_group": row["source_group"],
                "modality": row["modality"],
                "gene_family": row.get("gene_family", ""),
                "sample_family": row.get("sample_family", family_name),
                "stage_label": row["stage_label"],
                "strict_group": row.get("strict_group", ""),
                "split_group": row.get("split_group", ""),
                "sample_weight": float(best_components["hard_weights"][index]),
                "best_p_model_candidate_name": str(best_components["selected_model_candidate"][index]),
                "p_model_pred": p_model_pred[index],
                "p_model_conf": float(p_model_best_prob[index]),
                "early05_aux_score": float(best_components["early05_prob"][index]),
                "early05_rescue_applied": str(bool(rescue_flags[index])),
                "oof_ordinal_template_pred": ord_template_pred[index],
                "oof_ordinal_template_conf": float(np.max(best_components["ord_template_prob"][index])),
                "oof_ordinal_shape_pred": ord_shape_pred[index],
                "oof_ordinal_shape_conf": float(np.max(best_components["ord_shape_prob"][index])),
                "oof_fused_pred": fused_pred[index],
                "oof_fused_conf": float(fused_best_prob[index]),
                "oof_fused_margin": float(fused_margin[index]),
                "oof_maturity_score": maturity_score,
                "oof_reference_best_distance": float(best_components["reference_best_distance"][index]),
                "oof_correct": str(fused_pred[index] == truth[index]),
                "ece_bin": int(ece_bins[index]),
            }
        )
        for label in STAGE_LABELS:
            oof_rows[-1][f"p_model_prob_{label.replace('.', '_')}"] = float(best_components["model_prob"][index, STAGE_TO_INDEX[label]])
            oof_rows[-1][f"oof_template_prob_{label.replace('.', '_')}"] = float(best_components["p_template_grid"][(best_fusion["tau_template"], best_fusion["tau_reference"])][index, STAGE_TO_INDEX[label]])
            oof_rows[-1][f"oof_reference_prob_{label.replace('.', '_')}"] = float(best_components["p_reference_grid"][(best_fusion["tau_template"], best_fusion["tau_reference"])][index, STAGE_TO_INDEX[label]])
            oof_rows[-1][f"oof_ordinal_prob_{label.replace('.', '_')}"] = float(ordinal_blend[index, STAGE_TO_INDEX[label]])
            oof_rows[-1][f"oof_fused_prob_{label.replace('.', '_')}"] = float(fused_prob[index, STAGE_TO_INDEX[label]])

    leaderboard_rows = [
        {key: value for key, value in row.items() if key not in {"predicted_labels", "best_probability", "margin", "probabilities", "components"}}
        for row in candidate_runs
    ]
    ablation_rows = build_ablation_rows(candidate_runs, best_components, best_run, truth)
    source_group_holdout_rows = run_source_group_holdout_eval(
        feature_rows,
        final_bundle,
        row_matrix_fn=row_matrix_fn,
        stage_feature_summary_fn=stage_feature_summary_fn,
        shape_feature_names=shape_feature_names,
        template_feature_names=template_feature_names,
        mask32_feature_names=mask32_feature_names,
        mask64_feature_names=mask64_feature_names,
        logistic_c=float(best_run.get("C", 1.0)),
        random_state=random_state,
    )
    modality_holdout_rows = run_group_holdout_eval(
        feature_rows,
        final_bundle,
        row_matrix_fn=row_matrix_fn,
        stage_feature_summary_fn=stage_feature_summary_fn,
        shape_feature_names=shape_feature_names,
        template_feature_names=template_feature_names,
        mask32_feature_names=mask32_feature_names,
        mask64_feature_names=mask64_feature_names,
        logistic_c=float(best_run.get("C", 1.0)),
        random_state=random_state + 1000,
        group_field="modality",
        output_field="modality_holdout_fold",
        min_group_size=1,
    )
    gene_family_holdout_rows = run_group_holdout_eval(
        feature_rows,
        final_bundle,
        row_matrix_fn=row_matrix_fn,
        stage_feature_summary_fn=stage_feature_summary_fn,
        shape_feature_names=shape_feature_names,
        template_feature_names=template_feature_names,
        mask32_feature_names=mask32_feature_names,
        mask64_feature_names=mask64_feature_names,
        logistic_c=float(best_run.get("C", 1.0)),
        random_state=random_state + 1500,
        group_field="gene_family",
        output_field="gene_family_holdout_fold",
        min_group_size=3,
    )
    error_stratification_rows = summarize_error_stratification(oof_rows, deployment_thresholds)
    stage0_5_errors = [row for row in oof_rows if str(row.get("stage_label", "")) == "0.5" and str(row.get("oof_correct", "")) != "True"]
    human_in_loop_rows = run_human_in_loop_simulation(oof_rows, profile_settings)
    robustness_prediction_rows: list[dict[str, object]] = []
    robustness_summary_rows: list[dict[str, object]] = []
    if run_robustness:
        robustness_prediction_rows, robustness_summary_rows = run_robustness_eval(
            feature_rows,
            final_bundle,
            row_matrix_fn=row_matrix_fn,
            build_feature_row_fn=build_feature_row_fn,
            mask_sizes=mask_sizes,
        )
    experiment_scope_rows = [
        {
            "experiment": "strict_oof",
            "scope": "all_rows_in_manifest",
            "n_rows": int(len(feature_rows)),
            "notes": "group-held-out strict OOF over the full manifest",
        },
        {
            "experiment": "source_group_holdout",
            "scope": "all_rows_repeated_by_source_group",
            "n_rows": int(len(source_group_holdout_rows)),
            "notes": "each source_group is held out once; output rows are per-sample predictions",
        },
        {
            "experiment": "modality_holdout",
            "scope": "all_rows_repeated_by_modality",
            "n_rows": int(len(modality_holdout_rows)),
            "notes": "each modality is held out once; output rows are per-sample predictions",
        },
        {
            "experiment": "gene_family_holdout",
            "scope": "gene_families_with_n_ge_3",
            "n_rows": int(len(gene_family_holdout_rows)),
            "notes": "leave-one-gene-family-out only for families with at least 3 samples",
        },
        {
            "experiment": "human_in_loop",
            "scope": "all_oof_rows",
            "n_rows": int(len(oof_rows)),
            "notes": "simulation on OOF predictions from the full manifest",
        },
        {
            "experiment": "robustness",
            "scope": "all_rows_x_all_perturbations" if run_robustness else "skipped_in_training_run",
            "n_rows": int(len(robustness_prediction_rows)),
            "notes": "mask-space plus image-to-mask perturbations on every sample" if run_robustness else "robustness deferred to dedicated post-training step",
        },
    ]

    leaderboard_path = output_dir / f"{family_name}_stage_leaderboard.csv"
    oof_path = output_dir / f"{family_name}_stage_oof_predictions.csv"
    source_holdout_path = output_dir / f"{family_name}_stage_source_group_holdout.csv"
    modality_holdout_path = output_dir / f"{family_name}_stage_modality_holdout.csv"
    gene_family_holdout_path = output_dir / f"{family_name}_stage_gene_family_holdout.csv"
    ablation_path = output_dir / f"{family_name}_stage_ablation.csv"
    calibration_path = output_dir / f"{family_name}_stage_calibration.json"
    stratification_path = output_dir / f"{family_name}_stage_error_stratification.csv"
    human_in_loop_path = output_dir / f"{family_name}_stage_human_in_loop_simulation.csv"
    robustness_summary_path = output_dir / f"{family_name}_stage_robustness_summary.csv"
    robustness_predictions_path = output_dir / f"{family_name}_stage_robustness_predictions.csv"
    experiment_scope_path = output_dir / f"{family_name}_stage_validation_scope.csv"
    summary_path = output_dir / "training_summary.json"
    viz_path = output_dir / f"{family_name}_stage_classifier_overview.png"
    bundle_path = output_dir / "stage_model_bundle.pkl"
    risk_path = output_dir / "risk_coverage_curve.png"
    hard_audit_path = output_dir / f"{family_name}_stage_0_5_error_audit.csv"
    best_log_path = output_dir / "best_config_log.md"

    write_csv_rows(leaderboard_path, list(leaderboard_rows[0].keys()), leaderboard_rows)
    write_csv_rows(oof_path, list(oof_rows[0].keys()), oof_rows)
    if source_group_holdout_rows:
        write_csv_rows(source_holdout_path, list(source_group_holdout_rows[0].keys()), source_group_holdout_rows)
    if modality_holdout_rows:
        write_csv_rows(modality_holdout_path, list(modality_holdout_rows[0].keys()), modality_holdout_rows)
    if gene_family_holdout_rows:
        write_csv_rows(gene_family_holdout_path, list(gene_family_holdout_rows[0].keys()), gene_family_holdout_rows)
    if ablation_rows:
        write_csv_rows(ablation_path, list(ablation_rows[0].keys()), ablation_rows)
    if error_stratification_rows:
        write_csv_rows(stratification_path, list(error_stratification_rows[0].keys()), error_stratification_rows)
    if stage0_5_errors:
        write_csv_rows(hard_audit_path, list(stage0_5_errors[0].keys()), stage0_5_errors)
    if human_in_loop_rows:
        write_csv_rows(human_in_loop_path, list(human_in_loop_rows[0].keys()), human_in_loop_rows)
    if robustness_summary_rows:
        write_csv_rows(robustness_summary_path, list(robustness_summary_rows[0].keys()), robustness_summary_rows)
    if robustness_prediction_rows:
        write_csv_rows(robustness_predictions_path, list(robustness_prediction_rows[0].keys()), robustness_prediction_rows)
    if experiment_scope_rows:
        write_csv_rows(experiment_scope_path, list(experiment_scope_rows[0].keys()), experiment_scope_rows)
    save_stage_visualization(viz_path, family_name, leaderboard_rows[: min(12, len(leaderboard_rows))], truth, fused_pred)
    save_risk_coverage_curve(risk_path, oof_rows, family_name)
    save_pickle(bundle_path, final_bundle)

    calibration_payload = dict(calibration_summary)
    calibration_payload.pop("ece_bin_assignments", None)
    save_json(calibration_path, calibration_payload)
    summary = {
        "manifest_csv": str(manifest_csv),
        "total_rows": len(feature_rows),
        "counts_by_stage": dict(Counter(truth)),
        "counts_by_modality": dict(Counter(str(row.get("modality", "")) for row in feature_rows)),
        "counts_by_source_group": dict(Counter(str(row.get("source_group", "")) for row in feature_rows)),
        "counts_by_gene_family": dict(Counter(str(row.get("gene_family", "")) for row in feature_rows if str(row.get("gene_family", "")).strip())),
        "unique_strict_groups": len({str(row.get("strict_group", "")) for row in feature_rows}),
        "best_candidate": {
            "candidate": str(best_run["candidate"]),
            "accuracy": float(best_run["accuracy"]),
            "macro_f1": float(best_run["macro_f1"]),
            "C": float(best_run.get("C", 1.0)),
        },
        "best_overall_candidate": {
            "candidate": str(best_overall["candidate"]),
            "accuracy": float(best_overall["accuracy"]),
            "macro_f1": float(best_overall["macro_f1"]),
        },
        "best_single_candidate_name": final_bundle["best_single_candidate_name"],
        "best_p_model_candidate_name": final_bundle["best_p_model_candidate_name"],
        "best_p_model_family": final_bundle["best_p_model_family"],
        "fusion_weights": list(final_bundle["fusion_weights"]),
        "tau_template": float(final_bundle["tau_template"]),
        "tau_reference": float(final_bundle["tau_reference"]),
        "early05_aux_weight": float(final_bundle["early05_aux_weight"]),
        "forewing_early05_rescue_delta_model": float(final_bundle["forewing_early05_rescue_delta_model"]),
        "forewing_early05_rescue_factor": float(final_bundle["forewing_early05_rescue_factor"]),
        "auto_thresholds_research": research_thresholds,
        "auto_thresholds_deployment": deployment_thresholds,
        "accepted_metrics_research": accepted_metrics(truth, fused_pred, fused_best_prob, fused_margin, research_thresholds),
        "accepted_metrics_deployment": accepted_metrics(truth, fused_pred, fused_best_prob, fused_margin, deployment_thresholds),
        "calibration_summary": calibration_payload,
        "source_group_holdout_summary": _group_accuracy_rows(source_group_holdout_rows, "source_group_holdout_fold", "source_group"),
        "modality_holdout_summary": _group_accuracy_rows(modality_holdout_rows, "modality_holdout_fold", "modality"),
        "gene_family_holdout_summary": _group_accuracy_rows(gene_family_holdout_rows, "gene_family_holdout_fold", "gene_family"),
        "human_in_loop_summary": human_in_loop_rows,
        "robustness_summary": robustness_summary_rows,
        "target_accuracy_reached": bool(float(best_run["accuracy"]) > 0.8467),
        "artifacts": {
            "leaderboard_csv": str(leaderboard_path),
            "oof_predictions_csv": str(oof_path),
            "source_group_holdout_csv": str(source_holdout_path),
            "modality_holdout_csv": str(modality_holdout_path),
            "gene_family_holdout_csv": str(gene_family_holdout_path),
            "ablation_csv": str(ablation_path),
            "calibration_json": str(calibration_path),
            "error_stratification_csv": str(stratification_path),
            "human_in_loop_csv": str(human_in_loop_path),
            "robustness_summary_csv": str(robustness_summary_path),
            "robustness_predictions_csv": str(robustness_predictions_path),
            "validation_scope_csv": str(experiment_scope_path),
            "risk_coverage_curve": str(risk_path),
            "visualization_png": str(viz_path),
            "stage0_5_error_audit_csv": str(hard_audit_path),
            "model_bundle": str(bundle_path),
        },
    }
    save_json(summary_path, summary)
    best_log_path.write_text(
        "\n".join(
            [
                f"# {family_name.title()} Stage Upgrade V1.1 Log",
                "",
                "## Baseline",
                "- old strict OOF target: `> 0.8467`",
                "- focus: `p_model + 0.5 hard case`",
                "",
                "## Data",
                f"- manifest: `{manifest_csv}`",
                f"- rows: `{len(feature_rows)}`",
                f"- unique_strict_groups: `{summary['unique_strict_groups']}`",
                f"- counts_by_stage: `{summary['counts_by_stage']}`",
                f"- counts_by_modality: `{summary['counts_by_modality']}`",
                f"- counts_by_source_group: `{summary['counts_by_source_group']}`",
                "",
                "## p_model candidate pool",
                "- logistic: `shape only`, `mask only`, `shape+mask`, `shape+template`, `shape+template+mask`",
                "- hist: `shape only`, `mask only`, `shape+mask`, `shape+template`, `shape+template+mask`",
                f"- best p_model candidate: `{final_bundle['best_p_model_candidate_name']}`",
                f"- best p_model family: `{final_bundle['best_p_model_family']}`",
                "",
                "## 0.5 hard-case weighting",
                f"- config: `{final_bundle['hard_weight_config_forewing'] or final_bundle['hard_weight_config']}`",
                f"- early05 aux weight: `{float(final_bundle['early05_aux_weight']):.2f}`",
                f"- early05 rescue delta: `{float(final_bundle['forewing_early05_rescue_delta_model']):.2f}`",
                f"- early05 rescue factor: `{float(final_bundle['forewing_early05_rescue_factor']):.2f}`",
                "",
                "## Strict result",
                f"- primary candidate: `{best_run['candidate']}`",
                f"- accuracy: `{float(best_run['accuracy']):.6f}`",
                f"- macro_f1: `{float(best_run['macro_f1']):.6f}`",
                "",
                "## Calibration",
                f"- ECE: `{float(calibration_payload['ece']):.6f}`",
                f"- Brier: `{float(calibration_payload['brier_score']):.6f}`",
                f"- selective_accuracy: `{float(calibration_payload['selective_accuracy']):.6f}`",
                "",
                "## Profiles",
                f"- research_strict: `{final_bundle['profile_settings']['research_strict']}`",
                f"- deployment_balanced: `{final_bundle['profile_settings']['deployment_balanced']}`",
                "",
                "## Necessary tests",
                "- strict OOF",
                "- leave-one-source_group-out",
                "- leave-one-modality-out",
                "- ablation study",
                "- calibration (ECE/Brier/risk-coverage/selective accuracy)",
                "- error stratification",
                "",
                "## Bonus tests",
                "- robustness test",
                "- human-in-the-loop simulation",
                "- leave-one-gene-family-out (supplementary, n>=3 only)",
                "",
                "## Validation scope",
                "- strict OOF / ablation / calibration / error stratification / human-in-the-loop are all run on the full multimodal forewing-stage manifest",
                "- source_group holdout and modality holdout each cover the full manifest through repeated held-out folds",
                "- gene-family holdout is supplementary and only reported for families with at least 3 samples",
                "- robustness is run on every sample across all configured perturbations",
                "",
                "## New experiment snapshots",
                f"- source_group holdout summary: `{summary['source_group_holdout_summary']}`",
                f"- modality holdout summary: `{summary['modality_holdout_summary']}`",
                f"- gene-family holdout summary: `{summary['gene_family_holdout_summary']}`",
                f"- human-in-the-loop: `{summary['human_in_loop_summary']}`",
                "",
                "## Recommendation",
                "- research profile: `research_strict`",
                "- deployment profile: `deployment_balanced`",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "bundle": final_bundle,
        "feature_rows": feature_rows,
        "oof_rows": oof_rows,
        "leaderboard_rows": leaderboard_rows,
    }


def _probability_vector_from_component(bundle: dict[str, object], feature_row: dict[str, object], row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray]) -> np.ndarray:
    kind = str(bundle.get("bundle_kind", "")).strip().lower()
    if kind in {"hist_multiclass", "logistic_multiclass"}:
        feature_names = list(bundle["feature_names"])
        pred_prob = np.asarray(bundle["model"].predict_proba(row_matrix_fn([feature_row], feature_names)), dtype=np.float64)
        classes = [str(value) for value in getattr(bundle["model"], "classes_", STAGE_LABELS)]
        return fill_probability_matrix(pred_prob, classes)[0]
    if kind == "ordinal_logistic":
        return ordinal_probabilities_from_models(bundle, [feature_row], row_matrix_fn)[0]
    raise ValueError(f"Unsupported component kind: {kind}")


def apply_stage_fusion_bundle(
    bundle: dict[str, object],
    feature_row: dict[str, object],
    row_matrix_fn: Callable[[list[dict[str, object]], list[str]], np.ndarray],
    stage_profile: str = "",
) -> dict[str, object]:
    profile_name = str(stage_profile or bundle.get("default_stage_profile", "deployment_balanced")).strip()
    profiles = dict(bundle.get("profile_settings", {}))
    profile = dict(profiles.get(profile_name, profiles.get("deployment_balanced", {})))
    model_component = dict(bundle.get("model_component", bundle.get("hist_component", {})))
    model_prob = _probability_vector_from_component(model_component, feature_row, row_matrix_fn)
    ord_template_prob = _probability_vector_from_component(dict(bundle["ordinal_template_component"]), feature_row, row_matrix_fn)
    ord_shape_prob = _probability_vector_from_component(dict(bundle["ordinal_shape_component"]), feature_row, row_matrix_fn)
    ordinal_prob = blend_ordinal_probabilities(ord_template_prob.reshape(1, -1), ord_shape_prob.reshape(1, -1))[0]
    early05_prob = 0.5
    if bool(bundle.get("early05_aux_enabled", False)) and bundle.get("early05_aux_component") is not None:
        early05_feature_names = list(bundle.get("early05_aux_feature_names", []))
        early05_component = bundle["early05_aux_component"]
        early05_prob = float(np.asarray(early05_component.predict_proba(row_matrix_fn([feature_row], early05_feature_names))[:, 1], dtype=np.float64)[0])
        model_prob = augment_model_probabilities(model_prob.reshape(1, -1), np.asarray([early05_prob], dtype=np.float64), float(bundle.get("early05_aux_weight", 0.0)))[0]
    reference_payload = reference_bank_posteriors(
        feature_row,
        bank=dict(bundle["reference_bank"]),
        row_matrix_fn=row_matrix_fn,
        tau_template=float(bundle.get("tau_template", 1.0)),
        tau_reference=float(bundle.get("tau_reference", 1.0)),
    )
    base_fused_prob = fuse_probabilities(
        model_prob.reshape(1, -1),
        ordinal_prob.reshape(1, -1),
        np.asarray(reference_payload["p_template"], dtype=np.float64).reshape(1, -1),
        np.asarray(reference_payload["p_reference"], dtype=np.float64).reshape(1, -1),
        fusion_weights=tuple(bundle.get("fusion_weights", FUSION_WEIGHT_GRID[0])),
    )[0]
    rescue_applied = False
    fused_prob = np.asarray(base_fused_prob, dtype=np.float64)
    if bool(bundle.get("forewing_early05_rescue_enabled", False)):
        rescued_prob, rescue_flags = apply_early05_rescue(
            model_prob.reshape(1, -1),
            ordinal_prob.reshape(1, -1),
            np.asarray(reference_payload["p_template"], dtype=np.float64).reshape(1, -1),
            np.asarray(reference_payload["p_reference"], dtype=np.float64).reshape(1, -1),
            base_fused_prob.reshape(1, -1),
            np.asarray([early05_prob], dtype=np.float64),
            delta_model=float(bundle.get("forewing_early05_rescue_delta_model", 0.0)),
            rescue_factor=float(bundle.get("forewing_early05_rescue_factor", 1.0)),
        )
        fused_prob = np.asarray(rescued_prob[0], dtype=np.float64)
        rescue_applied = bool(rescue_flags[0])
    raw_single_name = str(bundle.get("best_single_candidate_name", bundle.get("best_p_model_candidate_name", "hist_shape_template")))
    if raw_single_name == "ordinal_template_mask64":
        raw_prob = ord_template_prob
    elif raw_single_name == "ordinal_shape_template_mask32":
        raw_prob = ord_shape_prob
    else:
        raw_prob = model_prob

    def summarize(prob: np.ndarray) -> tuple[str, float, float]:
        pred = STAGE_LABELS[int(np.argmax(prob))]
        best_prob = float(np.max(prob))
        sorted_prob = np.sort(prob)
        margin = float(sorted_prob[-1] - sorted_prob[-2]) if len(sorted_prob) >= 2 else best_prob
        return pred, best_prob, margin

    raw_pred, raw_conf, raw_margin = summarize(raw_prob)
    ordinal_pred, ordinal_conf, ordinal_margin = summarize(ordinal_prob)
    fused_pred, fused_conf, fused_margin = summarize(fused_prob)
    maturity_score = float(np.sum(fused_prob * STAGE_NUMERIC_VALUES))
    ordinal_stage_gap = abs(STAGE_TO_INDEX[raw_pred] - STAGE_TO_INDEX[ordinal_pred])
    maturity_offset = abs(maturity_score - stage_numeric(fused_pred))
    ordinal_conflict = bool(ordinal_stage_gap >= 2 or maturity_offset > float(profile.get("maturity_offset_max", 0.20)))
    reference_distance = float(reference_payload["best_reference_distance"])
    distance_ok = reference_distance <= float(profile.get("reference_nn_max", 1e9))
    far_override = bool(
        not distance_ok
        and bool(profile.get("allow_far_override", False))
        and fused_conf >= float(profile.get("far_override_conf_min", 1.1))
        and fused_margin >= float(profile.get("far_override_margin_min", 1.1))
    )
    auto_ok = bool(
        fused_conf >= float(profile.get("prob_min", 1.1))
        and fused_margin >= float(profile.get("margin_min", 1.1))
        and not ordinal_conflict
        and (distance_ok or far_override)
    )
    review_reasons: list[str] = []
    if fused_conf < float(profile.get("prob_min", 1.1)):
        review_reasons.append("low_fused_conf")
    if fused_margin < float(profile.get("margin_min", 1.1)):
        review_reasons.append("low_fused_margin")
    if ordinal_conflict:
        review_reasons.append("ordinal_conflict")
    if not distance_ok and not far_override:
        review_reasons.append("far_from_reference_bank")
    decision = f"auto_{bundle.get('family_name', 'wing')}_{fused_pred}" if auto_ok else "uncertain_review"
    return {
        "stage_model_pred_raw": raw_pred,
        "stage_model_conf_raw": raw_conf,
        "stage_model_margin_raw": raw_margin,
        "best_p_model_candidate_name": str(bundle.get("best_p_model_candidate_name", "")),
        "best_p_model_family": str(bundle.get("best_p_model_family", "")),
        "p_model_pred": raw_pred,
        "p_model_conf": raw_conf,
        "early05_aux_score": float(early05_prob),
        "early05_rescue_applied": rescue_applied,
        "ordinal_stage_pred": ordinal_pred,
        "ordinal_stage_conf": ordinal_conf,
        "ordinal_stage_margin": ordinal_margin,
        "fused_stage_pred": fused_pred,
        "fused_stage_conf": fused_conf,
        "fused_stage_margin": fused_margin,
        "maturity_score": maturity_score,
        "template_stage_probabilities": dict(reference_payload["template_probabilities"]),
        "reference_stage_probabilities": dict(reference_payload["reference_probabilities"]),
        "template_stage_costs": dict(reference_payload["template_costs"]),
        "reference_stage_costs": dict(reference_payload["reference_costs"]),
        "reference_feature_distances": dict(reference_payload["reference_feature_distances"]),
        "reference_nn_distances": dict(reference_payload["reference_nn_distances"]),
        "reference_topk_distances": dict(reference_payload["reference_topk_distances"]),
        "reference_best_distance": reference_distance,
        "best_reference_id": reference_payload["best_reference_id"],
        "best_reference_stage": reference_payload["best_reference_stage"],
        "stage_profile": profile_name,
        "ordinal_conflict": ordinal_conflict,
        "far_reference_override_applied": far_override,
        "decision": decision,
        "review_reason": ",".join(review_reasons),
        "p_model_probabilities": {label: float(model_prob[index]) for index, label in enumerate(STAGE_LABELS)},
        "probabilities": {label: float(fused_prob[index]) for index, label in enumerate(STAGE_LABELS)},
    }
