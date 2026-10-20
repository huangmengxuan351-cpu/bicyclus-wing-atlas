from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from pose_landmark_utils import LABELS, read_csv_rows, write_csv_rows
from shape_priority_hindwing_utils import (
    DEFAULT_CLASSIFIER_DIR,
    DEFAULT_GT100_CLEANING_CSV,
    DEFAULT_LANDMARK_WEIGHTS,
    DEFAULT_STAGE_FEATURES_CSV,
    add_shape_features_to_stage_row,
    build_wing_feature_row,
    clamp01,
    derive_split_group,
    load_embedding_baseline_rows,
    load_training_manifest_rows,
    rows_to_matrix,
    safe_float,
    save_csv,
    save_json,
    stage_feature_numeric_columns,
    wing_feature_numeric_columns,
)


WING_LABELS = ["forewing", "hindwing"]


def normalize_model_type(model_type: str) -> str:
    normalized = str(model_type).strip().lower()
    aliases = {
        "hist": "hist_gbdt",
        "histgradientboosting": "hist_gbdt",
        "hist_gbdt": "hist_gbdt",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "random_forest": "random_forest",
        "template": "template_cost",
        "template_cost": "template_cost",
    }
    return aliases.get(normalized, normalized)


def parse_model_types(model_spec: str | list[str], allow_template_cost: bool = False) -> list[str]:
    if isinstance(model_spec, (list, tuple)):
        raw_items = [str(item).strip() for item in model_spec]
    else:
        raw_items = [part.strip() for part in str(model_spec or "").split(",")]
    items: list[str] = []
    for item in raw_items:
        if not item:
            continue
        normalized = normalize_model_type(item)
        if normalized == "template_cost" and not allow_template_cost:
            continue
        items.append(normalized)
    unique_items: list[str] = []
    for item in items:
        if item not in unique_items:
            unique_items.append(item)
    return unique_items


def build_classifier(model_type: str, random_state: int) -> object:
    normalized = normalize_model_type(model_type)
    if normalized == "hist_gbdt":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            min_samples_leaf=5,
            l2_regularization=0.02,
            random_state=int(random_state),
    )
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=int(random_state),
        n_jobs=1,
    )


def group_train_val_split(rows: list[dict[str, object]], test_size: float, random_state: int) -> tuple[list[int], list[int]]:
    groups = np.asarray([str(row.get("split_group", "")) or str(row.get("id", "")) for row in rows], dtype=object)
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 2 or len(rows) < 8:
        indices = list(range(len(rows)))
        cutoff = max(1, int(round(len(indices) * (1.0 - float(test_size)))))
        cutoff = min(max(cutoff, 1), len(indices) - 1)
        return indices[:cutoff], indices[cutoff:]
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
    train_idx, val_idx = next(splitter.split(np.zeros(len(rows)), groups=groups))
    return train_idx.tolist(), val_idx.tolist()


def subset_rows(rows: list[dict[str, object]], indices: list[int]) -> list[dict[str, object]]:
    return [rows[index] for index in indices]


def clip_probability_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.clip(matrix, 1e-6, 1.0 - 1e-6)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return matrix / row_sums


def weighted_average_probabilities(probabilities: list[np.ndarray], weights: list[float] | np.ndarray) -> np.ndarray:
    if not probabilities:
        return np.zeros((0, 0), dtype=np.float64)
    normalized_weights = np.asarray(weights, dtype=np.float64)
    if normalized_weights.size != len(probabilities):
        normalized_weights = np.ones(len(probabilities), dtype=np.float64)
    normalized_weights = np.clip(normalized_weights, 0.0, np.inf)
    if float(normalized_weights.sum()) <= 0.0:
        normalized_weights = np.ones(len(probabilities), dtype=np.float64)
    normalized_weights = normalized_weights / normalized_weights.sum()
    combined = np.zeros_like(np.asarray(probabilities[0], dtype=np.float64), dtype=np.float64)
    for weight, matrix in zip(normalized_weights, probabilities):
        combined += float(weight) * np.asarray(matrix, dtype=np.float64)
    if combined.ndim == 2:
        return clip_probability_columns(combined)
    return np.clip(combined, 1e-6, 1.0 - 1e-6)


def fit_single_binary_probability_model(
    rows: list[dict[str, object]],
    feature_names: list[str],
    target_label: str,
    model_type: str,
    random_state: int,
) -> dict[str, object]:
    X = rows_to_matrix(rows, feature_names)
    y = np.asarray([1 if str(row.get("wing_type_label", "")) == target_label else 0 for row in rows], dtype=np.int32)
    groups = np.asarray([str(row.get("split_group", "")) or str(row.get("id", "")) for row in rows], dtype=object)
    final_model = build_classifier(model_type, random_state=random_state)
    calibrator = None
    oof_prob = None
    unique_groups = sorted(set(groups.tolist()))
    n_splits = max(2, min(5, len(unique_groups)))
    if len(unique_groups) >= 2 and len(set(y.tolist())) >= 2:
        try:
            splitter = GroupKFold(n_splits=n_splits)
            oof_prob = np.zeros(len(rows), dtype=np.float64)
            for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups)):
                fold_model = build_classifier(model_type, random_state=random_state + fold_index + 1)
                fold_model.fit(X[train_idx], y[train_idx])
                oof_prob[test_idx] = fold_model.predict_proba(X[test_idx])[:, 1]
            calibrator = LogisticRegression(class_weight="balanced", random_state=int(random_state + 77))
            calibrator.fit(oof_prob.reshape(-1, 1), y)
        except Exception:
            calibrator = None
            oof_prob = None
    final_model.fit(X, y)
    return {
        "feature_names": feature_names,
        "classes": [0, 1],
        "target_label": target_label,
        "model_type": model_type,
        "base_model": final_model,
        "calibrator": calibrator,
        "oof_probability": oof_prob,
    }


def tune_binary_ensemble_weights(
    truth: np.ndarray,
    component_probabilities: list[np.ndarray],
) -> list[float]:
    if len(component_probabilities) <= 1:
        return [1.0] * len(component_probabilities)
    truth = np.asarray(truth, dtype=bool)
    best_weights = [1.0 / len(component_probabilities)] * len(component_probabilities)
    best_accuracy = -1.0
    best_confidence = -1.0
    if len(component_probabilities) == 2:
        grid = np.linspace(0.0, 1.0, 21)
        candidates = [[float(weight), float(1.0 - weight)] for weight in grid]
    else:
        step = 0.1
        candidates = []
        values = np.arange(0.0, 1.0 + 1e-9, step)
        for first in values:
            for second in values:
                third = 1.0 - first - second
                if third < -1e-9:
                    continue
                candidates.append([float(first), float(second), float(max(0.0, third))])
    for weights in candidates:
        combined = weighted_average_probabilities(component_probabilities, weights)
        predicted = combined >= 0.5
        accuracy = float(np.mean(predicted == truth)) if truth.size else 0.0
        confidence = float(np.mean(np.abs(combined - 0.5))) if combined.size else 0.0
        if accuracy > best_accuracy or (math.isclose(accuracy, best_accuracy) and confidence > best_confidence):
            best_accuracy = accuracy
            best_confidence = confidence
            best_weights = list(weights)
    return best_weights


def fit_binary_probability_model(
    rows: list[dict[str, object]],
    feature_names: list[str],
    target_label: str,
    model_type: str,
    random_state: int,
) -> dict[str, object]:
    model_types = parse_model_types(model_type, allow_template_cost=False)
    if not model_types:
        model_types = ["hist_gbdt"]
    if len(model_types) == 1:
        return fit_single_binary_probability_model(
            rows,
            feature_names=feature_names,
            target_label=target_label,
            model_type=model_types[0],
            random_state=random_state,
        )
    components = []
    for index, component_type in enumerate(model_types):
        component_bundle = fit_single_binary_probability_model(
            rows,
            feature_names=feature_names,
            target_label=target_label,
            model_type=component_type,
            random_state=random_state + index * 31,
        )
        components.append(component_bundle)
    return {
        "ensemble_type": "binary_probability_average",
        "feature_names": feature_names,
        "target_label": target_label,
        "components": components,
        "weights": [1.0 / len(components)] * len(components),
    }


def apply_binary_probability_model(bundle: dict[str, object], rows: list[dict[str, object]]) -> np.ndarray:
    if "components" in bundle:
        component_probabilities = [apply_binary_probability_model(component, rows) for component in bundle["components"]]
        return weighted_average_probabilities(component_probabilities, bundle.get("weights", []))
    X = rows_to_matrix(rows, list(bundle["feature_names"]))
    raw = np.asarray(bundle["base_model"].predict_proba(X)[:, 1], dtype=np.float64)
    calibrator = bundle.get("calibrator")
    if calibrator is None:
        return raw
    calibrated = np.asarray(calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=np.float64)
    return np.clip(calibrated, 1e-6, 1.0 - 1e-6)


def fit_single_multiclass_probability_model(
    rows: list[dict[str, object]],
    feature_names: list[str],
    label_key: str,
    labels: list[str],
    model_type: str,
    random_state: int,
) -> dict[str, object]:
    X = rows_to_matrix(rows, feature_names)
    y = np.asarray([str(row.get(label_key, "")) for row in rows], dtype=object)
    final_model = build_classifier(model_type, random_state=random_state)
    final_model.fit(X, y)
    return {
        "feature_names": feature_names,
        "classes": labels,
        "label_key": label_key,
        "model_type": model_type,
        "base_model": final_model,
        "calibrators": {label: None for label in labels},
    }


def template_cost_probabilities(rows: list[dict[str, object]], labels: list[str]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(labels)), dtype=np.float64)
    matrix = np.zeros((len(rows), len(labels)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        costs = np.asarray(
            [safe_float(row.get(f"cost_to_{label.replace('.', '_')}", row.get("best_template_cost", 0.0)), 0.0) for label in labels],
            dtype=np.float64,
        )
        finite_costs = costs[np.isfinite(costs)]
        if finite_costs.size != len(labels):
            matrix[row_index] = np.full(len(labels), 1.0 / max(1, len(labels)), dtype=np.float64)
            continue
        shifted = costs - float(costs.min())
        scale = float(np.std(shifted))
        if scale <= 1e-6:
            positive = shifted[shifted > 1e-6]
            scale = float(np.mean(positive)) if positive.size else 1.0
        logits = -shifted / max(scale, 1e-6)
        logits -= float(logits.max())
        exp_logits = np.exp(logits)
        total = float(exp_logits.sum())
        matrix[row_index] = exp_logits / max(total, 1e-12)
    return clip_probability_columns(matrix)


def tune_multiclass_ensemble_weights(
    truth: list[str],
    component_probabilities: list[np.ndarray],
    labels: list[str],
) -> list[float]:
    if len(component_probabilities) <= 1:
        return [1.0] * len(component_probabilities)
    best_weights = [1.0 / len(component_probabilities)] * len(component_probabilities)
    best_accuracy = -1.0
    best_macro_f1 = -1.0
    best_confidence = -1.0
    if len(component_probabilities) == 2:
        values = np.linspace(0.0, 1.0, 21)
        candidates = [[float(first), float(1.0 - first)] for first in values]
    elif len(component_probabilities) == 3:
        step = 0.1
        values = np.arange(0.0, 1.0 + 1e-9, step)
        candidates = []
        for first in values:
            for second in values:
                third = 1.0 - first - second
                if third < -1e-9:
                    continue
                candidates.append([float(first), float(second), float(max(0.0, third))])
    else:
        return best_weights
    for weights in candidates:
        combined = weighted_average_probabilities(component_probabilities, weights)
        predicted = [labels[int(np.argmax(row))] for row in combined]
        accuracy = float(accuracy_score(truth, predicted)) if truth else 0.0
        macro = float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)) if truth else 0.0
        confidence = float(np.mean(np.max(combined, axis=1))) if combined.size else 0.0
        if (
            accuracy > best_accuracy
            or (math.isclose(accuracy, best_accuracy) and macro > best_macro_f1)
            or (
                math.isclose(accuracy, best_accuracy)
                and math.isclose(macro, best_macro_f1)
                and confidence > best_confidence
            )
        ):
            best_accuracy = accuracy
            best_macro_f1 = macro
            best_confidence = confidence
            best_weights = list(weights)
    return best_weights


def fit_multiclass_probability_model(
    rows: list[dict[str, object]],
    feature_names: list[str],
    label_key: str,
    labels: list[str],
    model_type: str,
    random_state: int,
) -> dict[str, object]:
    model_types = parse_model_types(model_type, allow_template_cost=True)
    if not model_types:
        model_types = ["hist_gbdt"]
    if len(model_types) == 1 and model_types[0] != "template_cost":
        return fit_single_multiclass_probability_model(
            rows,
            feature_names=feature_names,
            label_key=label_key,
            labels=labels,
            model_type=model_types[0],
            random_state=random_state,
        )
    components = []
    for index, component_type in enumerate(model_types):
        if component_type == "template_cost":
            components.append(
                {
                    "component_type": "template_cost",
                    "feature_names": feature_names,
                    "classes": labels,
                    "label_key": label_key,
                }
            )
            continue
        component_bundle = fit_single_multiclass_probability_model(
            rows,
            feature_names=feature_names,
            label_key=label_key,
            labels=labels,
            model_type=component_type,
            random_state=random_state + index * 31,
        )
        component_bundle["component_type"] = component_type
        components.append(component_bundle)
    return {
        "ensemble_type": "multiclass_probability_average",
        "feature_names": feature_names,
        "classes": labels,
        "label_key": label_key,
        "components": components,
        "weights": [1.0 / len(components)] * len(components),
    }


def apply_multiclass_probability_model(bundle: dict[str, object], rows: list[dict[str, object]]) -> np.ndarray:
    if "components" in bundle:
        component_probabilities = []
        for component in bundle["components"]:
            if component.get("component_type") == "template_cost":
                component_probabilities.append(template_cost_probabilities(rows, list(bundle["classes"])))
            else:
                component_probabilities.append(apply_multiclass_probability_model(component, rows))
        return weighted_average_probabilities(component_probabilities, bundle.get("weights", []))
    if bundle.get("component_type") == "template_cost":
        return template_cost_probabilities(rows, list(bundle["classes"]))
    X = rows_to_matrix(rows, list(bundle["feature_names"]))
    model = bundle["base_model"]
    raw_prob = np.asarray(model.predict_proba(X), dtype=np.float64)
    class_index = {str(label): index for index, label in enumerate(model.classes_)}
    labels = list(bundle["classes"])
    aligned = np.zeros((len(rows), len(labels)), dtype=np.float64)
    for label_index, label in enumerate(labels):
        if label in class_index:
            aligned[:, label_index] = raw_prob[:, class_index[label]]
    calibrators = dict(bundle.get("calibrators", {}))
    adjusted = np.zeros_like(aligned)
    for label_index, label in enumerate(labels):
        calibrator = calibrators.get(label)
        column = aligned[:, label_index]
        if calibrator is None:
            adjusted[:, label_index] = column
        else:
            adjusted[:, label_index] = calibrator.predict_proba(column.reshape(-1, 1))[:, 1]
    return clip_probability_columns(adjusted)


def choose_precision_threshold(
    scores: np.ndarray,
    truth_positive: np.ndarray,
    target_precision: float,
    high_is_positive: bool = True,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    truth_positive = np.asarray(truth_positive, dtype=bool)
    unique_thresholds = sorted(set(scores.tolist()), reverse=bool(high_is_positive))
    best = {
        "threshold": float(unique_thresholds[0] if unique_thresholds else 0.5),
        "precision": 0.0,
        "coverage": 0.0,
        "accepted": 0.0,
    }
    for threshold in unique_thresholds:
        if high_is_positive:
            selected = scores >= threshold
        else:
            selected = scores <= threshold
        accepted = int(selected.sum())
        if accepted <= 0:
            continue
        precision = float(truth_positive[selected].mean())
        coverage = float(selected.mean())
        candidate = {
            "threshold": float(threshold),
            "precision": precision,
            "coverage": coverage,
            "accepted": float(accepted),
        }
        if precision >= target_precision:
            if (
                candidate["coverage"] > best["coverage"]
                or (math.isclose(candidate["coverage"], best["coverage"]) and candidate["precision"] > best["precision"])
            ):
                best = candidate
        elif best["precision"] < target_precision:
            if (
                candidate["precision"] > best["precision"]
                or (math.isclose(candidate["precision"], best["precision"]) and candidate["coverage"] > best["coverage"])
            ):
                best = candidate
    return best


def tune_wing_thresholds(
    rows: list[dict[str, object]],
    hind_probability: np.ndarray,
    target_precision: float,
) -> dict[str, float]:
    if not rows:
        return {
            "hind_auto_min": 0.5,
            "fore_auto_max_hind_prob": 0.5,
            "target_auto_precision": float(target_precision),
            "accepted_auto_coverage": 0.0,
            "accepted_hind_precision": 0.0,
            "accepted_fore_precision": 0.0,
        }
    hind_probability = np.asarray(hind_probability, dtype=np.float64)
    hind_truth = np.asarray([str(row.get("wing_type_label", "")) == "hindwing" for row in rows], dtype=bool)
    fore_truth = np.asarray([str(row.get("wing_type_label", "")) == "forewing" for row in rows], dtype=bool)
    candidates = sorted(set(hind_probability.tolist()))
    if 0.0 not in candidates:
        candidates = [0.0] + candidates
    if 1.0 not in candidates:
        candidates = candidates + [1.0]
    best = {
        "hind_auto_min": 0.5,
        "fore_auto_max_hind_prob": 0.5,
        "target_auto_precision": float(target_precision),
        "accepted_auto_coverage": 0.0,
        "accepted_hind_precision": 0.0,
        "accepted_fore_precision": 0.0,
        "_target_met": False,
        "_score": -1.0,
    }
    for fore_threshold in candidates:
        for hind_threshold in candidates:
            if fore_threshold > hind_threshold:
                continue
            hind_selected = hind_probability >= hind_threshold
            fore_selected = hind_probability <= fore_threshold
            auto_selected = hind_selected | fore_selected
            hind_precision = float(hind_truth[hind_selected].mean()) if int(hind_selected.sum()) > 0 else 0.0
            fore_precision = float(fore_truth[fore_selected].mean()) if int(fore_selected.sum()) > 0 else 0.0
            auto_coverage = float(auto_selected.mean()) if auto_selected.size else 0.0
            target_met = hind_precision >= target_precision and fore_precision >= target_precision
            score = min(hind_precision, fore_precision)
            candidate = {
                "hind_auto_min": float(hind_threshold),
                "fore_auto_max_hind_prob": float(fore_threshold),
                "target_auto_precision": float(target_precision),
                "accepted_auto_coverage": auto_coverage,
                "accepted_hind_precision": hind_precision,
                "accepted_fore_precision": fore_precision,
                "_target_met": target_met,
                "_score": score,
            }
            if target_met:
                if (
                    not best["_target_met"]
                    or candidate["accepted_auto_coverage"] > best["accepted_auto_coverage"]
                    or (
                        math.isclose(candidate["accepted_auto_coverage"], best["accepted_auto_coverage"])
                        and score > best["_score"]
                    )
                ):
                    best = candidate
            elif not best["_target_met"]:
                if (
                    score > best["_score"]
                    or (
                        math.isclose(score, best["_score"])
                        and candidate["accepted_auto_coverage"] > best["accepted_auto_coverage"]
                    )
                ):
                    best = candidate
    best.pop("_target_met", None)
    best.pop("_score", None)
    return best


def stage_decisions_from_thresholds(
    rows: list[dict[str, object]],
    probabilities: np.ndarray,
    thresholds: dict[str, float],
) -> list[dict[str, object]]:
    """Assign a stage decision (auto or uncertain_review) for each hindwing row.

    There is no open-set rejection bucket.  Every row receives a stage prediction
    (the highest-probability class).  When the model is not confident enough the
    decision is ``uncertain_review``; the ``stage_pred`` field still carries the
    best guess so human reviewers have a reference.
    """
    decisions: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        probs = np.asarray(probabilities[row_index], dtype=np.float64)
        best_index = int(np.argmax(probs))
        best_label = LABELS[best_index]
        best_prob = float(probs[best_index])
        ordered = np.sort(probs)[::-1]
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
        best_template_label = str(row.get("best_template_label", "")).strip() or str(row.get("best_template", "")).strip() or best_label
        best_cost = safe_float(
            row.get(f"cost_to_{best_label.replace('.', '_')}", row.get("best_template_cost", 0.0)), 0.0
        )
        review_reasons = []
        if best_prob < thresholds["stage_prob_min"]:
            review_reasons.append("low_stage_prob")
        if margin < thresholds["stage_margin_min"]:
            review_reasons.append("low_stage_margin")
        if review_reasons:
            decision = "uncertain_review"
            review_reason = "+".join(review_reasons)
        else:
            decision = f"auto_hindwing_{best_label}"
            review_reason = ""
        decisions.append(
            {
                **row,
                "stage_pred": best_label,        # always populated — reference for reviewer
                "stage_conf": best_prob,
                "stage_margin": margin,
                "best_template": best_template_label,
                "template_margin": safe_float(row.get("template_margin", 0.0), 0.0),
                "nearest_template_cost": best_cost,
                "decision": decision,
                "review_reason": review_reason,
            }
        )
    return decisions


def tune_stage_thresholds(
    known_rows: list[dict[str, object]],
    known_probabilities: np.ndarray,
    target_precision: float,
) -> dict[str, float]:
    if not known_rows:
        return {
            "stage_prob_min": 0.0,
            "stage_margin_min": 0.0,
            "accepted_precision": 0.0,
            "accepted_coverage": 0.0,
            "full_val_accuracy": 0.0,
        }
    known_true = np.asarray([str(row.get("true_label", "")) for row in known_rows], dtype=object)
    known_pred = np.asarray([LABELS[int(np.argmax(probs))] for probs in known_probabilities], dtype=object)
    full_accuracy = float((known_pred == known_true).mean()) if len(known_rows) else 0.0
    known_probabilities = np.asarray(known_probabilities, dtype=np.float64)
    best_prob = np.max(known_probabilities, axis=1)
    sorted_prob = np.sort(known_probabilities, axis=1)[:, ::-1]
    margin = sorted_prob[:, 0] - sorted_prob[:, 1]
    correctness = known_pred == known_true
    prob_candidates = sorted(set(best_prob.tolist()))
    margin_candidates = sorted(set(margin.tolist()))
    if 0.0 not in prob_candidates:
        prob_candidates = [0.0] + prob_candidates
    if 0.0 not in margin_candidates:
        margin_candidates = [0.0] + margin_candidates
    best = {
        "stage_prob_min": 0.0,
        "stage_margin_min": 0.0,
        "accepted_precision": full_accuracy,
        "accepted_coverage": 1.0,
        "full_val_accuracy": full_accuracy,
        "_target_met": full_accuracy >= target_precision,
    }
    for prob_threshold in prob_candidates:
        for margin_threshold in margin_candidates:
            accepted = (best_prob >= prob_threshold) & (margin >= margin_threshold)
            accepted_count = int(accepted.sum())
            if accepted_count <= 0:
                continue
            precision = float(correctness[accepted].mean())
            coverage = float(accepted.mean())
            candidate = {
                "stage_prob_min": float(prob_threshold),
                "stage_margin_min": float(margin_threshold),
                "accepted_precision": precision,
                "accepted_coverage": coverage,
                "full_val_accuracy": full_accuracy,
                "_target_met": precision >= target_precision,
            }
            if candidate["_target_met"]:
                if (
                    not best["_target_met"]
                    or candidate["accepted_coverage"] > best["accepted_coverage"]
                    or (
                        math.isclose(candidate["accepted_coverage"], best["accepted_coverage"])
                        and candidate["accepted_precision"] > best["accepted_precision"]
                    )
                ):
                    best = candidate
            elif not best["_target_met"]:
                if (
                    candidate["accepted_precision"] > best["accepted_precision"]
                    or (
                        math.isclose(candidate["accepted_precision"], best["accepted_precision"])
                        and candidate["accepted_coverage"] > best["accepted_coverage"]
                    )
                ):
                    best = candidate
    best.pop("_target_met", None)
    return best


def compute_stage_mahalanobis_bundle(rows: list[dict[str, object]], feature_names: list[str]) -> dict[str, object]:
    X = rows_to_matrix(rows, feature_names)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    estimator = LedoitWolf().fit(X_scaled)
    precision = np.asarray(estimator.precision_, dtype=np.float64)
    centers = {}
    for label in LABELS:
        label_mask = np.asarray([str(row.get("true_label", "")) == label for row in rows], dtype=bool)
        if label_mask.any():
            centers[label] = X_scaled[label_mask].mean(axis=0)
    return {
        "feature_names": feature_names,
        "scaler": scaler,
        "precision_matrix": precision,
        "class_centers": centers,
    }


def mahalanobis_to_nearest_center(bundle: dict[str, object], rows: list[dict[str, object]]) -> np.ndarray:
    X = rows_to_matrix(rows, list(bundle["feature_names"]))
    X_scaled = bundle["scaler"].transform(X)
    precision = np.asarray(bundle["precision_matrix"], dtype=np.float64)
    centers = dict(bundle["class_centers"])
    distances = np.full(len(rows), np.inf, dtype=np.float64)
    for center in centers.values():
        center_array = np.asarray(center, dtype=np.float64).reshape(1, -1)
        delta = X_scaled - center_array
        values = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", delta, precision, delta)))
        distances = np.minimum(distances, values)
    distances[~np.isfinite(distances)] = 1e6
    return distances


def evaluate_wing_triage(
    rows: list[dict[str, object]],
    hind_probability: np.ndarray,
    thresholds: dict[str, float],
) -> list[dict[str, object]]:
    evaluated: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        p_hind = float(hind_probability[row_index])
        if p_hind >= thresholds["hind_auto_min"]:
            decision = "auto_hindwing"
            review_reason = ""
            wing_pred = "hindwing"
        elif p_hind <= thresholds["fore_auto_max_hind_prob"]:
            decision = "auto_forewing"
            review_reason = ""
            wing_pred = "forewing"
        else:
            decision = "uncertain_review"
            review_reason = "wing_probability_gap"
            wing_pred = "uncertain"
        evaluated.append(
            {
                **row,
                "wing_type_pred": wing_pred,
                "wing_type_conf": p_hind,
                "decision": decision,
                "review_reason": review_reason,
            }
        )
    return evaluated


def summarise_wing_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    hind_auto = [row for row in rows if row["decision"] == "auto_hindwing"]
    fore_auto = [row for row in rows if row["decision"] == "auto_forewing"]
    hind_precision = (
        float(np.mean([str(row.get("wing_type_label", "")) == "hindwing" for row in hind_auto])) if hind_auto else 0.0
    )
    hind_recall = (
        float(
            np.mean(
                [
                    str(row.get("wing_type_label", "")) == "hindwing" and row["decision"] == "auto_hindwing"
                    for row in rows
                    if str(row.get("wing_type_label", "")) == "hindwing"
                ]
            )
        )
        if any(str(row.get("wing_type_label", "")) == "hindwing" for row in rows)
        else 0.0
    )
    fore_precision = (
        float(np.mean([str(row.get("wing_type_label", "")) == "forewing" for row in fore_auto])) if fore_auto else 0.0
    )
    review_rate = float(np.mean([row["decision"] == "uncertain_review" for row in rows])) if rows else 0.0
    return {
        "hindwing_precision": hind_precision,
        "hindwing_recall": hind_recall,
        "forewing_precision": fore_precision,
        "uncertain_review_rate": review_rate,
        "auto_hindwing_count": float(len(hind_auto)),
        "auto_forewing_count": float(len(fore_auto)),
    }


def comparison_rows_from_embedding() -> list[dict[str, object]]:
    comparison = []
    for row in load_embedding_baseline_rows():
        comparison.append(
            {
                "system": str(row.get("model", "")),
                "accuracy": safe_float(row.get("accuracy", 0.0), 0.0),
                "macro_f1": safe_float(row.get("macro_f1", 0.0), 0.0),
                "notes": "existing_embedding_baseline",
            }
        )
    return comparison


def save_pickle(path: str | Path, payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the shape-priority hindwing cascade.")
    parser.add_argument("--gt100-cleaning-csv", default=str(DEFAULT_GT100_CLEANING_CSV))
    parser.add_argument("--stage-features-csv", default=str(DEFAULT_STAGE_FEATURES_CSV))
    parser.add_argument("--stage-project-dir", default=str(Path(DEFAULT_STAGE_FEATURES_CSV).parents[0]))
    parser.add_argument("--output-dir", default=str(DEFAULT_CLASSIFIER_DIR))
    parser.add_argument("--other-stage-manifest-csv", default="")
    parser.add_argument("--other-stage-features-csv", default="")
    parser.add_argument("--wing-model", default="hist_gbdt,random_forest")
    parser.add_argument("--stage-model", default="hist_gbdt,random_forest,template_cost")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--target-auto-precision", type=float, default=0.95)
    parser.add_argument("--landmark-weights", default=str(DEFAULT_LANDMARK_WEIGHTS))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_project_dir = Path(args.stage_project_dir).resolve()

    unified_manifest_rows = load_training_manifest_rows(
        gt100_cleaning_csv=args.gt100_cleaning_csv,
        stage_project_manifest_csv=stage_project_dir / "annotation_project_manifest.csv",
        other_stage_manifest_csv=args.other_stage_manifest_csv or None,
    )
    save_csv(output_dir / "training_manifest.csv", unified_manifest_rows)

    wing_training_manifest = [row for row in unified_manifest_rows if str(row.get("wing_type_label", "")) in WING_LABELS]
    wing_feature_rows = [
        build_wing_feature_row(
            item_id=str(row["id"]),
            image_path=str(row["image_path"]),
            project_dir=stage_project_dir,
            source_group=str(row.get("source_group", "")),
            modality=str(row.get("modality", "")),
            wing_type_label=str(row.get("wing_type_label", "")),
            stage_label=str(row.get("stage_label", "")),
            is_other_stage=int(row.get("is_other_stage", 0)),
            split_group=str(row.get("split_group", "")),
        )
        for row in wing_training_manifest
    ]
    save_csv(output_dir / "wing_shape_features.csv", wing_feature_rows)
    wing_feature_names = wing_feature_numeric_columns(wing_feature_rows)

    wing_train_idx, wing_val_idx = group_train_val_split(wing_feature_rows, args.test_size, args.random_state)
    wing_train_rows = subset_rows(wing_feature_rows, wing_train_idx)
    wing_val_rows = subset_rows(wing_feature_rows, wing_val_idx)

    wing_train_bundle = fit_binary_probability_model(
        wing_train_rows,
        feature_names=wing_feature_names,
        target_label="hindwing",
        model_type=args.wing_model,
        random_state=args.random_state,
    )
    if "components" in wing_train_bundle:
        wing_val_component_probabilities = [apply_binary_probability_model(component, wing_val_rows) for component in wing_train_bundle["components"]]
        wing_truth = np.asarray([str(row.get("wing_type_label", "")) == "hindwing" for row in wing_val_rows], dtype=bool)
        wing_train_bundle["weights"] = tune_binary_ensemble_weights(wing_truth, wing_val_component_probabilities)
    hind_val_prob = apply_binary_probability_model(wing_train_bundle, wing_val_rows)
    wing_thresholds = tune_wing_thresholds(wing_val_rows, hind_val_prob, target_precision=float(args.target_auto_precision))
    wing_val_pred_rows = evaluate_wing_triage(wing_val_rows, hind_val_prob, wing_thresholds)
    save_csv(output_dir / "wing_validation_predictions.csv", wing_val_pred_rows)
    wing_metrics = summarise_wing_metrics(wing_val_pred_rows)

    wing_final_bundle = fit_binary_probability_model(
        wing_feature_rows,
        feature_names=wing_feature_names,
        target_label="hindwing",
        model_type=args.wing_model,
        random_state=args.random_state,
    )
    if "components" in wing_final_bundle:
        wing_final_bundle["weights"] = list(wing_train_bundle.get("weights", wing_final_bundle.get("weights", [])))
    wing_final_bundle["thresholds"] = wing_thresholds
    wing_final_bundle["project_dir"] = str(stage_project_dir)
    save_pickle(output_dir / "wing_model_bundle.pkl", wing_final_bundle)

    raw_stage_rows = [row for row in read_csv_rows(args.stage_features_csv) if str(row.get("true_label", "")).strip() in LABELS]
    stage_rows = [add_shape_features_to_stage_row(row, stage_project_dir) for row in raw_stage_rows]
    save_csv(output_dir / "stage_training_features.csv", stage_rows)
    stage_feature_names = stage_feature_numeric_columns(stage_rows)
    stage_shape_only_features = [name for name in stage_feature_names if name.startswith("shape_") or name in {"mask_threshold", "mask_quality"}]
    shape_only_model_spec = ",".join(parse_model_types(args.stage_model, allow_template_cost=False)) or "hist_gbdt"

    stage_train_idx, stage_val_idx = group_train_val_split(stage_rows, args.test_size, args.random_state)
    stage_train_rows = subset_rows(stage_rows, stage_train_idx)
    stage_val_rows = subset_rows(stage_rows, stage_val_idx)

    stage_train_bundle = fit_multiclass_probability_model(
        stage_train_rows,
        feature_names=stage_feature_names,
        label_key="true_label",
        labels=LABELS,
        model_type=args.stage_model,
        random_state=args.random_state,
    )
    if "components" in stage_train_bundle:
        stage_val_component_probabilities = [apply_multiclass_probability_model(component, stage_val_rows) for component in stage_train_bundle["components"]]
        stage_val_truth = [str(row.get("true_label", "")) for row in stage_val_rows]
        stage_train_bundle["weights"] = tune_multiclass_ensemble_weights(stage_val_truth, stage_val_component_probabilities, LABELS)
    stage_val_prob = apply_multiclass_probability_model(stage_train_bundle, stage_val_rows)

    stage_thresholds = tune_stage_thresholds(
        known_rows=stage_val_rows,
        known_probabilities=stage_val_prob,
        target_precision=args.target_auto_precision,
    )
    stage_val_decisions = stage_decisions_from_thresholds(stage_val_rows, stage_val_prob, stage_thresholds)
    save_csv(output_dir / "stage_validation_predictions.csv", stage_val_decisions)

    auto_stage_rows = [row for row in stage_val_decisions if str(row["decision"]).startswith("auto_hindwing_")]
    all_val_truth = [str(row.get("true_label", "")) for row in stage_val_decisions]
    all_val_pred  = [str(row.get("stage_pred",  "")) for row in stage_val_decisions]
    stage_metrics = {
        "total_val_count": float(len(stage_val_decisions)),
        "auto_accepted_count": float(len(auto_stage_rows)),
        "auto_accepted_coverage": float(len(auto_stage_rows) / max(1, len(stage_val_decisions))),
        "auto_accepted_precision": float(
            np.mean([str(row["stage_pred"]) == str(row["true_label"]) for row in auto_stage_rows])
            if auto_stage_rows else 0.0
        ),
        "full_val_accuracy": float(
            np.mean([p == t for p, t in zip(all_val_pred, all_val_truth)]) if all_val_truth else 0.0
        ),
        "macro_f1_full": float(
            f1_score(all_val_truth, all_val_pred, labels=LABELS, average="macro", zero_division=0)
            if all_val_truth else 0.0
        ),
        "macro_f1_on_accepted": float(
            f1_score(
                [str(row["true_label"]) for row in auto_stage_rows],
                [str(row["stage_pred"])  for row in auto_stage_rows],
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
            if auto_stage_rows else 0.0
        ),
    }

    stage_shape_only_bundle = fit_multiclass_probability_model(
        stage_train_rows,
        feature_names=stage_shape_only_features,
        label_key="true_label",
        labels=LABELS,
        model_type=shape_only_model_spec,
        random_state=args.random_state + 19,
    )
    shape_only_truth = [str(row.get("true_label", "")) for row in stage_val_rows]
    if "components" in stage_shape_only_bundle:
        shape_only_component_probabilities = [apply_multiclass_probability_model(component, stage_val_rows) for component in stage_shape_only_bundle["components"]]
        stage_shape_only_bundle["weights"] = tune_multiclass_ensemble_weights(shape_only_truth, shape_only_component_probabilities, LABELS)
    stage_shape_only_prob = apply_multiclass_probability_model(stage_shape_only_bundle, stage_val_rows)
    shape_only_pred = [LABELS[int(np.argmax(row))] for row in stage_shape_only_prob]

    stage_final_bundle = fit_multiclass_probability_model(
        stage_rows,
        feature_names=stage_feature_names,
        label_key="true_label",
        labels=LABELS,
        model_type=args.stage_model,
        random_state=args.random_state,
    )
    if "components" in stage_final_bundle:
        stage_final_bundle["weights"] = list(stage_train_bundle.get("weights", stage_final_bundle.get("weights", [])))
    stage_final_bundle["thresholds"] = stage_thresholds
    stage_final_bundle["project_dir"] = str(stage_project_dir)
    stage_final_bundle["landmark_weights"] = str(Path(args.landmark_weights).resolve())
    # Note: no mahalanobis bundle — open-set rejection removed
    save_pickle(output_dir / "stage_model_bundle.pkl", stage_final_bundle)

    comparison_rows = comparison_rows_from_embedding()
    comparison_rows.append(
        {
            "system": "mask_shape_only_stage",
            "accuracy": float(accuracy_score(shape_only_truth, shape_only_pred)) if shape_only_truth else 0.0,
            "macro_f1": float(f1_score(shape_only_truth, shape_only_pred, labels=LABELS, average="macro", zero_division=0))
            if shape_only_truth
            else 0.0,
            "notes": "hindwing_only_validation",
        }
    )
    comparison_rows.append(
        {
            "system": "mask_full_stage_val",
            "accuracy": float(stage_metrics["full_val_accuracy"]),
            "macro_f1": float(stage_metrics["macro_f1_full"]),
            "notes": f"auto_coverage={stage_metrics['auto_accepted_coverage']:.3f}",
        }
    )
    save_csv(output_dir / "model_comparison.csv", comparison_rows)

    summary = {
        "output_dir": str(output_dir),
        "wing_train_count": int(len(wing_train_rows)),
        "wing_val_count": int(len(wing_val_rows)),
        "stage_train_count": int(len(stage_train_rows)),
        "stage_val_count": int(len(stage_val_rows)),
        "wing_model_spec": str(args.wing_model),
        "stage_model_spec": str(args.stage_model),
        "wing_ensemble_weights": list(wing_train_bundle.get("weights", [])) if "components" in wing_train_bundle else [1.0],
        "stage_ensemble_weights": list(stage_train_bundle.get("weights", [])) if "components" in stage_train_bundle else [1.0],
        "wing_thresholds": wing_thresholds,
        "wing_metrics": wing_metrics,
        "stage_thresholds": stage_thresholds,
        "stage_metrics": stage_metrics,
    }
    save_json(output_dir / "training_summary.json", summary)
    save_json(
        output_dir / "thresholds.json",
        {
            "wing": wing_thresholds,
            "stage": stage_thresholds,
        },
    )

    print(f"Classifier output: {output_dir}")
    print(f"Wing validation rows: {len(wing_val_rows)}")
    print(f"Stage validation rows: {len(stage_val_rows)}")
    print(f"Wing thresholds: hind>={wing_thresholds['hind_auto_min']:.3f}, fore<={wing_thresholds['fore_auto_max_hind_prob']:.3f}")
    print(
        f"Stage thresholds: prob>={stage_thresholds['stage_prob_min']:.3f}, "
        f"margin>={stage_thresholds['stage_margin_min']:.3f}"
    )
    print(
        f"Stage metrics: full_accuracy={stage_metrics['full_val_accuracy']:.3f}, "
        f"auto_coverage={stage_metrics['auto_accepted_coverage']:.3f}, "
        f"auto_precision={stage_metrics['auto_accepted_precision']:.3f}"
    )


if __name__ == "__main__":
    main()
