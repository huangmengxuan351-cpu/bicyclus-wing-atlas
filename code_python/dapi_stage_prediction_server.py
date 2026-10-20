from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mimetypes
import traceback
from datetime import datetime
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageOps

from dapi_wing_stage_classifier import (
    FAMILY_CONFIGS,
    STAGE_LABELS,
    align_sample_to_family,
    apply_aligned_wing_bundle,
    extract_best_wing_mask,
    extract_wing_mask,
    load_forewing_stage_model_bundle,
    load_gray,
    load_hindwing_stage_model_bundle,
    load_mask,
    load_wing_model_bundle,
    mirror_gray_and_mask,
    select_best_variant,
)
from forewing_stage_classifier_utils import apply_forewing_stage_bundle
from hindwing_stage_classifier_utils import apply_hindwing_stage_bundle


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent / "dapi_stage_prediction_app"
PORTAL_DIR = Path(__file__).resolve().parent / "dapi_stage_portal_app"
ATLAS_DIR = Path(__file__).resolve().parent / "stage_atlas_viewer_app"
REPORT_PATH = ROOT / "output_results" / "v1_v1_1_experiment_report" / "V1_V1_1_final_experiment_report.md"
TYPE_BUNDLE_PATH = ROOT / "output_results" / "stage_upgrade_v1" / "aligned_wing_type_classifier" / "wing_model_bundle.pkl"
HIND_STAGE_BUNDLE_PATH = ROOT / "output_results" / "stage_upgrade_v1" / "hindwing_stage_classifier" / "stage_model_bundle.pkl"
FORE_STAGE_BUNDLE_PATH = ROOT / "output_results" / "stage_upgrade_v1_1" / "forewing_stage_classifier" / "stage_model_bundle.pkl"
DEFAULT_OUTPUT_DIR = ROOT / "output_results" / "dapi_stage_prediction_web_v1"
DEFAULT_STAGE_PROFILE = "deployment_balanced"
PREVIEW_MAX_DIM = 1100
ALIGNMENT_REVIEW_THRESHOLD = 0.62
STAGE_075_GUIDE_MASKS = {
    "forewing": ROOT
    / "output_results"
    / "standard_forewing_templates_stalign_tps_multisample_v2"
    / "stage_0_75"
    / "stage_standard_mask.png",
    "hindwing": ROOT
    / "output_results"
    / "stage_standard_vein_dapi_atlas"
    / "stage_0_75"
    / "recommended"
    / "atlas_wing_mask.png",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_id(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("._-")
    return text or "item"


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def json_response(handler: BaseHTTPRequestHandler, payload: object, status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def bytes_response(handler: BaseHTTPRequestHandler, data: bytes, content_type: str, status: int = 200) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200) -> None:
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_multipart(handler: BaseHTTPRequestHandler) -> dict[str, dict[str, object]]:
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("Expected multipart/form-data request.")
    length = int(handler.headers.get("Content-Length", "0") or 0)
    body = handler.rfile.read(length)
    raw = b"Content-Type: " + content_type.encode("utf-8") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    message = BytesParser(policy=default).parsebytes(raw)
    fields: dict[str, dict[str, object]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        fields[str(name)] = {
            "filename": part.get_filename() or "",
            "content_type": part.get_content_type(),
            "data": part.get_payload(decode=True) or b"",
        }
    return fields


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        raise ValueError("Missing JSON request body.")
    payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON request body must be an object.")
    return payload


def save_preview(path: Path, array: np.ndarray) -> str:
    ensure_dir(path.parent)
    arr = np.asarray(array)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    else:
        arr = np.asarray(np.clip(arr, 0, 255), dtype=np.uint8)
    h, w = arr.shape[:2]
    scale = min(1.0, PREVIEW_MAX_DIM / float(max(h, w)))
    if scale < 0.999:
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        resample = Image.Resampling.NEAREST if np.unique(arr[: min(h, 64), : min(w, 64)]).size <= 3 else Image.Resampling.BILINEAR
        image = Image.fromarray(arr, mode="L").resize(size, resample=resample)
    else:
        image = Image.fromarray(arr, mode="L")
    image.save(path)
    return path.name


def save_full_gray(path: Path, array: np.ndarray) -> str:
    ensure_dir(path.parent)
    arr = np.asarray(array)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    else:
        arr = np.asarray(np.clip(arr, 0, 255), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)
    return path.name


def save_reference_contour(path: Path, mask: np.ndarray) -> str:
    ensure_dir(path.parent)
    mask_bool = np.asarray(mask, dtype=bool)
    height, width = mask_bool.shape
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        points = [(int(point[0][0]), int(point[0][1])) for point in contour]
        draw = ImageDraw.Draw(canvas)
        draw.line(points + [points[0]], fill=(255, 225, 0, 255), width=10, joint="curve")
    scale = min(1.0, PREVIEW_MAX_DIM / float(max(height, width)))
    if scale < 0.999:
        canvas = canvas.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.LANCZOS,
        )
    canvas.save(path)
    return path.name


def mask_overlap_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_bool = np.asarray(left, dtype=bool)
    right_bool = np.asarray(right, dtype=bool)
    intersection = int(np.logical_and(left_bool, right_bool).sum())
    union = int(np.logical_or(left_bool, right_bool).sum())
    total = int(left_bool.sum() + right_bool.sum())
    return {
        "dice": float(2.0 * intersection / total) if total else 0.0,
        "iou": float(intersection / union) if union else 0.0,
    }


def stage_075_guide_mask(family: str) -> np.ndarray:
    path = STAGE_075_GUIDE_MASKS[family]
    if not path.exists():
        raise FileNotFoundError(f"Missing {family} Stage 0.75 guide mask: {path}")
    return load_mask(path)


def normalize_prob_map(prob_map: dict[str, object], labels: list[str]) -> dict[str, float]:
    values = {label: max(0.0, float(prob_map.get(label, 0.0))) for label in labels}
    total = sum(values.values())
    if total <= 0.0:
        return {label: 1.0 / len(labels) for label in labels}
    return {label: values[label] / total for label in labels}


def top_label(prob_map: dict[str, float]) -> tuple[str, float, float]:
    ordered = sorted(prob_map.items(), key=lambda item: item[1], reverse=True)
    if not ordered:
        return "", 0.0, 0.0
    best = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    return best[0], float(best[1]), float(best[1] - second)


def compact_stage_payload(payload: dict[str, object], labels: list[str]) -> dict[str, object]:
    probabilities = normalize_prob_map(dict(payload.get("probabilities", {})), labels)
    stage_pred, stage_conf, stage_margin = top_label(probabilities)
    return {
        "stage_pred": str(payload.get("stage_pred", stage_pred)),
        "stage_conf": float(payload.get("stage_conf", stage_conf)),
        "stage_margin": float(payload.get("stage_margin", stage_margin)),
        "decision": str(payload.get("decision", "")),
        "probabilities": probabilities,
        "stage_model_pred_raw": str(payload.get("stage_model_pred_raw", "")),
        "stage_model_conf_raw": float(payload.get("stage_model_conf_raw", 0.0) or 0.0),
        "ordinal_stage_pred": str(payload.get("ordinal_stage_pred", "")),
        "ordinal_stage_conf": float(payload.get("ordinal_stage_conf", 0.0) or 0.0),
        "fused_stage_pred": str(payload.get("fused_stage_pred", payload.get("stage_pred", stage_pred))),
        "fused_stage_conf": float(payload.get("fused_stage_conf", payload.get("stage_conf", stage_conf)) or 0.0),
        "fused_stage_margin": float(payload.get("fused_stage_margin", payload.get("stage_margin", stage_margin)) or 0.0),
        "stage_profile": str(payload.get("stage_profile", DEFAULT_STAGE_PROFILE)),
        "ordinal_conflict": bool(payload.get("ordinal_conflict", False)),
        "far_reference_override_applied": bool(payload.get("far_reference_override_applied", False)),
        "p_model_pred": str(payload.get("p_model_pred", payload.get("stage_model_pred_raw", ""))),
        "p_model_conf": float(payload.get("p_model_conf", payload.get("stage_model_conf_raw", 0.0)) or 0.0),
        "p_model_probabilities": normalize_prob_map(dict(payload.get("p_model_probabilities", {})), labels),
        "template_stage_probabilities": normalize_prob_map(dict(payload.get("template_stage_probabilities", {})), labels),
        "reference_stage_probabilities": normalize_prob_map(dict(payload.get("reference_stage_probabilities", {})), labels),
        "template_stage_costs": dict(payload.get("template_stage_costs", {})),
        "reference_stage_costs": dict(payload.get("reference_stage_costs", {})),
        "reference_best_distance": float(payload.get("reference_best_distance", 1.0) or 1.0),
        "best_reference_stage": str(payload.get("best_reference_stage", "")),
        "best_reference_id": str(payload.get("best_reference_id", "")),
        "maturity_score": float(payload.get("maturity_score", 0.0) or 0.0),
        "review_reason": str(payload.get("review_reason", "")),
    }


def mask_sanity_score(mask: np.ndarray) -> float:
    area_fraction = float(np.asarray(mask, dtype=bool).mean())
    return float(np.clip(1.0 - abs(area_fraction - 0.32) / 0.30, 0.0, 1.0))


def mask_geometry(mask: np.ndarray) -> dict[str, object]:
    binary = np.asarray(mask, dtype=bool)
    y, x = np.where(binary)
    if len(x) == 0:
        return {
            "centroid": [binary.shape[1] / 2.0, binary.shape[0] / 2.0],
            "bbox": [0, 0, 0, 0],
            "area_pixels": 0,
            "touches_border": False,
        }
    return {
        "centroid": [float(x.mean()), float(y.mean())],
        "bbox": [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
        "area_pixels": int(binary.sum()),
        "touches_border": bool(
            binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any()
        ),
    }


def alignment_diagnostics(
    variant: dict[str, object],
    family: str,
    aligned_mask: np.ndarray,
    stage_payload: dict[str, object],
) -> dict[str, object]:
    family_rows = [row for row in list(variant.get("ranked", [])) if str(getattr(row, "wing_type", "")) == family]
    family_rows.sort(key=lambda row: float(getattr(row, "combined_cost", math.inf)))
    best = family_rows[0] if family_rows else None
    nn_distance = float(getattr(best, "nn_distance", stage_payload.get("reference_best_distance", 1.0))) if best else float(
        stage_payload.get("reference_best_distance", 1.0)
    )
    template_cost = float(getattr(best, "template_cost", math.inf)) if best else math.inf
    reference_overlap = float(np.clip(1.0 - nn_distance, 0.0, 1.0))
    template_similarity = 0.0 if not np.isfinite(template_cost) else float(1.0 / (1.0 + max(0.0, template_cost)))
    mask_sanity = mask_sanity_score(aligned_mask)
    confidence = float(np.clip(0.62 * reference_overlap + 0.23 * template_similarity + 0.15 * mask_sanity, 0.0, 1.0))
    return {
        "confidence": confidence,
        "status": "stable" if confidence >= ALIGNMENT_REVIEW_THRESHOLD else "low_confidence",
        "needs_manual_alignment": bool(confidence < ALIGNMENT_REVIEW_THRESHOLD),
        "threshold": ALIGNMENT_REVIEW_THRESHOLD,
        "reference_overlap": reference_overlap,
        "reference_nn_distance": nn_distance,
        "template_similarity": template_similarity,
        "template_cost": template_cost if np.isfinite(template_cost) else None,
        "mask_sanity": mask_sanity,
        "mask_area_fraction": float(np.asarray(aligned_mask, dtype=bool).mean()),
        "mirrored_variant_selected": bool(variant.get("mirrored", False)),
        "note": "Heuristic alignment confidence, not a biological probability.",
    }


def transform_aligned_pair(
    gray: np.ndarray,
    mask: np.ndarray,
    rotation_deg: float,
    scale: float,
    translate_x: int,
    translate_y: int,
    flip_x: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    gray_array = np.asarray(gray, dtype=np.uint8)
    mask_array = np.asarray(mask, dtype=bool)
    height, width = mask_array.shape
    base_geometry = mask_geometry(mask_array)
    center_x, center_y = [float(value) for value in base_geometry["centroid"]]

    # Match the browser's CSS transform exactly: scale/flip around the wing
    # centroid, rotate clockwise in screen coordinates, then translate in
    # full-resolution canvas pixels.
    angle = math.radians(float(rotation_deg))
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    scale_x = -float(scale) if flip_x else float(scale)
    scale_y = float(scale)
    matrix = np.asarray(
        [
            [
                cos_angle * scale_x,
                -sin_angle * scale_y,
                center_x
                + float(translate_x)
                - cos_angle * scale_x * center_x
                + sin_angle * scale_y * center_y,
            ],
            [
                sin_angle * scale_x,
                cos_angle * scale_y,
                center_y
                + float(translate_y)
                - sin_angle * scale_x * center_x
                - cos_angle * scale_y * center_y,
            ],
        ],
        dtype=np.float64,
    )
    aligned_gray = cv2.warpAffine(
        gray_array,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    aligned_mask = cv2.warpAffine(
        mask_array.astype(np.uint8) * 255,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 127
    output_geometry = mask_geometry(aligned_mask)
    expected_area = max(1.0, float(base_geometry["area_pixels"]) * float(scale) ** 2)
    retained_fraction = float(min(1.0, float(output_geometry["area_pixels"]) / expected_area))
    diagnostics = {
        "transform_center": [center_x, center_y],
        "base_area_pixels": int(base_geometry["area_pixels"]),
        "output_area_pixels": int(output_geometry["area_pixels"]),
        "expected_area_pixels": expected_area,
        "retained_fraction": retained_fraction,
        "output_bbox": output_geometry["bbox"],
        "output_centroid": output_geometry["centroid"],
        "touches_border": output_geometry["touches_border"],
    }
    return aligned_gray, aligned_mask, diagnostics


def model_info_payload() -> dict[str, object]:
    wing_bundle = load_wing_model_bundle()
    hind_bundle = load_hindwing_stage_model_bundle()
    fore_bundle = load_forewing_stage_model_bundle()
    return {
        "app_version": "dapi_stage_atlas_portal_v1",
        "report_path": str(REPORT_PATH),
        "stage_profile": DEFAULT_STAGE_PROFILE,
        "alignment_review_threshold": ALIGNMENT_REVIEW_THRESHOLD,
        "models": {
            "wing_type": {
                "path": str(TYPE_BUNDLE_PATH),
                "exists": TYPE_BUNDLE_PATH.exists(),
                "loaded": wing_bundle is not None,
                "bundle_kind": str(wing_bundle.get("bundle_kind", "")) if wing_bundle else "",
                "best_candidate": str(wing_bundle.get("best_candidate_name", "")) if wing_bundle else "",
            },
            "hindwing_stage": {
                "path": str(HIND_STAGE_BUNDLE_PATH),
                "exists": HIND_STAGE_BUNDLE_PATH.exists(),
                "loaded": hind_bundle is not None,
                "bundle_kind": str(hind_bundle.get("bundle_kind", "")) if hind_bundle else "",
                "best_candidate": str(hind_bundle.get("best_candidate_name", "")) if hind_bundle else "",
            },
            "forewing_stage": {
                "path": str(FORE_STAGE_BUNDLE_PATH),
                "exists": FORE_STAGE_BUNDLE_PATH.exists(),
                "loaded": fore_bundle is not None,
                "bundle_kind": str(fore_bundle.get("bundle_kind", "")) if fore_bundle else "",
                "best_candidate": str(fore_bundle.get("best_candidate_name", "")) if fore_bundle else "",
            },
        },
    }


def choose_variant(gray: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    # select_best_variant evaluates normal and mirrored inputs using template/reference evidence.
    variant = select_best_variant(gray, mask)
    if bool(variant.get("mirrored", False)):
        source_gray, source_mask = mirror_gray_and_mask(gray, mask)
    else:
        source_gray, source_mask = gray, mask
    variant["source_gray"] = source_gray
    variant["source_mask"] = source_mask
    return variant


def fallback_family_probs(variant: dict[str, object]) -> dict[str, float]:
    ranked = list(variant.get("ranked", []))
    totals = {family: 0.0 for family in FAMILY_CONFIGS}
    for row in ranked:
        totals[str(row.wing_type)] += float(row.class_probability)
    return normalize_prob_map(totals, list(FAMILY_CONFIGS.keys()))


def run_prediction(image_path: Path, mask_path: Path | None, run_dir: Path, item_id: str) -> dict[str, object]:
    labels = [str(label) for label in STAGE_LABELS]
    gray = load_gray(image_path)
    if mask_path:
        mask = extract_wing_mask(load_mask(mask_path).astype(np.uint8) * 255)
        mask_source = "provided_mask"
    else:
        preprocess = extract_best_wing_mask(gray)
        mask = np.asarray(preprocess["mask"], dtype=bool)
        mask_source = str(preprocess.get("variant", "extracted_mask"))

    preview_dir = ensure_dir(run_dir / "previews")
    full_dir = ensure_dir(run_dir / "aligned_full")
    save_preview(preview_dir / "gray.png", gray)
    save_preview(preview_dir / "mask.png", mask)
    guide_masks: dict[str, np.ndarray] = {}
    for family in ("forewing", "hindwing"):
        guide_mask = stage_075_guide_mask(family)
        guide_masks[family] = guide_mask
        save_reference_contour(preview_dir / f"{family}_stage_0_75_contour.png", guide_mask)

    wing_bundle = load_wing_model_bundle()
    hind_bundle = load_hindwing_stage_model_bundle()
    fore_bundle = load_forewing_stage_model_bundle()
    if wing_bundle is None or hind_bundle is None or fore_bundle is None:
        raise RuntimeError("One or more final model bundles could not be loaded.")

    variant = choose_variant(gray, mask)
    family_probs_raw: dict[str, float] = {}
    type_evidence: dict[str, object] = {}
    alignments_out: dict[str, object] = {}
    stage_payloads: dict[str, dict[str, object]] = {}
    errors: list[str] = []

    for family in ["forewing", "hindwing"]:
        alignment = dict(variant.get("alignments", {})).get(family)
        if alignment is None:
            errors.append(f"{family} alignment failed")
            family_probs_raw[family] = 0.0
            stage_payloads[family] = {
                "stage_pred": "",
                "stage_conf": 0.0,
                "stage_margin": 0.0,
                "decision": "alignment_failed",
                "probabilities": {label: 1.0 / len(labels) for label in labels},
                "stage_profile": DEFAULT_STAGE_PROFILE,
            }
            continue

        aligned_gray = np.asarray(alignment["aligned_image"], dtype=np.uint8)
        aligned_mask = np.asarray(alignment["aligned_mask"], dtype=bool)
        save_preview(preview_dir / f"{family}_aligned_gray.png", aligned_gray)
        save_preview(preview_dir / f"{family}_aligned_mask.png", aligned_mask)
        save_full_gray(full_dir / f"{family}_aligned_gray.png", aligned_gray)
        save_full_gray(full_dir / f"{family}_aligned_mask.png", aligned_mask)

        _, _, hind_probability = apply_aligned_wing_bundle(
            wing_bundle,
            item_id=f"{item_id}_{family}",
            image_path=image_path,
            mask_path=mask_path,
            gray_override=aligned_gray,
            mask_override=aligned_mask,
            mask_source_override=f"{family}_aligned_mask",
        )
        family_probs_raw[family] = float(1.0 - hind_probability if family == "forewing" else hind_probability)
        type_evidence[family] = {
            "hind_probability_from_type_model": float(hind_probability),
            "family_support_used": float(family_probs_raw[family]),
        }

        if family == "forewing":
            stage_payload = apply_forewing_stage_bundle(
                fore_bundle,
                item_id=f"{item_id}_forewing",
                image_path=image_path,
                mask_path=mask_path,
                stage_profile=DEFAULT_STAGE_PROFILE,
                gray_override=aligned_gray,
                mask_override=aligned_mask,
                mask_source_override="forewing_aligned_mask",
            )
        else:
            stage_payload = apply_hindwing_stage_bundle(
                hind_bundle,
                item_id=f"{item_id}_hindwing",
                image_path=image_path,
                mask_path=mask_path,
                stage_profile=DEFAULT_STAGE_PROFILE,
                gray_override=aligned_gray,
                mask_override=aligned_mask,
                mask_source_override="hindwing_aligned_mask",
            )
        stage_payloads[family] = compact_stage_payload(stage_payload, labels)
        diagnostics = alignment_diagnostics(variant, family, aligned_mask, stage_payloads[family])
        guide_overlap = mask_overlap_metrics(aligned_mask, guide_masks[family])
        diagnostics["guide_stage"] = "0.75"
        diagnostics["guide_dice"] = guide_overlap["dice"]
        diagnostics["guide_iou"] = guide_overlap["iou"]
        alignments_out[family] = {
            "aligned_gray_asset": f"{family}_aligned_gray.png",
            "aligned_mask_asset": f"{family}_aligned_mask.png",
            "reference_contour_asset": f"{family}_stage_0_75_contour.png",
            "reference_contour_stage": "0.75",
            "mask_pixels": int(aligned_mask.sum()),
            "mask_centroid": mask_geometry(aligned_mask)["centroid"],
            "canvas_width": int(aligned_mask.shape[1]),
            "canvas_height": int(aligned_mask.shape[0]),
            "diagnostics": diagnostics,
        }

    family_probabilities = normalize_prob_map(family_probs_raw, ["forewing", "hindwing"])
    if sum(family_probs_raw.values()) <= 0.0:
        family_probabilities = fallback_family_probs(variant)

    joint: dict[str, dict[str, float]] = {}
    for family in ["forewing", "hindwing"]:
        probs = normalize_prob_map(dict(stage_payloads[family].get("probabilities", {})), labels)
        joint[family] = {label: float(family_probabilities[family] * probs[label]) for label in labels}
    total_joint = sum(sum(values.values()) for values in joint.values())
    if total_joint > 0:
        for family in joint:
            for label in joint[family]:
                joint[family][label] = float(joint[family][label] / total_joint)

    overall_stage = {label: float(joint["forewing"][label] + joint["hindwing"][label]) for label in labels}
    wing_pred, wing_conf, wing_margin = top_label(family_probabilities)
    selected_stage_probs = normalize_prob_map(
        dict(stage_payloads.get(wing_pred, {}).get("probabilities", {})),
        labels,
    )
    stage_pred, stage_conf, stage_margin = top_label(selected_stage_probs)
    top_joint_family = ""
    top_joint_stage = ""
    top_joint_conf = -1.0
    for family, stage_map in joint.items():
        for label, value in stage_map.items():
            if value > top_joint_conf:
                top_joint_family = family
                top_joint_stage = label
                top_joint_conf = float(value)

    selected_family = wing_pred or top_joint_family
    selected_alignment = dict(alignments_out.get(selected_family, {})).get("diagnostics", {})
    if selected_family in alignments_out:
        selected_gray = preview_dir / f"{selected_family}_aligned_gray.png"
        selected_mask = preview_dir / f"{selected_family}_aligned_mask.png"
        selected_gray_target = preview_dir / "selected_aligned_gray.png"
        selected_mask_target = preview_dir / "selected_aligned_mask.png"
        if selected_gray.exists():
            selected_gray_target.write_bytes(selected_gray.read_bytes())
        if selected_mask.exists():
            selected_mask_target.write_bytes(selected_mask.read_bytes())

    review_reasons: list[str] = []
    if errors:
        review_reasons.extend(errors)
    if wing_conf < 0.62:
        review_reasons.append("wing_type_probability_below_0.62")
    if wing_margin < 0.06:
        review_reasons.append("wing_type_margin_below_0.06")
    if stage_conf < 0.48:
        review_reasons.append("stage_probability_below_0.48")
    if stage_margin < 0.04:
        review_reasons.append("stage_margin_below_0.04")
    selected_decision = str(stage_payloads.get(selected_family, {}).get("decision", ""))
    if selected_decision and not selected_decision.startswith("auto_"):
        review_reasons.append(f"stage_bundle_decision={selected_decision}")
    if bool(stage_payloads.get(selected_family, {}).get("ordinal_conflict", False)):
        review_reasons.append("ordinal_conflict")
    if bool(dict(selected_alignment).get("needs_manual_alignment", False)):
        review_reasons.append("alignment_confidence_below_threshold")

    final_decision = "needs_review" if review_reasons else "auto_candidate"

    result = {
        "run_id": run_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "image_path": str(image_path),
            "mask_path": str(mask_path) if mask_path else "",
            "mask_source": mask_source,
            "image_shape": [int(gray.shape[0]), int(gray.shape[1])],
            "mask_area_pixels": int(mask.sum()),
        },
        "model_reference": model_info_payload(),
        "preprocessing": {
            "mirrored_variant_selected": bool(variant.get("mirrored", False)),
            "gray_asset": "gray.png",
            "mask_asset": "mask.png",
            "selected_aligned_gray_asset": "selected_aligned_gray.png",
            "selected_aligned_mask_asset": "selected_aligned_mask.png",
            "family_alignments": alignments_out,
            "selected_family_for_stage": selected_family,
            "selected_alignment": selected_alignment,
            "manual_alignment_available": True,
        },
        "family_probabilities": family_probabilities,
        "conditional_stage_probabilities": {
            family: dict(stage_payloads[family].get("probabilities", {})) for family in ["forewing", "hindwing"]
        },
        "overall_stage_probabilities": overall_stage,
        "joint_probabilities": joint,
        "type_evidence": type_evidence,
        "stage_payloads": stage_payloads,
        "summary": {
            "wing_type_pred": wing_pred,
            "wing_type_conf": wing_conf,
            "wing_type_margin": wing_margin,
            "stage_pred": stage_pred,
            "stage_conf": stage_conf,
            "stage_margin": stage_margin,
            "joint_pred": f"{top_joint_family}_{top_joint_stage}",
            "joint_conf": top_joint_conf,
            "stage_profile": DEFAULT_STAGE_PROFILE,
            "stage_family": selected_family,
            "alignment_confidence": float(dict(selected_alignment).get("confidence", 0.0) or 0.0),
            "decision": final_decision,
            "review_reason": "; ".join(review_reasons),
        },
    }
    return result


def run_manual_realign(
    current: dict[str, object],
    run_dir: Path,
    family: str,
    rotation_deg: float,
    scale: float,
    translate_x: int,
    translate_y: int,
    flip_x: bool,
) -> dict[str, object]:
    if family not in {"forewing", "hindwing"}:
        raise ValueError("family must be forewing or hindwing")
    rotation_deg = float(np.clip(rotation_deg, -180.0, 180.0))
    scale = float(np.clip(scale, 0.50, 1.50))
    translate_x = int(np.clip(translate_x, -512, 512))
    translate_y = int(np.clip(translate_y, -512, 512))

    full_dir = run_dir / "aligned_full"
    base_gray_path = full_dir / f"{family}_aligned_gray.png"
    base_mask_path = full_dir / f"{family}_aligned_mask.png"
    if not base_gray_path.exists() or not base_mask_path.exists():
        raise FileNotFoundError("Full-resolution automatic alignment is unavailable for this run.")
    base_gray = load_gray(base_gray_path)
    base_mask = load_mask(base_mask_path)
    aligned_gray, aligned_mask, transform_diagnostics = transform_aligned_pair(
        base_gray,
        base_mask,
        rotation_deg=rotation_deg,
        scale=scale,
        translate_x=translate_x,
        translate_y=translate_y,
        flip_x=flip_x,
    )
    if int(aligned_mask.sum()) < 64:
        raise ValueError("Manual transform moved the wing outside the canvas.")
    if float(transform_diagnostics["retained_fraction"]) < 0.97:
        raise ValueError(
            "Manual transform would crop the wing. Reduce translation/scale or adjust rotation before applying."
        )

    preview_dir = ensure_dir(run_dir / "previews")
    save_preview(preview_dir / "manual_aligned_gray.png", aligned_gray)
    save_preview(preview_dir / "manual_aligned_mask.png", aligned_mask)
    save_full_gray(full_dir / "manual_aligned_gray.png", aligned_gray)
    save_full_gray(full_dir / "manual_aligned_mask.png", aligned_mask)

    wing_bundle = load_wing_model_bundle()
    fore_bundle = load_forewing_stage_model_bundle()
    hind_bundle = load_hindwing_stage_model_bundle()
    if wing_bundle is None or fore_bundle is None or hind_bundle is None:
        raise RuntimeError("One or more final model bundles could not be loaded.")
    input_payload = dict(current.get("input", {}))
    image_path = Path(str(input_payload.get("image_path", "")))
    _, _, hind_probability = apply_aligned_wing_bundle(
        wing_bundle,
        item_id=f"{run_dir.name}_{family}_manual",
        image_path=image_path,
        mask_path=None,
        gray_override=aligned_gray,
        mask_override=aligned_mask,
        mask_source_override="manual_adjusted_aligned_mask",
    )
    family_probabilities = normalize_prob_map(
        {"forewing": 1.0 - float(hind_probability), "hindwing": float(hind_probability)},
        ["forewing", "hindwing"],
    )
    kwargs = {
        "item_id": f"{run_dir.name}_{family}_manual_stage",
        "image_path": image_path,
        "mask_path": None,
        "stage_profile": DEFAULT_STAGE_PROFILE,
        "gray_override": aligned_gray,
        "mask_override": aligned_mask,
        "mask_source_override": "manual_adjusted_aligned_mask",
    }
    raw_stage = apply_forewing_stage_bundle(fore_bundle, **kwargs) if family == "forewing" else apply_hindwing_stage_bundle(
        hind_bundle, **kwargs
    )
    labels = [str(label) for label in STAGE_LABELS]
    stage_payload = compact_stage_payload(raw_stage, labels)
    stage_probabilities = normalize_prob_map(dict(stage_payload.get("probabilities", {})), labels)
    wing_pred, wing_conf, wing_margin = top_label(family_probabilities)
    stage_pred, stage_conf, stage_margin = top_label(stage_probabilities)

    reference_distance = float(stage_payload.get("reference_best_distance", 1.0) or 1.0)
    # Stage feature distance is not bounded to [0, 1], so map it monotonically.
    reference_overlap = float(np.clip(1.0 - reference_distance, 0.0, 1.0))
    template_cost_values = [
        float(value)
        for value in dict(stage_payload.get("template_stage_costs", {})).values()
        if value is not None and np.isfinite(float(value))
    ]
    template_cost = min(template_cost_values) if template_cost_values else math.inf
    template_similarity = 0.0 if not np.isfinite(template_cost) else float(1.0 / (1.0 + max(0.0, template_cost)))
    mask_sanity = mask_sanity_score(aligned_mask)
    alignment_confidence = float(
        np.clip(0.62 * reference_overlap + 0.23 * template_similarity + 0.15 * mask_sanity, 0.0, 1.0)
    )
    guide_mask = stage_075_guide_mask(family)
    guide_overlap = mask_overlap_metrics(aligned_mask, guide_mask)
    alignment_payload = {
        "confidence": alignment_confidence,
        "status": "stable" if alignment_confidence >= ALIGNMENT_REVIEW_THRESHOLD else "low_confidence",
        "needs_manual_alignment": bool(alignment_confidence < ALIGNMENT_REVIEW_THRESHOLD),
        "threshold": ALIGNMENT_REVIEW_THRESHOLD,
        "reference_overlap": reference_overlap,
        "reference_nn_distance": reference_distance,
        "template_similarity": template_similarity,
        "template_cost": template_cost if np.isfinite(template_cost) else None,
        "mask_sanity": mask_sanity,
        "mask_area_fraction": float(aligned_mask.mean()),
        "guide_stage": "0.75",
        "guide_dice": guide_overlap["dice"],
        "guide_iou": guide_overlap["iou"],
        "user_adjusted": True,
        "transform_retained_fraction": float(transform_diagnostics["retained_fraction"]),
        "transform_touches_border": bool(transform_diagnostics["touches_border"]),
        "confidence_method": "same_weighting_v1_as_auto_alignment",
        "note": "Recomputed after user alignment adjustment.",
    }

    conditional = dict(current.get("conditional_stage_probabilities", {}))
    conditional[family] = stage_probabilities
    stage_payloads = dict(current.get("stage_payloads", {}))
    stage_payloads[family] = stage_payload
    joint: dict[str, dict[str, float]] = {}
    for candidate_family in ["forewing", "hindwing"]:
        candidate_probs = normalize_prob_map(dict(conditional.get(candidate_family, {})), labels)
        joint[candidate_family] = {
            label: float(family_probabilities[candidate_family] * candidate_probs[label]) for label in labels
        }
    overall_stage = {label: float(joint["forewing"][label] + joint["hindwing"][label]) for label in labels}

    review_reasons: list[str] = []
    if wing_pred != family:
        review_reasons.append("manual_family_conflicts_with_type_model")
    if alignment_confidence < ALIGNMENT_REVIEW_THRESHOLD:
        review_reasons.append("alignment_confidence_below_threshold")
    if stage_conf < 0.48:
        review_reasons.append("stage_probability_below_0.48")
    if stage_margin < 0.04:
        review_reasons.append("stage_margin_below_0.04")
    if bool(stage_payload.get("ordinal_conflict", False)):
        review_reasons.append("ordinal_conflict")
    stage_decision = str(stage_payload.get("decision", ""))
    if stage_decision and not stage_decision.startswith("auto_"):
        review_reasons.append(f"stage_bundle_decision={stage_decision}")

    result = dict(current)
    result["timestamp"] = datetime.now().isoformat(timespec="seconds")
    result["family_probabilities"] = family_probabilities
    result["conditional_stage_probabilities"] = conditional
    result["overall_stage_probabilities"] = overall_stage
    result["joint_probabilities"] = joint
    result["stage_payloads"] = stage_payloads
    preprocessing = dict(result.get("preprocessing", {}))
    preprocessing.update(
        {
            "selected_family_for_stage": family,
            "selected_aligned_gray_asset": "manual_aligned_gray.png",
            "selected_aligned_mask_asset": "manual_aligned_mask.png",
            "selected_alignment": alignment_payload,
            "manual_alignment": {
                "family": family,
                "rotation_deg": rotation_deg,
                "scale": scale,
                "translate_x": translate_x,
                "translate_y": translate_y,
                "flip_x": flip_x,
                "transform_diagnostics": transform_diagnostics,
            },
        }
    )
    result["preprocessing"] = preprocessing
    result["summary"] = {
        "wing_type_pred": wing_pred,
        "wing_type_conf": wing_conf,
        "wing_type_margin": wing_margin,
        "stage_pred": stage_pred,
        "stage_conf": stage_conf,
        "stage_margin": stage_margin,
        "stage_family": family,
        "stage_profile": DEFAULT_STAGE_PROFILE,
        "alignment_confidence": alignment_confidence,
        "decision": "needs_review" if review_reasons else "manual_candidate",
        "review_reason": "; ".join(review_reasons),
    }
    result["revision"] = "manual_alignment"
    return result


def append_prediction_log(output_dir: Path, payload: dict[str, object], original_filename: str) -> None:
    predictions_dir = ensure_dir(output_dir / "predictions")
    log_path = predictions_dir / "prediction_log.csv"
    summary = dict(payload.get("summary", {}))
    family = dict(payload.get("family_probabilities", {}))
    stages = dict(payload.get("overall_stage_probabilities", {}))
    row = {
        "timestamp": payload.get("timestamp", ""),
        "run_id": payload.get("run_id", ""),
        "original_filename": original_filename,
        "decision": summary.get("decision", ""),
        "review_reason": summary.get("review_reason", ""),
        "wing_type_pred": summary.get("wing_type_pred", ""),
        "wing_type_conf": summary.get("wing_type_conf", 0.0),
        "stage_pred": summary.get("stage_pred", ""),
        "stage_conf": summary.get("stage_conf", 0.0),
        "forewing_probability": family.get("forewing", 0.0),
        "hindwing_probability": family.get("hindwing", 0.0),
        "stage_0_5_probability": stages.get("0.5", 0.0),
        "stage_0_75_probability": stages.get("0.75", 0.0),
        "stage_1_0_probability": stages.get("1.0", 0.0),
    }
    exists = log_path.exists()
    with log_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class StagePredictionState:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = ensure_dir(output_dir)
        ensure_dir(self.output_dir / "uploads")
        ensure_dir(self.output_dir / "predictions")

    def history(self, limit: int = 50) -> list[dict[str, str]]:
        log_path = self.output_dir / "predictions" / "prediction_log.csv"
        if not log_path.exists():
            return []
        with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return rows[-limit:][::-1]

    def predict_from_upload(self, fields: dict[str, dict[str, object]]) -> dict[str, object]:
        image_field = fields.get("image")
        if not image_field or not image_field.get("data"):
            raise ValueError("Missing uploaded image field named 'image'.")
        original_name = safe_id(str(image_field.get("filename", "uploaded.png")))
        digest = hashlib.sha1(bytes(image_field["data"])).hexdigest()[:10]  # type: ignore[arg-type]
        run_id = safe_id(f"{now_id()}_{Path(original_name).stem}_{digest}")
        run_dir = ensure_dir(self.output_dir / "uploads" / run_id)
        image_ext = Path(original_name).suffix.lower() or ".png"
        image_path = run_dir / f"input{image_ext}"
        image_path.write_bytes(bytes(image_field["data"]))  # type: ignore[arg-type]

        mask_path: Path | None = None
        mask_field = fields.get("mask")
        if mask_field and mask_field.get("data"):
            mask_name = safe_id(str(mask_field.get("filename", "mask.png")))
            mask_ext = Path(mask_name).suffix.lower() or ".png"
            mask_path = run_dir / f"mask{mask_ext}"
            mask_path.write_bytes(bytes(mask_field["data"]))  # type: ignore[arg-type]

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Force PIL to validate that the upload is readable before running the expensive pipeline.
        with Image.open(image_path) as check:
            check.verify()

        result = run_prediction(image_path=image_path, mask_path=mask_path, run_dir=run_dir, item_id=run_id)
        prediction_path = self.output_dir / "predictions" / f"{run_id}.json"
        prediction_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        append_prediction_log(self.output_dir, result, original_name)
        return result

    def result_path(self, run_id: str) -> Path:
        return self.output_dir / "predictions" / f"{safe_id(run_id)}.json"

    def realign(self, payload: dict[str, object]) -> dict[str, object]:
        run_id = safe_id(str(payload.get("run_id", "")))
        if not run_id:
            raise ValueError("run_id is required")
        result_path = self.result_path(run_id)
        if not result_path.exists():
            raise FileNotFoundError("Prediction result not found.")
        current = json.loads(result_path.read_text(encoding="utf-8"))
        auto_path = result_path.with_name(f"{run_id}_auto.json")
        if not auto_path.exists():
            auto_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        result = run_manual_realign(
            current=current,
            run_dir=self.output_dir / "uploads" / run_id,
            family=str(payload.get("family", "")),
            rotation_deg=float(payload.get("rotation_deg", 0.0) or 0.0),
            scale=float(payload.get("scale", 1.0) or 1.0),
            translate_x=int(float(payload.get("translate_x", 0) or 0)),
            translate_y=int(float(payload.get("translate_y", 0) or 0)),
            flip_x=bool(payload.get("flip_x", False)),
        )
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        append_prediction_log(self.output_dir, result, f"{run_id} [manual]")
        return result

    def asset_path(self, run_id: str, asset: str) -> Path:
        return self.output_dir / "uploads" / safe_id(run_id) / "previews" / safe_id(asset)


def make_handler(state: StagePredictionState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def serve_static_file(self, root: Path, relative_path: str) -> None:
            static_path = (root / relative_path.lstrip("/")).resolve()
            static_root = root.resolve()
            if static_path != static_root and static_root not in static_path.parents:
                text_response(self, "Forbidden", HTTPStatus.FORBIDDEN)
                return
            if not static_path.exists() or not static_path.is_file():
                text_response(self, "Not found", HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(str(static_path))[0] or "application/octet-stream"
            bytes_response(self, static_path.read_bytes(), content_type)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == "/api/model-info":
                json_response(self, model_info_payload())
                return
            if path == "/api/history":
                json_response(self, {"items": state.history()})
                return
            if path.startswith("/api/result/"):
                run_id = path[len("/api/result/") :]
                result_path = state.result_path(run_id)
                if not result_path.exists():
                    text_response(self, "Result not found", HTTPStatus.NOT_FOUND)
                    return
                bytes_response(self, result_path.read_bytes(), "application/json; charset=utf-8")
                return
            if path.startswith("/api/file/"):
                parts = path.split("/")
                if len(parts) < 5:
                    text_response(self, "Bad file route", HTTPStatus.BAD_REQUEST)
                    return
                asset_path = state.asset_path(parts[3], parts[4])
                if not asset_path.exists():
                    text_response(self, "Asset not found", HTTPStatus.NOT_FOUND)
                    return
                bytes_response(self, asset_path.read_bytes(), mimetypes.guess_type(str(asset_path))[0] or "image/png")
                return

            if path == "/":
                self.serve_static_file(PORTAL_DIR, "index.html")
                return
            if path in {"/result", "/result/"}:
                self.serve_static_file(PORTAL_DIR, "result.html")
                return
            if path.startswith("/portal/"):
                self.serve_static_file(PORTAL_DIR, path[len("/portal/") :])
                return
            if path in {"/predict", "/predict/"}:
                self.serve_static_file(APP_DIR, "index.html")
                return
            if path.startswith("/predict/"):
                self.serve_static_file(APP_DIR, path[len("/predict/") :])
                return
            if path in {"/atlas", "/atlas/"}:
                self.serve_static_file(ATLAS_DIR, "index.html")
                return
            if path.startswith("/atlas/"):
                self.serve_static_file(ATLAS_DIR, path[len("/atlas/") :])
                return
            text_response(self, "Not found", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path not in {"/api/predict", "/api/re-align"}:
                json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            request_payload: dict[str, object] | None = None
            try:
                if path == "/api/predict":
                    fields = read_multipart(self)
                    result = state.predict_from_upload(fields)
                else:
                    request_payload = read_json_body(self)
                    result = state.realign(request_payload)
                json_response(self, result)
            except Exception as exc:
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] {path} failed: {exc}; "
                    f"payload={request_payload or {}}",
                    flush=True,
                )
                json_response(
                    self,
                    {
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    },
                    HTTPStatus.BAD_REQUEST,
                )

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAPI fore/hind wing and stage prediction web server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8820)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = StagePredictionState(Path(args.output_dir))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"DAPI stage prediction app: http://{args.host}:{args.port}/")
    print(f"Output: {state.output_dir}")
    print(f"Report: {REPORT_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
