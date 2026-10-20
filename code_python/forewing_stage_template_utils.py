from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stage_template_geometry import (
    compute_notch_features,
    convex_hull_points,
    extract_contour_points,
    from_canonical,
    to_canonical,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "output_results" / "dapi_forewing_standardized_manual_exported"
DEFAULT_PROJECT_DIR = ROOT / "output_results" / "forewing_stage_template_project"
DEFAULT_STANDARD_TEMPLATE_DIR = ROOT / "output_results" / "standard_forewing_templates"
LABELS = ["0.5", "0.75", "1.0"]


def stage_dir_name(label: str) -> str:
    return f"stage_{label.replace('.', '_')}"


def label_from_stage_dir(stage_dir: str) -> str:
    return stage_dir.replace("stage_", "").replace("_", ".")


def project_paths(project_dir: str | Path) -> dict[str, Path]:
    project_dir = Path(project_dir)
    return {
        "project_dir": project_dir,
        "manifest_csv": project_dir / "forewing_template_manifest.csv",
        "pseudo_landmarks_csv": project_dir / "forewing_pseudo_landmarks.csv",
        "template_system_dir": project_dir / "template_system",
        "base_template_json": project_dir / "template_system" / "base_template.json",
        "stage_template_0_5_json": project_dir / "template_system" / "stage_template_0_5.json",
        "stage_template_0_75_json": project_dir / "template_system" / "stage_template_0_75.json",
        "stage_template_1_0_json": project_dir / "template_system" / "stage_template_1_0.json",
        "template_qc_csv": project_dir / "template_system" / "template_qc.csv",
        "template_sample_alignment_csv": project_dir / "template_system" / "template_sample_alignment.csv",
        "standard_template_dir": DEFAULT_STANDARD_TEMPLATE_DIR,
    }


def read_gray(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def read_mask(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def base_band_points(contour: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    if len(contour) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    x_values = contour[:, 0]
    threshold = np.quantile(x_values, 1.0 - fraction)
    points = contour[x_values >= threshold]
    if len(points) < 8:
        threshold = np.quantile(x_values, 0.80)
        points = contour[x_values >= threshold]
    return points.astype(np.float64)


def _temporary_landmarks(
    base_dorsal: np.ndarray,
    base_ventral: np.ndarray,
    distal_tip: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            base_dorsal.astype(np.float64),
            base_ventral.astype(np.float64),
            distal_tip.astype(np.float64),
            base_ventral.astype(np.float64),
        ],
        dtype=np.float64,
    )


def infer_posterior_notch(
    contour: np.ndarray,
    base_dorsal: np.ndarray,
    base_ventral: np.ndarray,
    distal_tip: np.ndarray,
) -> np.ndarray:
    temp_landmarks = _temporary_landmarks(base_dorsal, base_ventral, distal_tip)
    contour_canonical = to_canonical(contour, temp_landmarks)
    hull_canonical = convex_hull_points(contour_canonical)
    notch_info = compute_notch_features(contour_canonical, hull_canonical)
    notch_canonical = np.asarray(notch_info["notch_point"], dtype=np.float64)

    base_center = (base_dorsal + base_ventral) / 2.0
    canonical_candidates = contour_canonical[
        (contour_canonical[:, 1] >= 0.03)
        & (contour_canonical[:, 0] >= -0.15)
        & (contour_canonical[:, 0] <= 0.35)
    ]
    if len(canonical_candidates):
        center = canonical_candidates.mean(axis=0)
        score = 2.0 * canonical_candidates[:, 1] - np.abs(canonical_candidates[:, 0] - center[0])
        fallback_canonical = canonical_candidates[int(np.argmax(score))]
    else:
        fallback_canonical = np.asarray([0.05, 0.20], dtype=np.float64)

    # Reject degenerate notch estimates that collapse onto the ventral base or land outside
    # the expected posterior sector.
    notch_distance = float(np.linalg.norm(notch_canonical - to_canonical(base_ventral.reshape(1, 2), temp_landmarks)[0]))
    if (
        notch_distance < 0.08
        or notch_canonical[1] < 0.02
        or notch_canonical[0] < -0.25
        or notch_canonical[0] > 0.45
    ):
        notch_canonical = fallback_canonical

    notch_image = from_canonical(notch_canonical.reshape(1, 2), temp_landmarks)[0]
    distances = np.linalg.norm(contour - notch_image.reshape(1, 2), axis=1)
    return contour[int(np.argmin(distances))].astype(np.float64)


def infer_forewing_landmarks(mask: np.ndarray) -> dict[str, dict[str, float]]:
    contour = extract_contour_points(mask, n_points=512)
    if len(contour) < 16:
        raise ValueError("Mask contour too small to infer forewing landmarks.")

    base_points = base_band_points(contour, fraction=0.10)
    if len(base_points) == 0:
        raise ValueError("Unable to locate forewing base band from contour.")

    base_dorsal = base_points[int(np.argmin(base_points[:, 1]))]
    base_ventral = base_points[int(np.argmax(base_points[:, 1]))]
    base_center = (base_dorsal + base_ventral) / 2.0

    left_candidates = contour[contour[:, 0] <= np.quantile(contour[:, 0], 0.08)]
    if len(left_candidates) == 0:
        left_candidates = contour
    distances = np.linalg.norm(left_candidates - base_center.reshape(1, 2), axis=1)
    distal_tip = left_candidates[int(np.argmax(distances))]
    posterior_notch = infer_posterior_notch(contour, base_dorsal, base_ventral, distal_tip)

    return {
        "base_dorsal": {"x": float(base_dorsal[0]), "y": float(base_dorsal[1])},
        "base_ventral": {"x": float(base_ventral[0]), "y": float(base_ventral[1])},
        "distal_tip": {"x": float(distal_tip[0]), "y": float(distal_tip[1])},
        "posterior_notch": {"x": float(posterior_notch[0]), "y": float(posterior_notch[1])},
    }


def load_forewing_rows(source_root: str | Path = DEFAULT_INPUT_ROOT) -> list[dict[str, object]]:
    source_root = Path(source_root)
    rows: list[dict[str, object]] = []
    for label in LABELS:
        stage_dir = source_root / stage_dir_name(label)
        if not stage_dir.exists():
            continue
        for sample_dir in sorted(path for path in stage_dir.iterdir() if path.is_dir()):
            image_path = sample_dir / "standardized_dapi_gray.png"
            mask_path = sample_dir / "standardized_wing_mask.png"
            if not image_path.exists() or not mask_path.exists():
                continue
            mask = read_mask(mask_path)
            landmarks = infer_forewing_landmarks(mask)
            rows.append(
                {
                    "id": sample_dir.name,
                    "label": label,
                    "stage_dir": stage_dir.name,
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "landmarks": landmarks,
                }
            )
    if not rows:
        raise ValueError(f"No standardized forewing rows found under {source_root}")
    return rows
