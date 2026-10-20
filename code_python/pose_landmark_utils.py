from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data_images" / "labeled" / "standardized_hindwing_reference_plus_test_hind_image_only"
DEFAULT_LABELS_CSV = ROOT / "embedding_results" / "embedding_only" / "labels.csv"
DEFAULT_PROJECT_DIR = ROOT / "output_results" / "landmark_pose_project"
DEFAULT_STANDARD_TEMPLATE_DIR = ROOT / "output_results" / "standard_hindwing_templates"
LABELS = ["0.5", "0.75", "1.0"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LANDMARK_NAMES = ["base_dorsal", "base_ventral", "distal_tip", "posterior_notch"]
LANDMARK_TABLE_FIELDNAMES = [
    "id",
    "image_path",
    "label",
    "confidence",
    "base_dorsal_x",
    "base_dorsal_y",
    "base_ventral_x",
    "base_ventral_y",
    "distal_tip_x",
    "distal_tip_y",
    "posterior_notch_x",
    "posterior_notch_y",
]

KEYPOINT_SCHEMA = [
    {
        "name": "base_dorsal",
        "short_label": "P1",
        "display_name": "Base Dorsal",
        "description": "翅膀与身体连接区域的上端点，落在真实轮廓上。",
        "color": "#ff5d73",
    },
    {
        "name": "base_ventral",
        "short_label": "P2",
        "display_name": "Base Ventral",
        "description": "翅膀与身体连接区域的下端点，和 P1 构成基部轴。",
        "color": "#ffb84d",
    },
    {
        "name": "distal_tip",
        "short_label": "P3",
        "display_name": "Distal Tip",
        "description": "距基部中心最远的外缘点，优先取最远尖端。",
        "color": "#4cc38a",
    },
    {
        "name": "posterior_notch",
        "short_label": "P4",
        "display_name": "Posterior Notch",
        "description": "后缘最明显的内凹点；若不明显，取后缘最大凹曲率位置。",
        "color": "#57a0ff",
    },
]


def parse_label_from_id(item_id: str) -> str:
    text = item_id.lower()
    if re.search(r"(^|_)0_75(_|$)", text) or re.search(r"(^|-)0\.75($|-|_)", text):
        return "0.75"
    if re.search(r"(^|_)0_5(_|$)", text) or re.search(r"(^|-)0\.5($|-|_)", text):
        return "0.5"
    if re.search(r"(^|_)1_0(_|$)", text) or re.search(r"(^|-)1\.0($|-|_)", text):
        return "1.0"
    return ""


def project_paths(project_dir: str | Path) -> dict[str, Path]:
    project_dir = Path(project_dir)
    return {
        "project_dir": project_dir,
        "manifest_csv": project_dir / "annotation_project_manifest.csv",
        "schema_json": project_dir / "keypoint_schema.json",
        "status_csv": project_dir / "annotation_status.csv",
        "annotation_export_csv": project_dir / "annotation_export.csv",
        "annotations_dir": project_dir / "annotations",
        "yolo_dataset_dir": project_dir / "yolo_pose_dataset",
        "registered_dir": project_dir / "registered_images",
        "registered_manifest_csv": project_dir / "registered_manifest.csv",
        "registered_labels_csv": project_dir / "registered_labels.csv",
        "registration_qc_csv": project_dir / "registration_qc.csv",
        "template_json": project_dir / "registration_template.json",
        "qc_dir": project_dir / "registration_qc",
        "template_system_dir": project_dir / "template_system",
        "base_template_json": project_dir / "template_system" / "base_template.json",
        "stage_template_0_5_json": project_dir / "template_system" / "stage_template_0_5.json",
        "stage_template_0_75_json": project_dir / "template_system" / "stage_template_0_75.json",
        "stage_template_1_0_json": project_dir / "template_system" / "stage_template_1_0.json",
        "template_qc_csv": project_dir / "template_system" / "template_qc.csv",
        "stage_registered_root": project_dir / "stage_registered",
        "stage_registration_qc_dir": project_dir / "stage_registration_qc",
        "stage_registration_scores_csv": project_dir / "stage_registration_scores.csv",
        "stage_registration_features_csv": project_dir / "stage_registration_features.csv",
        "standard_template_dir": DEFAULT_STANDARD_TEMPLATE_DIR,
        "standard_template_base_png": DEFAULT_STANDARD_TEMPLATE_DIR / "base_mother.png",
        "standard_template_0_5_png": DEFAULT_STANDARD_TEMPLATE_DIR / "standard_0_5.png",
        "standard_template_0_75_png": DEFAULT_STANDARD_TEMPLATE_DIR / "standard_0_75.png",
        "standard_template_1_0_png": DEFAULT_STANDARD_TEMPLATE_DIR / "standard_1_0.png",
        "standard_template_base_mask_png": DEFAULT_STANDARD_TEMPLATE_DIR / "base_mother_mask.png",
        "standard_template_0_5_mask_png": DEFAULT_STANDARD_TEMPLATE_DIR / "standard_0_5_mask.png",
        "standard_template_0_75_mask_png": DEFAULT_STANDARD_TEMPLATE_DIR / "standard_0_75_mask.png",
        "standard_template_1_0_mask_png": DEFAULT_STANDARD_TEMPLATE_DIR / "standard_1_0_mask.png",
        "stalign_registered_root": project_dir / "stalign_registered",
        "stalign_qc_dir": project_dir / "stalign_qc",
        "stalign_deformation_root": project_dir / "stalign_deformation_fields",
        "stalign_registration_scores_csv": project_dir / "stalign_registration_scores.csv",
        "stalign_registration_features_csv": project_dir / "stalign_registration_features.csv",
        "stalign_run_config_json": project_dir / "stalign_run_config.json",
        "stalign_reference_selection_csv": project_dir / "stalign_reference_selection.csv",
        "stalign_comparison_report_md": project_dir / "stalign_comparison_report.md",
        "stalign_comparison_rows_csv": project_dir / "stalign_comparison_rows.csv",
        "stalign_ablation_summary_csv": project_dir / "stalign_ablation_summary.csv",
        "stalign_visualization_dir": project_dir / "stalign_visualizations",
    }


def ensure_project_dirs(project_dir: str | Path) -> dict[str, Path]:
    paths = project_paths(project_dir)
    paths["project_dir"].mkdir(parents=True, exist_ok=True)
    paths["annotations_dir"].mkdir(parents=True, exist_ok=True)
    paths["qc_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: (value or "") for key, value in row.items()} for row in reader]


def write_csv_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def load_label_map(labels_csv: str | Path = DEFAULT_LABELS_CSV) -> dict[str, str]:
    rows = read_csv_rows(labels_csv)
    label_map: dict[str, str] = {}
    for row in rows:
        item_id = str(row.get("id", "")).strip()
        label = str(row.get("label", "")).strip()
        if item_id:
            label_map[item_id] = label
    return label_map


def load_labeled_image_table(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    labels_csv: str | Path = DEFAULT_LABELS_CSV,
) -> list[dict[str, str]]:
    data_dir = Path(data_dir)
    image_paths = sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    label_map = load_label_map(labels_csv)
    rows: list[dict[str, str]] = []
    for path in image_paths:
        item_id = path.stem
        label = label_map.get(item_id, "")
        if label not in LABELS:
            label = parse_label_from_id(item_id)
        if label in LABELS:
            rows.append(
                {
                    "id": item_id,
                    "label": label,
                    "image_path": str(path.resolve()),
                }
            )
    if not rows:
        raise ValueError("No labeled images found after matching labels.")
    return rows


def write_keypoint_schema(schema_path: str | Path) -> Path:
    schema_path = Path(schema_path)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "Hindwing Landmark Annotation",
        "version": 1,
        "keypoints": KEYPOINT_SCHEMA,
    }
    schema_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return schema_path


def load_project_manifest(manifest_csv: str | Path) -> list[dict[str, str]]:
    rows = read_csv_rows(manifest_csv)
    required = {"id", "label", "image_path"}
    if not rows:
        return []
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return rows


def annotation_json_path(annotations_dir: str | Path, item_id: str) -> Path:
    return Path(annotations_dir) / f"{item_id}.json"


def load_annotations(annotations_dir: str | Path) -> dict[str, dict]:
    annotations_dir = Path(annotations_dir)
    annotations: dict[str, dict] = {}
    if not annotations_dir.exists():
        return annotations
    for path in sorted(annotations_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        item_id = str(payload.get("id") or path.stem)
        annotations[item_id] = payload
    return annotations


def annotation_points_complete(payload: dict) -> bool:
    points = payload.get("points", {})
    for spec in KEYPOINT_SCHEMA:
        point = points.get(spec["name"])
        if not point:
            return False
        if point.get("x") is None or point.get("y") is None:
            return False
    return True


def annotation_rows(manifest: list[dict[str, str]], annotations: dict[str, dict]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in manifest:
        payload = annotations.get(str(row["id"]), {})
        status = "complete" if annotation_points_complete(payload) else "pending"
        updated_at = str(payload.get("updated_at", ""))
        rows.append(
            {
                "id": str(row["id"]),
                "label": str(row["label"]),
                "image_path": str(row["image_path"]),
                "annotation_status": status,
                "updated_at": updated_at,
            }
        )
    return rows


def save_annotation_status(project_dir: str | Path) -> Path:
    paths = ensure_project_dirs(project_dir)
    manifest = load_project_manifest(paths["manifest_csv"])
    annotations = load_annotations(paths["annotations_dir"])
    status = annotation_rows(manifest, annotations)
    write_csv_rows(paths["status_csv"], ["id", "label", "image_path", "annotation_status", "updated_at"], status)
    return paths["status_csv"]


def stalign_variant_paths(project_dir: str | Path, suffix: str = "") -> dict[str, Path]:
    project_dir = Path(project_dir)
    suffix = str(suffix or "").strip()
    normalized_suffix = f"_{suffix}" if suffix else ""
    return {
        "registered_root": project_dir / f"stalign_registered{normalized_suffix}",
        "qc_dir": project_dir / f"stalign_qc{normalized_suffix}",
        "deformation_root": project_dir / f"stalign_deformation_fields{normalized_suffix}",
        "scores_csv": project_dir / f"stalign_registration_scores{normalized_suffix}.csv",
        "features_csv": project_dir / f"stalign_registration_features{normalized_suffix}.csv",
        "run_config_json": project_dir / f"stalign_run_config{normalized_suffix}.json",
        "reference_selection_csv": project_dir / f"stalign_reference_selection{normalized_suffix}.csv",
    }


def complete_landmark_points(points: dict[str, object]) -> bool:
    for name in LANDMARK_NAMES:
        point = points.get(name, {})
        if point.get("x") in (None, "") or point.get("y") in (None, ""):
            return False
    return True


def landmark_points_from_row(row: dict[str, object]) -> dict[str, dict[str, float]] | None:
    points: dict[str, dict[str, float]] = {}
    for name in LANDMARK_NAMES:
        x_value = row.get(f"{name}_x", "")
        y_value = row.get(f"{name}_y", "")
        if x_value in (None, "") or y_value in (None, ""):
            return None
        points[name] = {"x": float(x_value), "y": float(y_value)}
    return points


def landmark_array_from_row(row: dict[str, object]) -> list[list[float]] | None:
    points = landmark_points_from_row(row)
    if not points:
        return None
    return [[float(points[name]["x"]), float(points[name]["y"])] for name in LANDMARK_NAMES]


def landmark_table_row(
    item_id: str,
    image_path: str,
    label: str,
    points: dict[str, dict[str, float]],
    confidence: float = 1.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": str(item_id),
        "image_path": str(image_path),
        "label": str(label),
        "confidence": float(confidence),
    }
    for name in LANDMARK_NAMES:
        row[f"{name}_x"] = float(points[name]["x"])
        row[f"{name}_y"] = float(points[name]["y"])
    return row


def load_landmark_rows_from_annotations(project_dir: str | Path = DEFAULT_PROJECT_DIR) -> list[dict[str, object]]:
    paths = project_paths(project_dir)
    manifest = load_project_manifest(paths["manifest_csv"])
    annotations = load_annotations(paths["annotations_dir"])
    rows: list[dict[str, object]] = []
    for row in manifest:
        item_id = str(row["id"])
        payload = annotations.get(item_id)
        if not payload or not annotation_points_complete(payload):
            continue
        rows.append(
            landmark_table_row(
                item_id=item_id,
                image_path=str(row["image_path"]),
                label=str(row["label"]),
                points={name: payload["points"][name] for name in LANDMARK_NAMES},
                confidence=1.0,
            )
        )
    return rows


def load_landmark_rows_from_csv(csv_path: str | Path) -> list[dict[str, object]]:
    rows = read_csv_rows(csv_path)
    parsed: list[dict[str, object]] = []
    for row in rows:
        item_id = str(row.get("id", "")).strip()
        image_path = str(row.get("image_path", "")).strip()
        if not item_id or not image_path:
            continue
        points = landmark_points_from_row(row)
        if not points:
            continue
        label = str(row.get("label", "")).strip()
        confidence_raw = row.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw if confidence_raw not in ("", None) else 0.0)
        except Exception:
            confidence = 0.0
        parsed.append(
            landmark_table_row(
                item_id=item_id,
                image_path=image_path,
                label=label,
                points=points,
                confidence=confidence,
            )
        )
    return parsed
