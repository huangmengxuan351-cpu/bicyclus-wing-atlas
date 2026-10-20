from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from pose_landmark_utils import (
    DEFAULT_PROJECT_DIR,
    LABELS,
    annotation_points_complete,
    load_annotations,
    load_project_manifest,
    project_paths,
    write_csv_rows,
)
from stage_template_geometry import (
    CANVAS_SIZE,
    apply_similarity,
    base_center_from_landmarks,
    compute_notch_features,
    compute_radial_profile,
    compute_region_anchors,
    compute_width_profile,
    convex_hull_points,
    extract_contour_points,
    extract_wing_mask,
    mask_iou,
    named_landmarks_from_points,
    points_from_named_landmarks,
    save_registration_qc,
    smooth_circular,
    template_to_canonical,
    tps_warp,
    transform_image_similarity,
    transform_mask_similarity,
    umeyama_similarity,
)


def load_template(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_path"] = str(path.resolve())
    payload["_landmarks"] = points_from_named_landmarks(payload["landmarks"])
    payload["_notch_anchor"] = np.asarray(
        [float(payload["notch_anchor"]["x"]), float(payload["notch_anchor"]["y"])],
        dtype=np.float64,
    )
    payload["_region_anchors"] = {
        name: np.asarray([float(point["x"]), float(point["y"])], dtype=np.float64)
        for name, point in payload.get("region_anchors", {}).items()
    }
    payload["_radial_profile"] = np.asarray(payload["radial_profile"], dtype=np.float64)
    payload["_width_profile"] = np.asarray(payload["width_profile"], dtype=np.float64)
    mask_path = payload.get("mask_path", "")
    if mask_path:
        with Image.open(mask_path) as image:
            payload["_mask"] = np.asarray(image.convert("L"), dtype=np.uint8) > 127
    else:
        payload["_mask"] = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    return payload


def load_stage_templates(project_dir: Path) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    paths = project_paths(project_dir)
    base_template = load_template(paths["base_template_json"])
    stage_templates = {
        "0.5": load_template(paths["stage_template_0_5_json"]),
        "0.75": load_template(paths["stage_template_0_75_json"]),
        "1.0": load_template(paths["stage_template_1_0_json"]),
    }
    return base_template, stage_templates


def load_landmarks_from_annotations(project_dir: Path) -> list[dict[str, object]]:
    paths = project_paths(project_dir)
    manifest = load_project_manifest(paths["manifest_csv"])
    annotations = load_annotations(paths["annotations_dir"])
    rows = []
    for row in manifest:
        item_id = str(row["id"])
        payload = annotations.get(item_id)
        if not payload or not annotation_points_complete(payload):
            continue
        rows.append(
            {
                "id": item_id,
                "label": str(row["label"]),
                "image_path": str(row["image_path"]),
                "landmarks": points_from_named_landmarks(payload["points"]),
            }
        )
    return rows


def load_landmarks_from_csv(csv_path: Path) -> list[dict[str, object]]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item_id = str(row.get("id", "")).strip()
            image_path = str(row.get("image_path", "")).strip()
            label = str(row.get("label", "")).strip()
            if not item_id or not image_path:
                continue
            points = []
            for name in ["base_dorsal", "base_ventral", "distal_tip", "posterior_notch"]:
                x = row.get(f"{name}_x", "")
                y = row.get(f"{name}_y", "")
                if x == "" or y == "":
                    points = []
                    break
                points.append([float(x), float(y)])
            if len(points) != 4:
                continue
            rows.append(
                {
                    "id": item_id,
                    "label": label,
                    "image_path": image_path,
                    "landmarks": np.asarray(points, dtype=np.float64),
                }
            )
    return rows


def build_registration_features(
    aligned_mask: np.ndarray,
    aligned_contour: np.ndarray,
    aligned_landmarks: np.ndarray,
    stage_template: dict[str, object],
) -> dict[str, float]:
    center = base_center_from_landmarks(aligned_landmarks)
    radial = smooth_circular(compute_radial_profile(aligned_contour, center), window=9)
    width = smooth_circular(compute_width_profile(aligned_mask, aligned_landmarks), window=5)
    canonical_contour = template_to_canonical(aligned_contour)
    notch = compute_notch_features(canonical_contour, convex_hull_points(canonical_contour))
    notch_anchor = np.asarray([notch["notch_point"]], dtype=np.float64)
    notch_anchor = np.asarray(
        [
            [
                stage_template["canvas_size"] * 0.0 + (1536.0 - 1400.0 * notch_anchor[0, 0]),
                1024.0 + 1400.0 * notch_anchor[0, 1],
            ]
        ],
        dtype=np.float64,
    )[0]
    landmark_error = float(np.linalg.norm(aligned_landmarks - stage_template["_landmarks"], axis=1).mean())
    radial_l2 = float(np.sqrt(np.mean((radial - stage_template["_radial_profile"]) ** 2)))
    width_l2 = float(np.sqrt(np.mean((width - stage_template["_width_profile"]) ** 2)))
    notch_offset = float(np.linalg.norm(notch_anchor - stage_template["_notch_anchor"]))
    mask_score = float(mask_iou(aligned_mask, stage_template["_mask"]))
    return {
        "landmark_similarity_error": landmark_error,
        "radial_profile_l2": radial_l2,
        "width_profile_l2": width_l2,
        "notch_offset": notch_offset,
        "mask_iou_to_template": mask_score,
    }


def make_feature_row(
    item_id: str,
    true_label: str,
    image_path: str,
    per_stage: dict[str, dict[str, float]],
) -> dict[str, object]:
    best = sorted(per_stage.items(), key=lambda item: item[1]["overall_template_cost"])
    best_label, best_metrics = best[0]
    second_cost = float(best[1][1]["overall_template_cost"]) if len(best) > 1 else float(best_metrics["overall_template_cost"])
    row = {
        "id": item_id,
        "true_label": true_label,
        "image_path": image_path,
        "best_template_label": best_label,
        "best_template_cost": float(best_metrics["overall_template_cost"]),
        "second_template_cost": second_cost,
        "template_margin": float(second_cost - float(best_metrics["overall_template_cost"])),
    }
    for label in LABELS:
        metrics = per_stage[label]
        row[f"cost_to_{label.replace('.', '_')}"] = float(metrics["overall_template_cost"])
        row[f"landmark_error_to_{label.replace('.', '_')}"] = float(metrics["landmark_similarity_error"])
        row[f"tps_energy_to_{label.replace('.', '_')}"] = float(metrics["tps_bending_energy"])
        row[f"radial_l2_to_{label.replace('.', '_')}"] = float(metrics["radial_profile_l2"])
        row[f"width_l2_to_{label.replace('.', '_')}"] = float(metrics["width_profile_l2"])
        row[f"notch_offset_to_{label.replace('.', '_')}"] = float(metrics["notch_offset"])
        row[f"mask_iou_to_{label.replace('.', '_')}"] = float(metrics["mask_iou_to_template"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Register each image against all stage templates.")
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    parser.add_argument("--predicted-landmarks-csv", default="")
    parser.add_argument("--save-qc", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    paths = project_paths(project_dir)
    if args.clean_output:
        if paths["stage_registered_root"].exists():
            shutil.rmtree(paths["stage_registered_root"])
        if paths["stage_registration_qc_dir"].exists():
            shutil.rmtree(paths["stage_registration_qc_dir"])
        if paths["stage_registration_scores_csv"].exists():
            paths["stage_registration_scores_csv"].unlink()
        if paths["stage_registration_features_csv"].exists():
            paths["stage_registration_features_csv"].unlink()
    paths["stage_registered_root"].mkdir(parents=True, exist_ok=True)
    paths["stage_registration_qc_dir"].mkdir(parents=True, exist_ok=True)
    for label in ["0_5", "0_75", "1_0"]:
        (paths["stage_registered_root"] / label).mkdir(parents=True, exist_ok=True)

    base_template, stage_templates = load_stage_templates(project_dir)
    rows = (
        load_landmarks_from_csv(Path(args.predicted_landmarks_csv))
        if args.predicted_landmarks_csv
        else load_landmarks_from_annotations(project_dir)
    )
    if not rows:
        raise ValueError("No usable landmarks found for registration.")
    if args.limit > 0:
        rows = rows[: int(args.limit)]

    score_rows = []
    feature_rows = []
    for row in rows:
        item_id = str(row["id"])
        true_label = str(row.get("label", ""))
        image_path = Path(str(row["image_path"]))
        landmarks = np.asarray(row["landmarks"], dtype=np.float64)

        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
        mask = extract_wing_mask(gray)
        contour = extract_contour_points(mask)

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
                np.vstack([source_region_anchors[name] for name in ["proximal", "distal_outer_margin", "posterior_notch", "dorsal_ventral_transition"]]),
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

            stage_registered_image, tps_energy = tps_warp(base_aligned_image, source_controls, destination_controls, order=1)
            stage_dir = paths["stage_registered_root"] / template_label.replace(".", "_")
            registered_path = stage_dir / f"{item_id}.png"
            Image.fromarray(stage_registered_image, mode="L").save(registered_path)

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
            metrics["registered_image_path"] = str(registered_path.resolve())
            per_stage[template_label] = metrics

            qc_path = ""
            if args.save_qc:
                qc_output = paths["stage_registration_qc_dir"] / f"{item_id}__{template_label.replace('.', '_')}.png"
                save_registration_qc(
                    source_image=gray,
                    base_aligned_image=base_aligned_image,
                    stage_registered_image=stage_registered_image,
                    base_landmarks=base_template["_landmarks"],
                    stage_landmarks=template_payload["_landmarks"],
                    source_controls=source_controls,
                    stage_controls=destination_controls,
                    output_path=qc_output,
                )
                qc_path = str(qc_output.resolve())

            score_rows.append(
                {
                    "id": item_id,
                    "true_label": true_label,
                    "image_path": str(image_path.resolve()),
                    "template_label": template_label,
                    "registered_image_path": metrics["registered_image_path"],
                    "landmark_similarity_error": metrics["landmark_similarity_error"],
                    "tps_bending_energy": metrics["tps_bending_energy"],
                    "radial_profile_l2": metrics["radial_profile_l2"],
                    "width_profile_l2": metrics["width_profile_l2"],
                    "notch_offset": metrics["notch_offset"],
                    "mask_iou_to_template": metrics["mask_iou_to_template"],
                    "overall_template_cost": metrics["overall_template_cost"],
                    "qc_path": qc_path,
                }
            )

        feature_rows.append(make_feature_row(item_id, true_label, str(image_path.resolve()), per_stage))

    write_csv_rows(
        paths["stage_registration_scores_csv"],
        [
            "id",
            "true_label",
            "image_path",
            "template_label",
            "registered_image_path",
            "landmark_similarity_error",
            "tps_bending_energy",
            "radial_profile_l2",
            "width_profile_l2",
            "notch_offset",
            "mask_iou_to_template",
            "overall_template_cost",
            "qc_path",
        ],
        score_rows,
    )

    feature_fieldnames = list(feature_rows[0].keys()) if feature_rows else ["id", "true_label", "image_path"]
    write_csv_rows(paths["stage_registration_features_csv"], feature_fieldnames, feature_rows)

    print(f"Stage registration scores: {paths['stage_registration_scores_csv']}")
    print(f"Stage registration features: {paths['stage_registration_features_csv']}")
    print(f"Registered stage images: {paths['stage_registered_root']}")


if __name__ == "__main__":
    main()
