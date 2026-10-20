from __future__ import annotations

import re
from pathlib import Path


KNOWN_SAMPLE_PREFIXES = (
    "forewing_stage::",
    "hindwing_stage::",
    "stage_0_5::",
    "stage_0_75::",
    "stage_1_0::",
    "aligned_dapi_forewing::",
    "aligned_hcr_forewing::",
    "aligned_hcr_hindwing::",
    "aligned_dapi_hindwing_stage1::",
    "labeling_auto_align_forewing::",
    "labeling_review_hindwing::",
    "manual_standardized_forewing_dapi::",
    "reviewed_forewing_gt100::",
    "reviewed_hindwing_early::",
    "reviewed_hindwing_stage1::",
    "auto_forewing_gt100::",
    "auto_forewing_sal::",
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


def _strip_known_prefixes(text: str) -> str:
    lowered = text.lower()
    for prefix in KNOWN_SAMPLE_PREFIXES:
        if lowered.startswith(prefix):
            return lowered[len(prefix) :]
    return lowered


def canonical_original_id(original_id: object = "", item_id: object = "", image_path: str | Path = "") -> str:
    raw = str(original_id or "").strip() or str(item_id or "").strip()
    if not raw and image_path:
        raw = Path(image_path).stem
    raw = re.sub(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", "", raw, flags=re.IGNORECASE)
    lowered = _strip_known_prefixes(raw)
    lowered = re.sub(r"_(488|514|546|647)hcr$", "", lowered)
    lowered = re.sub(r"_dapi$", "", lowered)
    lowered = re.sub(r"_bright_only$", "", lowered)
    lowered = re.sub(r"_bright$", "", lowered)
    lowered = re.sub(r"_aggressive$", "", lowered)
    lowered = lowered.strip("_")
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    if lowered:
        return lowered
    if image_path:
        return re.sub(r"[^a-z0-9]+", "_", Path(image_path).stem.lower()).strip("_") or "sample"
    return "sample"


def strict_group_key(original_id: object = "", item_id: object = "", image_path: str | Path = "") -> str:
    return f"strict::{canonical_original_id(original_id=original_id, item_id=item_id, image_path=image_path)}"


def infer_gene_family(original_id: object = "", item_id: object = "", image_path: str | Path = "") -> str:
    raw = str(item_id or "").strip() or str(original_id or "").strip()
    if not raw and image_path:
        raw = Path(image_path).stem
    raw = re.sub(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", "", raw, flags=re.IGNORECASE)
    lowered = _strip_known_prefixes(raw).strip("_")
    if not lowered:
        lowered = canonical_original_id(original_id=original_id, item_id=item_id, image_path=image_path)
    token = re.split(r"[^a-zA-Z0-9]+", lowered)[0].lower().strip("_")
    token = re.sub(r"b\d+$", "", token)
    token = re.sub(r"\d+$", "", token)
    token = token.strip("_")
    if token in {"stage", "sample", "aligned", "manual", "review"}:
        token = ""
    return token or ""


def group_value(row: dict[str, object]) -> str:
    return (
        str(row.get("strict_group", "")).strip()
        or str(row.get("split_group", "")).strip()
        or strict_group_key(
            original_id=row.get("original_id", ""),
            item_id=row.get("id", ""),
            image_path=str(row.get("image_path", "")),
        )
    )
