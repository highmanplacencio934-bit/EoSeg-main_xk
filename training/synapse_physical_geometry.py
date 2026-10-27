"""Verified case geometry and MedPy HD95 in millimetres for Synapse.

This module is opt-in. No geometry is inferred from an H5 shape or case number.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np
from medpy import metric

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def label_fingerprint(label):
    """Stable digest of the actual 0..8 labels, independent of storage dtype."""
    label = np.asarray(label)
    if label.ndim != 3 or not np.isfinite(label).all():
        raise ValueError("Expected a finite 3D Synapse label volume")
    if (label < 0).any() or (label > 8).any() or not np.equal(label, np.floor(label)).all():
        raise ValueError("Expected integer Synapse labels 0..8")
    return hashlib.sha256(np.ascontiguousarray(label, dtype=np.uint8).tobytes()).hexdigest()


def checked_spacing(values):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("spacing_zyx_mm must contain exactly three values in Z,Y,X order")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError("spacing_zyx_mm must contain real numbers, not null/placeholders")
    spacing = tuple(float(v) for v in values)
    if not all(np.isfinite(v) and v > 0 for v in spacing):
        raise ValueError("spacing_zyx_mm must be finite and strictly positive")
    return spacing


@dataclass(frozen=True)
class CaseGeometry:
    spacing_zyx_mm: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    label_sha256: str
    source: str
    mapping: str


class SpacingManifest:
    def __init__(self, path):
        self.path = project_path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Verified physical spacing is missing: {self.path}. "
                "Build it from the matching original NIfTI labels; no unit-spacing fallback is allowed."
            )
        self.raw_bytes = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.raw_bytes).hexdigest()
        data = json.loads(self.raw_bytes)
        if data.get("schema_version") != 1 or data.get("units") != "mm" or data.get("array_axis_order") != "zyx":
            raise ValueError("Expected schema_version=1, units=mm, array_axis_order=zyx")
        if data.get("verified") is not True or data.get("orthogonal_grid") is not True:
            raise ValueError("Spacing manifest must verify geometry and an orthogonal regular grid")
        cases = data.get("cases")
        if not isinstance(cases, dict) or not cases:
            raise ValueError("Spacing manifest must contain per-case geometry")
        self.cases = {}
        for case, row in cases.items():
            if not re.fullmatch(r"case[0-9]+", case):
                raise ValueError(f"Invalid case ID: {case}")
            spacing = checked_spacing(row.get("spacing_zyx_mm"))
            shape = row.get("shape_zyx")
            if (not isinstance(shape, list) or len(shape) != 3 or
                    any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)):
                raise ValueError(f"Invalid shape_zyx for {case}")
            digest = row.get("label_sha256", "")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Missing canonical label SHA256 for {case}")
            if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("source", "mapping")):
                raise ValueError(f"Missing original source or geometry mapping evidence for {case}")
            self.cases[case] = CaseGeometry(spacing, tuple(shape), digest, row["source"], row["mapping"])

    def for_case(self, case, label):
        if case not in self.cases:
            raise ValueError(f"No verified physical spacing for {case}")
        geometry = self.cases[case]
        label = np.asarray(label)
        if tuple(label.shape) != geometry.shape_zyx or label_fingerprint(label) != geometry.label_sha256:
            raise ValueError(f"Geometry does not match the actual H5 label volume: {case}")
        return geometry

    def validate_dataset(self, base_dir, list_file):
        names = [line.strip() for line in project_path(list_file).read_text().splitlines() if line.strip()]
        if not names or len(names) != len(set(names)) or any(not re.fullmatch(r"case[0-9]+", n) for n in names):
            raise ValueError("Expected distinct case IDs in the test split")
        missing = set(names) - set(self.cases)
        if missing:
            raise ValueError(f"Physical spacing is incomplete; missing {sorted(missing)}")
        result = []
        for case in names:
            with h5py.File(project_path(base_dir) / f"{case}.npy.h5", "r") as volume:
                if volume["image"].shape != volume["label"].shape:
                    raise ValueError(f"Image/label shape mismatch: {case}")
                geometry = self.for_case(case, volume["label"][:])
            result.append({"case": case, "spacing_zyx_mm": geometry.spacing_zyx_mm,
                           "shape_zyx": geometry.shape_zyx, "label_sha256": geometry.label_sha256})
        return result


def binary_metrics_mm(pred, gt, spacing_zyx_mm):
    """Change distance units only; keep the existing baseline Dice/empty policy.

    Empty masks have no measured surface distance. Their legacy zeros and Dice
    values are retained and explicitly marked, rather than silently repaired.
    """
    spacing = checked_spacing(spacing_zyx_mm)
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if pred.ndim != 3 or pred.shape != gt.shape:
        raise ValueError("Physical HD95 requires equal-shape 3D masks")
    has_pred, has_gt = bool(pred.any()), bool(gt.any())
    if has_pred and has_gt:
        return {"dice": float(metric.binary.dc(pred, gt)),
                "hd95_mm": float(metric.binary.hd95(pred, gt, voxelspacing=spacing)),
                "hd95_voxel": float(metric.binary.hd95(pred, gt)), "status": "nonempty"}
    status = ("empty_gt_legacy_zero" if has_pred else
              "empty_pred_legacy_zero" if has_gt else "both_empty_legacy_zero")
    return {"dice": 1.0 if has_pred and not has_gt else 0.0,
            "hd95_mm": 0.0, "hd95_voxel": 0.0, "status": status}
