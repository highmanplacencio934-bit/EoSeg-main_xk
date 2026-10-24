"""Build verified Z,Y,X millimetre spacing only from exactly matching NIfTI labels.

This CPU-only tool reads original labels and H5 files; neither is modified.
Cropping/resampling is deliberately rejected until its geometry is known.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from training.synapse_physical_geometry import checked_spacing, label_fingerprint, project_path, SpacingManifest


def require_nifti_mm(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        header = stream.read(348)
    if len(header) < 348:
        raise ValueError(f"Truncated NIfTI header: {path}")
    size = struct.unpack("<i", header[:4])[0]
    if size != 348:
        size = struct.unpack(">i", header[:4])[0]
    if size != 348 or header[344:348] != b"n+1\x00":
        raise ValueError("Expected a single-file NIfTI-1 .nii/.nii.gz with explicit units")
    if header[123] & 7 != 2:
        raise ValueError(f"Original NIfTI does not explicitly declare millimetres: {path}")


def map_source_labels(raw, label_map, source_labels):
    if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError("Original labels must contain finite integers")
    if source_labels == "synapse8":
        label_fingerprint(raw)
        return raw.astype(np.uint8)
    if raw.min() < 0 or raw.max() > 13:
        raise ValueError("Expected original BTCV labels 0..13")
    remapped = np.zeros(raw.shape, dtype=np.uint8)
    for row in label_map["classes"]:
        remapped[raw == row["raw_btcv_label_id"]] = row["label_id"]
    return remapped


def match_label_geometry(raw_zyx, target_zyx, spacing_source_zyx):
    """Require complete voxel agreement after an axis permutation and flips."""
    spacing_source_zyx = checked_spacing(list(spacing_source_zyx))
    label_fingerprint(target_zyx)
    if raw_zyx.ndim != 3 or not np.any(target_zyx):
        raise ValueError("Expected nonempty 3D source and target labels")
    probe = tuple(slice(None, None, max(1, n // 32)) for n in target_zyx.shape)
    matches = []
    for axes in permutations(range(3)):
        arranged = raw_zyx.transpose(axes)
        if arranged.shape != target_zyx.shape:
            continue
        for flips in product((False, True), repeat=3):
            candidate = arranged[tuple(slice(None, None, -1 if f else 1) for f in flips)]
            if np.array_equal(candidate[probe], target_zyx[probe]) and np.array_equal(candidate, target_zyx):
                matches.append({"output_to_source_axes": list(axes), "output_axis_flips": list(flips),
                                "spacing_zyx_mm": [spacing_source_zyx[i] for i in axes]})
    if not matches:
        raise ValueError("Original/H5 labels do not exactly match after axis permutations/flips; "
                         "cropping/resampling or a different source needs explicit geometry evidence")
    reference = matches[0]["spacing_zyx_mm"]
    if any(not np.allclose(m["spacing_zyx_mm"], reference, rtol=0, atol=1e-8) for m in matches):
        raise ValueError("Ambiguous axis mapping gives different physical spacing")
    return reference, matches


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-label-dir", type=Path)
    parser.add_argument("--label-pattern", default="label{case_id}.nii.gz",
                        help="Relative filename template supporting {case} and {case_id}")
    parser.add_argument("--source-labels", choices=("btcv13", "synapse8"), default="btcv13")
    parser.add_argument("--test-base-dir", default="project_TransUNet/data/Synapse/test_vol_h5")
    parser.add_argument("--list-file", default="project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt")
    parser.add_argument("--label-map", default="priors/synapse_label_map.verified.json")
    parser.add_argument("--output", default="priors/synapse_spacing.verified.json")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate an existing --output manifest against all current test labels")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.validate_only:
        manifest = SpacingManifest(args.output)
        rows = manifest.validate_dataset(args.test_base_dir, args.list_file)
        print(json.dumps({"validated_cases": rows, "manifest_sha256": manifest.sha256}, indent=2))
        return
    if args.raw_label_dir is None:
        raise ValueError("--raw-label-dir is required when building a new manifest")
    import SimpleITK as sitk
    output = project_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing geometry: {output}")
    mapping = json.loads(project_path(args.label_map).read_text(encoding="utf-8"))
    if mapping.get("verified") is not True or {r["label_id"] for r in mapping["classes"]} != set(range(1, 9)):
        raise ValueError("A verified mapping of all eight foreground organs is required")
    names = [s.strip() for s in project_path(args.list_file).read_text().splitlines() if s.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("Expected a nonempty, unique test-case list")
    cases = {}
    for case in names:
        if not case.startswith("case") or not case[4:].isdigit():
            raise ValueError(f"Invalid case ID {case}")
        original = project_path(args.raw_label_dir) / args.label_pattern.format(case=case, case_id=case[4:])
        require_nifti_mm(original)
        image = sitk.ReadImage(str(original))
        if image.GetDimension() != 3:
            raise ValueError(f"Original label is not 3D: {original}")
        direction = np.array(image.GetDirection()).reshape(3, 3)
        if not np.allclose(direction.T @ direction, np.eye(3), rtol=0, atol=1e-5):
            raise ValueError(f"Nonorthogonal physical grid is unsupported by voxelspacing: {original}")
        raw = map_source_labels(sitk.GetArrayFromImage(image), mapping, args.source_labels)
        h5 = project_path(args.test_base_dir) / f"{case}.npy.h5"
        with h5py.File(h5, "r") as volume:
            target = volume["label"][:]
            if volume["image"].shape != target.shape:
                raise ValueError(f"Image/label shape mismatch: {case}")
        spacing, matches = match_label_geometry(raw, target, tuple(reversed(image.GetSpacing())))
        cases[case] = {"spacing_zyx_mm": spacing, "shape_zyx": list(target.shape),
                       "label_sha256": label_fingerprint(target), "source": str(original.resolve()),
                       "source_label_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                       "source_spatial_units": "NIfTI-1 xyzt_units explicitly mm",
                       "mapping": "Full H5-label equality after foreground remapping, axis permutation and flips; no resampling",
                       "source_label_format": args.source_labels, "matching_transforms": matches}
        print(f"Verified {case}: spacing_zyx_mm={spacing}", flush=True)
    payload = {"schema_version": 1, "units": "mm", "array_axis_order": "zyx", "verified": True,
               "orthogonal_grid": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "label_map_sha256": hashlib.sha256(project_path(args.label_map).read_bytes()).hexdigest(),
               "cases": cases}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Wrote {output}; {len(cases)} verified cases", flush=True)


if __name__ == "__main__":
    main()
