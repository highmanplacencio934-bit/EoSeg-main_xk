"""Resolve Synapse paths and validate a fixed case-level train/test split."""
from pathlib import Path
import re
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = "project_TransUNet/data/Synapse"
DEFAULT_LIST_DIR = "project_TransUNet/TransUNet/lists/lists_Synapse"


def resolve_project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_split(list_dir, split):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", split):
        raise ValueError(f"Invalid split name: {split!r}")
    path = Path(list_dir) / f"{split}.txt"
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Empty Synapse split: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate entries in Synapse split: {path}")
    for name in names:
        if not re.fullmatch(r"case[0-9]+(?:_slice[0-9]+)?", name):
            raise ValueError(f"Invalid Synapse entry {name!r} in {path}")
    return names


def validate_synapse_split(train_base_dir, test_base_dir, list_dir,
                           train_split="train", test_split="test_vol", *,
                           check_train_files=True, check_test_files=True):
    """Always validate both manifests; check only the requested data directories."""
    train = read_split(list_dir, train_split)
    test = read_split(list_dir, test_split)
    if any("_slice" not in name for name in train):
        raise ValueError("Training entries must be 2D slice names (caseNNNN_sliceNNN)")
    if any("_slice" in name for name in test):
        raise ValueError("Test entries must be volume case names (caseNNNN)")
    train_cases = {name.split("_slice")[0] for name in train}
    overlap = train_cases.intersection(test)
    if overlap:
        raise ValueError(f"Synapse train/test case overlap: {sorted(overlap)}")
    # Enumerate only active-stage directories, once each (important for SSHFS).
    missing = []
    if check_train_files:
        with os.scandir(train_base_dir) as entries:
            train_files = {entry.name for entry in entries if entry.is_file()}
        missing.extend(str(Path(train_base_dir) / f"{name}.npz") for name in train
                       if f"{name}.npz" not in train_files)
    if check_test_files:
        with os.scandir(test_base_dir) as entries:
            test_files = {entry.name for entry in entries if entry.is_file()}
        missing.extend(str(Path(test_base_dir) / f"{name}.npy.h5") for name in test
                       if f"{name}.npy.h5" not in test_files)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} Synapse files; first entries: {missing[:5]}")
    return {"train_slices": len(train), "train_cases": sorted(train_cases),
            "test_cases": test, "case_overlap": sorted(overlap)}
