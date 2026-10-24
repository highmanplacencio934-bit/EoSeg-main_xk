"""Build ASEE raw text/cooccurrence priors from fixed TRAIN slices only.

Run with python -m scripts.build_synapse_asee_prior --help. No work at import.
A local encoder and a verified label-ID mapping are required; no downloads.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from datasets.synapse_splits import resolve_project_path, validate_synapse_split, read_split
from models.asee import TRAIN_LIST_SHA256, TEST_LIST_SHA256, validate_prior

DEFAULT_LIST_DIR = "project_TransUNet/TransUNet/lists/lists_Synapse"
DEFAULT_TRAIN_DIR = "project_TransUNet/data/Synapse/train_npz"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_label_mapping(path):
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if mapping.get("verified") is not True or not mapping.get("source", "").strip():
        raise ValueError("Label mapping is unverified; supply the preprocessing label-map source")
    classes = mapping.get("classes", [])
    if [c.get("label_id") for c in classes] != list(range(1, 9)):
        raise ValueError("Mapping must have eight rows ordered by label ID 1..8")
    for row in classes:
        if not row.get("name", "").strip() or not row.get("description", "").strip():
            raise ValueError("Each label needs an organ name and semantic description")
    return mapping


def conditional_from_counts(counts, cooccurrence, num_samples, eps=1e-5):
    """Reference smoothing is retained, including smoothing before the ratio."""
    counts = np.asarray(counts, dtype=np.float64)
    cooccurrence = np.asarray(cooccurrence, dtype=np.float64)
    if (isinstance(num_samples, bool) or not isinstance(num_samples, (int, np.integer))
            or num_samples <= 0 or not np.isfinite(eps) or eps <= 0):
        raise ValueError("Expected a positive integer sample count and finite positive epsilon")
    if counts.ndim != 1 or not counts.size or cooccurrence.shape != (len(counts), len(counts)):
        raise ValueError("Invalid class/cooccurrence count shapes")
    if (not np.isfinite(counts).all() or not np.isfinite(cooccurrence).all()
            or not np.equal(counts, np.floor(counts)).all()
            or not np.equal(cooccurrence, np.floor(cooccurrence)).all()):
        raise ValueError("Counts must be finite integers")
    if (counts <= 0).any() or (counts > num_samples).any():
        raise ValueError("Every foreground class must occur between one and N times")
    if (not np.array_equal(cooccurrence, cooccurrence.T)
            or not np.array_equal(np.diag(cooccurrence), counts)
            or (cooccurrence < 0).any()
            or (cooccurrence > np.minimum(counts[:, None], counts[None, :])).any()):
        raise ValueError("Inconsistent cooccurrence counts, diagonal or symmetry")
    p_class = (counts + eps) / (num_samples + eps)
    p_conditional = (cooccurrence + eps) / (counts[:, None] + eps)
    return torch.tensor(np.log((p_conditional + eps) / (p_class[None, :] + eps)), dtype=torch.float32)


def count_training_slices(train_dir, names):
    """Stream raw unaugmented masks. One slice is one statistical sample."""
    train_dir = Path(train_dir).resolve()
    counts = np.zeros(8, dtype=np.int64)
    cooccurrence = np.zeros((8, 8), dtype=np.int64)
    files = {}
    if len(names) != len(set(names)) or not names:
        raise ValueError("Training names must be unique and nonempty")
    for name in names:
        # This helper also rejects traversal when used independently of split validation.
        import re
        if not re.fullmatch(r"case[0-9]+_slice[0-9]+", name):
            raise ValueError(f"Not a training slice: {name}")
        path = (train_dir / f"{name}.npz").resolve()
        if path.parent != train_dir:
            raise ValueError("Training NPZ symlink escapes the training directory")
        before = sha256(path)
        with np.load(path, allow_pickle=False) as sample:
            label = sample["label"]
            numeric = np.issubdtype(label.dtype, np.integer) or np.issubdtype(label.dtype, np.floating)
            if label.ndim != 2 or label.size == 0 or not numeric:
                raise ValueError(f"Expected nonempty numeric 2D raw labels: {path}")
            # The project's NPZ labels are float32 with exact integer values.
            # Validate before casting: never silently round interpolated/fractional labels.
            if not np.isfinite(label).all() or not np.equal(label, np.floor(label)).all():
                raise ValueError(f"Expected finite integer-valued labels: {path}")
            ids = np.unique(label)
            if ((ids < 0) | (ids > 8)).any():
                raise ValueError(f"Unexpected raw labels {ids.tolist()}: {path}")
            present = ids[ids > 0].astype(int) - 1
            counts[present] += 1
            cooccurrence[np.ix_(present, present)] += 1
        if sha256(path) != before:
            raise RuntimeError(f"Training file changed during prior construction: {path}")
        files[f"{name}.npz"] = before
    return counts, cooccurrence, files


def encode_text(descriptions, encoder_path):
    # Imports are lazy; tests/statistics/help never load transformers or a large model.
    from transformers import AutoModel, AutoTokenizer
    encoder_path = Path(encoder_path).expanduser().resolve(strict=True)
    if not encoder_path.is_dir():
        raise ValueError("--encoder-path must be a local pretrained encoder directory")
    encoder_files = {str(f.relative_to(encoder_path)): sha256(f)
                     for f in sorted(encoder_path.rglob("*")) if f.is_file()}
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_path), local_files_only=True)
    encoder = AutoModel.from_pretrained(str(encoder_path), local_files_only=True).cpu().eval()
    max_length = getattr(encoder.config, "max_position_embeddings", None)
    if not isinstance(max_length, int) or max_length < 2:
        raise ValueError("Local encoder must declare a valid maximum position count")
    tokenizer_limit = tokenizer.model_max_length
    if isinstance(tokenizer_limit, int) and tokenizer_limit > 1:
        max_length = min(max_length, tokenizer_limit)
    inputs = tokenizer(descriptions, padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    with torch.inference_mode():
        features = encoder(**inputs).last_hidden_state[:, 0, :].detach().float().clone()
    after = {str(f.relative_to(encoder_path)): sha256(f)
             for f in sorted(encoder_path.rglob("*")) if f.is_file()}
    if after != encoder_files:
        raise RuntimeError("Local text encoder files changed during feature extraction")
    return features, {"requested_model": "emilyalsentzer/Bio_ClinicalBERT",
                      "identity_verification": "Local files hashed; verify checkpoint origin before use",
                      "path": str(encoder_path), "pooling": "last_hidden_state_CLS",
                      "local_files_only": True, "max_length": max_length, "file_sha256": encoder_files}


def build_prior(args):
    output = resolve_project_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prior: {output}")
    list_dir = resolve_project_path(args.list_dir)
    train_dir = resolve_project_path(args.train_dir)
    mapping_path = resolve_project_path(args.label_map_json)
    mapping_hash = sha256(mapping_path)
    mapping = load_label_mapping(mapping_path)
    hashes = {name: sha256(list_dir / name) for name in ("train.txt", "test_vol.txt")}
    if hashes != {"train.txt": TRAIN_LIST_SHA256, "test_vol.txt": TEST_LIST_SHA256}:
        raise ValueError("Prior construction is restricted to this project's unchanged fixed split")
    split = validate_synapse_split(train_dir, None, list_dir,
                                   check_train_files=True, check_test_files=False)
    if split["train_slices"] != 2211 or len(split["train_cases"]) != 18 or len(split["test_cases"]) != 12:
        raise ValueError("Unexpected fixed Synapse split dimensions")
    names = read_split(list_dir, "train")
    counts, pairs, input_hashes = count_training_slices(train_dir, names)
    matrix = conditional_from_counts(counts, pairs, len(names))
    text, encoder_metadata = encode_text([r["description"] for r in mapping["classes"]], args.encoder_path)
    if text.shape != (8, 768):
        raise ValueError("Expected BioClinicalBERT CLS features [8,768]; verify the selected encoder")
    # Recheck small inputs before publishing; test volumes are never opened.
    if hashes != {name: sha256(list_dir / name) for name in hashes}:
        raise RuntimeError("Split manifest changed during construction")
    if sha256(mapping_path) != mapping_hash:
        raise RuntimeError("Label mapping changed during construction")
    prior = {
        "schema_version": 1, "dataset": "synapse", "class_ids": list(range(1, 9)),
        "class_names": [r["name"] for r in mapping["classes"]],
        "descriptions": [r["description"] for r in mapping["classes"]],
        "label_mapping_verified": True, "label_mapping_source": mapping["source"],
        "label_mapping_sha256": mapping_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "text_features": text, "conditional_matrix": matrix, "text_encoder": encoder_metadata,
        "statistics": {"unit": "train_npz_slice", "eps": 1e-5, "num_samples": len(names),
                       "train_list_sha256": hashes["train.txt"], "test_list_sha256": hashes["test_vol.txt"],
                       "train_cases": split["train_cases"], "class_counts": counts.tolist(),
                       "cooccurrence_counts": pairs.tolist(), "train_npz_sha256": input_hashes},
    }
    validate_prior(prior)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids silently replacing a prior used by another run.
    with output.open("xb") as stream:
        torch.save(prior, stream)
    print(json.dumps({"prior": str(output), "sha256": sha256(output), "samples": len(names)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--list-dir", default=DEFAULT_LIST_DIR)
    parser.add_argument("--label-map-json", required=True)
    parser.add_argument("--encoder-path", required=True, help="Local BioClinicalBERT directory; CPU/offline only")
    parser.add_argument("--output", default="priors/synapse_asee_v1.pt")
    args = parser.parse_args()
    torch.set_num_threads(2)
    build_prior(args)


if __name__ == "__main__":
    main()
