"""Code-only logging and per-run provenance; never records credential environment variables."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import logging
import platform
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"project_TransUNet", "outputs", "wandb", "lightning_logs", ".codex",
            ".git", "__pycache__", ".pytest_cache", "synapse_test_selection"}


def log_project_code(logger, root=PROJECT_ROOT):
    if logger is None:
        return
    root = Path(root).resolve()
    try:
        experiment = getattr(logger, "experiment", None)
        if not callable(getattr(experiment, "log_code", None)):
            return
        from gitignore_parser import parse_gitignore
        ignore_path = root / ".gitignore"
        ignore = parse_gitignore(str(ignore_path), base_dir=str(root)) if ignore_path.is_file() else lambda path: False

        def excluded(path):
            path = Path(path)
            path = path if path.is_absolute() else root / path
            try:
                parts = path.relative_to(root).parts
            except ValueError:
                return True
            return any(part in EXCLUDED for part in parts) or ignore(str(path))

        experiment.log_code(str(root), include_fn=lambda path: Path(path).suffix in {".py", ".yaml", ".yml"},
                            exclude_fn=excluded)
    except Exception:
        logging.warning("Code archival failed; continuing without code upload", exc_info=True)


def write_run_manifest(trainer, model, run_dir):
    import torch
    versions = {}
    for name in ("torch", "torchvision", "lightning", "torchmetrics", "timm", "transformers",
                 "jsonargparse", "wandb", "numpy", "scipy", "medpy", "h5py"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = [PROJECT_ROOT / "main.py", PROJECT_ROOT / "requirements.txt", PROJECT_ROOT / "sitecustomize.py"]
    for folder in ("models", "datasets", "training", "configs/synapse"):
        sources.extend(f for f in (PROJECT_ROOT / folder).rglob("*") if f.suffix in {".py", ".yaml"})
    hashes = {str(f.relative_to(PROJECT_ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sources if f.is_file()}
    data = trainer.datamodule
    paths = {name: str(getattr(data, name)) for name in ("train_base_dir", "test_base_dir", "list_dir")
             if hasattr(data, name)}
    manifests = {}
    if data is not None and hasattr(data, "list_dir"):
        for attr in ("train_split", "test_split"):
            file = Path(data.list_dir) / f"{getattr(data, attr)}.txt"
            manifests[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    gpu = None
    if model.device.type == "cuda":
        props = torch.cuda.get_device_properties(model.device)
        gpu = {"name": props.name, "total_memory_bytes": props.total_memory,
               "capability": list(torch.cuda.get_device_capability(model.device)),
               "compiled_arches": torch.cuda.get_arch_list()}
    result = {
        "hostname": platform.node(), "python": sys.executable, "python_version": platform.python_version(),
        "cwd": str(Path.cwd()), "argv": sys.argv, "project_root": str(PROJECT_ROOT),
        "versions": versions, "torch_cuda": torch.version.cuda, "device": str(model.device), "gpu": gpu,
        "source_sha256": hashes, "manifest_sha256": manifests, "data_paths": paths,
        "resolved_cli_config": getattr(trainer, "baseline_resolved_config", None),
        "estimated_steps": trainer.estimated_stepping_batches,
        "optimizer_defaults": [opt.defaults for opt in trainer.optimizers],
        "optimizer_groups": [[{"name": group.get("name"), "lr": group["lr"],
                               "initial_lr": group.get("initial_lr"),
                               "numel": sum(param.numel() for param in group["params"])}
                              for group in opt.param_groups] for opt in trainer.optimizers],
        "callbacks": [type(cb).__module__ + "." + type(cb).__name__ for cb in trainer.callbacks],
        "selection_policy": "test set used for checkpoint selection; no gradient updates",
        "sitecustomize": getattr(sys.modules.get("sitecustomize"), "__file__", None),
    }
    (Path(run_dir) / "run_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
