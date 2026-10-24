"""Read-only data audit and opt-in instrumentation; never enabled by baseline YAML."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(output):
    import h5py
    import numpy as np
    import torch
    from datasets.synapse_splits import validate_synapse_split
    from training.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule

    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    lists = ROOT / "project_TransUNet/TransUNet/lists/lists_Synapse"
    data = ROOT / "project_TransUNet/data/Synapse"
    split = validate_synapse_split(data / "train_npz", data / "test_vol_h5", lists)
    environment = {
        "hostname": platform.node(), "python": sys.executable, "cwd": str(Path.cwd()),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "compiled_arches": torch.cuda.get_arch_list(),
        "sitecustomize": getattr(sys.modules.get("sitecustomize"), "__file__", None),
        "offline_environment": {k: os.getenv(k) for k in ("HF_HUB_OFFLINE", "TORCH_HOME", "TIMM_USE_OLD_CACHE")},
        "disk_bytes": dict(zip(("total", "used", "free"), shutil.disk_usage(ROOT))),
    }
    for name, command in {
        "gpu": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"],
        "memory": ["free", "-b"], "shared_memory": ["df", "-B1", "/dev/shm"],
    }.items():
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        environment[name] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    write_json(output / "environment.json", environment)
    records, failures = [], []
    for group, manifest, suffix, dims in (("train_npz", "train.txt", ".npz", 2),
                                           ("test_vol_h5", "test_vol.txt", ".npy.h5", 3)):
        names = (lists / manifest).read_text().split()
        for index, name in enumerate(names):
            path = data / group / (name + suffix)
            try:
                opener = np.load if dims == 2 else lambda p: h5py.File(p, "r")
                with opener(path) as arrays:
                    assert {"image", "label"} <= set(arrays.keys()), "missing fields"
                    image, label = arrays["image"][:], arrays["label"][:]
                assert image.ndim == dims and image.shape == label.shape and all(image.shape), "shape mismatch"
                assert np.isfinite(image).all() and np.isfinite(label).all(), "NaN/Inf"
                assert np.equal(label, np.floor(label)).all(), "fractional label"
                classes, counts = np.unique(label, return_counts=True)
                assert set(classes.tolist()) <= set(range(9)), "invalid class"
                records.append(dict(split=group, file=str(path.relative_to(ROOT)), shape=list(image.shape),
                                    image_dtype=str(image.dtype), label_dtype=str(label.dtype),
                                    image_min=float(image.min()), image_max=float(image.max()),
                                    class_counts=dict(zip(map(lambda x: str(int(x)), classes), map(int, counts))),
                                    sha256=sha256(path)))
            except Exception as error:
                failures.append(dict(file=str(path), error=repr(error)))
            if (index + 1) % 250 == 0 or index + 1 == len(names):
                print(group, index + 1, "/", len(names), "failures", len(failures), flush=True)
    write_json(output / "data_files.json", records)
    write_json(output / "data_summary.json", dict(
        split=split, checked_files=len(records), failures=failures,
        list_sha256={n: sha256(lists / n) for n in ("train.txt", "test_vol.txt")},
        image_min=min((r["image_min"] for r in records), default=None),
        image_max=max((r["image_max"] for r in records), default=None),
        elapsed_seconds=time.monotonic()-start,
        scope="full stored arrays; not proof of raw CT provenance or organ semantics"))
    params = [torch.nn.Parameter(torch.zeros(())) for _ in range(2)]
    opt = torch.optim.AdamW([dict(params=[p], lr=2e-4) for p in params], weight_decay=0.05)
    scheduler = TwoStageWarmupPolySchedule(opt, 1, [500, 1000], 30360, 0.9)
    rows = []
    for step in (0, 1, 499, 500, 501, 1499, 1500, 1501, 30359, 30360):
        scheduler.last_epoch = step
        rows.append(dict(step=step, backbone=scheduler.get_lr()[0], head=scheduler.get_lr()[1]))
    write_json(output / "schedule_boundaries.json", dict(scope="CPU schedule probe, not observed training", rows=rows))
    if failures:
        raise RuntimeError(f"{len(failures)} data files failed")
    print("Preflight complete:", output, flush=True)


# Opt-in diagnostic callback for temporary audit CLI overrides only.
from lightning.pytorch.callbacks import Callback


class RuntimeAudit(Callback):
    def __init__(self, output_dir: str, reference_csv: str = ""):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.reference_csv = reference_csv
        self.steps, self.cases, self.resources = [], [], []
        self.started = time.monotonic()

    def _sample(self, label):
        import torch
        pending, seen, rss = [os.getpid()], set(), 0
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                proc = Path("/proc") / str(pid)
                for line in (proc / "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss += int(line.split()[1]) * 1024
                pending.extend(map(int, (proc / "task" / str(pid) / "children").read_text().split()))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        available = next(int(line.split()[1])*1024 for line in
                         Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
        row = dict(label=label, elapsed_seconds=time.monotonic()-self.started,
                   process_tree_rss_sum_bytes=rss, host_available_bytes=available,
                   shared_memory_used_bytes=shutil.disk_usage("/dev/shm").used)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            row.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        self.resources.append(row)

    def on_fit_start(self, trainer, pl_module):
        import torch
        torch.cuda.reset_peak_memory_stats()
        self._sample("fit_start")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        import torch
        grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
        finite = bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
        assert finite, "nonfinite unscaled gradient"
        self.steps.append(dict(global_step_before_update=trainer.global_step,
                               finite_unscaled_gradients=finite, gradient_tensors=len(grads)))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        import torch
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        assert torch.isfinite(loss).all(), "nonfinite loss"
        self.steps[-1].update(loss=float(loss.detach()), scaler=trainer.precision_plugin.scaler.get_scale(),
                              optimizer_state_entries=len(trainer.optimizers[0].state))
        self._sample(f"train_batch_{batch_idx}_end")

    def on_fit_end(self, trainer, pl_module):
        import torch
        self._sample("fit_end_after_test")
        write_json(self.output / "runtime_fit.json", dict(
            steps=self.steps, resources=self.resources,
            dynamo_counters={k: dict(v) for k, v in torch._dynamo.utils.counters.items()},
            note="RSS sum double-counts shared pages; samples are not continuous RAM peak; timings include audit overhead"))

    def on_exception(self, trainer, pl_module, exception):
        write_json(self.output / "runtime_failure.json", dict(
            error=repr(exception), steps=self.steps, resources=self.resources))

    def on_test_start(self, trainer, pl_module):
        import torch
        torch.cuda.reset_peak_memory_stats()
        checkpoint = torch.load(trainer.ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        saved, actual = checkpoint["state_dict"], pl_module.state_dict()
        missing, extra = sorted(actual.keys()-saved.keys()), sorted(saved.keys()-actual.keys())
        mismatch = [k for k in actual.keys() & saved.keys() if actual[k].shape != saved[k].shape]
        assert not missing and not extra and not mismatch, (missing, extra, mismatch)
        assert all("_orig_mod" not in k for k in saved), "compiled prefix remains"
        different_values = [k for k in actual if not torch.equal(actual[k].detach().cpu(), saved[k])]
        assert not different_values, different_values
        write_json(self.output / "checkpoint_schema.json", dict(
            path=trainer.ckpt_path, sha256=sha256(trainer.ckpt_path),
            epoch_zero_based=checkpoint["epoch"], global_step=checkpoint["global_step"],
            key_count=len(saved), missing=missing, unexpected=extra, shape_mismatch=mismatch,
            has_optimizer_states="optimizer_states" in checkpoint, value_mismatch=different_values))
        self._sample("test_start_after_reload")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.cases.append(outputs)
        self._sample(f"test_case_{batch_idx}_end")

    def on_test_end(self, trainer, pl_module):
        import numpy as np
        assert len(self.cases) == 12 and len({r["case"] for r in self.cases}) == 12
        comparison = None
        if self.reference_csv:
            with Path(self.reference_csv).open() as stream:
                original = {r["case"]: r for r in csv.DictReader(stream)}
            assert set(original) == {r["case"] for r in self.cases}
            differences = [abs(v-float(original[r["case"]][f"{metric}_class_{i}"]))
                           for r in self.cases for metric in ("dice", "hd95")
                           for i, v in enumerate(r[metric+"_per_class"], 1)]
            comparison = dict(max_absolute_difference=max(differences), reference=self.reference_csv)
            comparison["within_1e_10"] = bool(np.allclose(differences, 0, atol=1e-10, rtol=0))
        write_json(self.output / "runtime_test.json", dict(
            cases=self.cases, resources=self.resources, comparison=comparison,
            mean_dice=float(np.mean([r["mean_dice"] for r in self.cases])),
            mean_hd95=float(np.mean([r["mean_hd95"] for r in self.cases]))))
        if comparison is not None:
            assert comparison["within_1e_10"], comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    preflight(parser.parse_args().output_dir)
