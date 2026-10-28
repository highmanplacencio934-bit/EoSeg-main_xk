#!/usr/bin/env python3
"""Independent G1/G2/G3 checks for the opt-in deterministic loss protocol.

This script does not import or alter the production fit/test entry point. GPU
checks must be scheduled serially with the other controlled diagnostic jobs.
Thresholds are registered on disk before torch is imported or tests are run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = "ce2d_stableseg_v1"
# Mathematical agreement is separate from exact candidate repeatability.
THRESHOLDS = {
    "protocol": PROTOCOL,
    "version": 1,
    "repeatability": {"minimum_repeats": 30, "tensor_hashes": "exact_equal"},
    "cpu_float64": {"atol": 1e-8, "rtol": 1e-6},
    "float32": {"atol": 1e-5, "rtol": 1e-4},
    "float16": {"atol": 1e-3, "rtol": 3e-3},
    "gradcheck": {"eps": 1e-6, "atol": 1e-5, "rtol": 1e-3},
    "explanation": {
        "cpu_float64": "Double reference and custom input gradient; small finite interpolation sums.",
        "float32": "Allows FP32 reduction rounding versus CPU double; candidate repeats still require exact hashes.",
        "float16": "FP16 unit roundoff is about 4.88e-4; 0.3% relative and 1e-3 absolute cover final half cast and tiny values.",
        "amp_float32": "Autocast FP32 outputs/leaf gradients use float32 thresholds, not the looser half threshold.",
        "native_forward": "Sampling uses native forward, so matching native output hashes is mandatory regardless of tolerance.",
        "coordinates": "CPU reference consumes the actually rounded 2*coords-1 grid, cast to double, not remapped double coords.",
    },
}


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def register_thresholds(output: Path) -> None:
    path = output / "validation_thresholds.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != THRESHOLDS:
            raise RuntimeError("Existing thresholds differ; use a new protocol/output directory, do not relax a completed test.")
    else:
        json_write(path, THRESHOLDS)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: Any) -> str:
    value = value.detach().cpu().contiguous()
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def digest_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_hash(value)}
    if isinstance(value, dict):
        return {str(key): digest_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [digest_tree(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    return value


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(digest_tree(value), sort_keys=True, default=str).encode()).hexdigest()


def rng_state() -> dict:
    result = {"cpu": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
    if torch.cuda.is_initialized():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state: dict, device: str) -> None:
    torch.set_rng_state(state["cpu"].cpu())
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if device.startswith("cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def math_comparison(actual: Any, expected: Any, *, threshold: str | None = None) -> dict:
    dtype = actual.dtype
    key = threshold or ("float16" if dtype == torch.float16 else "cpu_float64" if dtype == torch.float64 else "float32")
    tol = THRESHOLDS[key]
    a = actual.detach().cpu().to(torch.float64)
    b = expected.detach().cpu().to(torch.float64)
    difference = (a - b).abs()
    allowed = tol["atol"] + tol["rtol"] * b.abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    passed = finite and bool(torch.all(difference <= allowed))
    result = {
        "passed": passed, "threshold": key, **tol,
        "actual_dtype": str(dtype), "reference_dtype": str(expected.dtype),
        "shape": list(actual.shape), "finite": finite,
        "max_abs_error": float(difference.max()) if difference.numel() else 0.0,
        "max_error_over_allowed": float((difference / allowed).max()) if difference.numel() else 0.0,
        "violating_elements": int((difference > allowed).sum()),
    }
    if not passed:
        raise AssertionError(json.dumps(result))
    return result


def strict_environment(device: str) -> dict:
    if device.startswith("cuda"):
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("GPU protocol requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")
        if os.environ.get("PYTHONHASHSEED") != "0":
            raise RuntimeError("GPU protocol requires PYTHONHASHSEED=0 before Python starts")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no fallback to CPU")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("medium")
    torch.set_num_threads(1)
    report = {
        "torch": torch.__version__, "python": sys.version,
        "device": device, "strict_deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "threads": torch.get_num_threads(),
        "environment": {key: os.environ.get(key) for key in (
            "HF_HUB_OFFLINE", "TORCH_HOME", "TIMM_USE_OLD_CACHE", "CUBLAS_WORKSPACE_CONFIG",
            "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")},
    }
    if device.startswith("cuda"):
        report.update(gpu=torch.cuda.get_device_name(), cuda_runtime=torch.version.cuda)
    return report


def gpu_admission(min_free_mib: int) -> dict:
    rows = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free,memory.total,driver_version", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    free, total = int(rows[0].strip()), int(rows[1].strip())
    if free < min_free_mib:
        raise RuntimeError(f"Insufficient free GPU memory: {free} MiB, require {min_free_mib} MiB including headroom")
    return {"free_mib": free, "total_mib": total, "driver_version": rows[2].strip(), "required_mib": min_free_mib}


def criterion_new(device: str):
    from training.deterministic_mask_loss import DeterministicMaskClassificationLoss
    return DeterministicMaskClassificationLoss(12544, 3.0, .75, 5.0, 5.0, 2.0, 9, .1).to(device)


def native_sample(features, coordinates):
    add_dimension = coordinates.ndim == 3
    grid = coordinates.unsqueeze(2) if add_dimension else coordinates
    output = F.grid_sample(features, 2.0 * grid - 1.0, mode="bilinear", padding_mode="zeros", align_corners=False)
    return output.squeeze(3) if add_dimension else output


def cpu_reference_sample(features, coordinates, grad_output):
    # Reproduce the actual coordinate mapping rounding before converting to FP64.
    grid = (2.0 * coordinates.detach() - 1.0).cpu().double()
    added = grid.ndim == 3
    if added:
        grid = grid.unsqueeze(2)
    x = features.detach().cpu().double().requires_grad_(True)
    output = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if added:
        output = output.squeeze(3)
    (output * grad_output.detach().cpu().double()).sum().backward()
    return output.detach(), x.grad.detach()


def coords_with_boundaries(batch: int, points: int, device: str, dtype):
    coords = torch.rand(batch, points, 2, device=device, dtype=dtype)
    boundary = torch.tensor([[0., 0.], [1., 1.], [.5, .5], [.5, .5], [-.1, .5], [1.1, .5], [.5, 0.], [.5, 1.]], device=device, dtype=dtype)
    coords[:, :len(boundary)] = boundary
    return coords


def cpu_checks() -> dict:
    from training.deterministic_point_sampling import sample_point
    torch.manual_seed(731)
    report = {"scope": "CPU mathematical oracle; does not establish CUDA repeatability", "sampling": []}
    for non_contiguous in (False, True):
        features = torch.randn(2, 3, 7, 5, dtype=torch.float64)
        if non_contiguous:
            features = features.transpose(2, 3)
        coords = coords_with_boundaries(2, 13, "cpu", torch.float64)
        features = features.detach().requires_grad_(True)
        output = sample_point(features, coords, align_corners=False)
        grad_output = torch.linspace(-.8, 1.2, output.numel(), dtype=output.dtype).reshape(output.shape)
        (output * grad_output).sum().backward()
        expected, expected_grad = cpu_reference_sample(features, coords, grad_output)
        case = {"non_contiguous": non_contiguous,
                "forward": math_comparison(output, expected), "gradient": math_comparison(features.grad, expected_grad)}
        if tensor_hash(output) != tensor_hash(native_sample(features, coords)):
            raise AssertionError("Candidate sampling forward differs from native forward")
        report["sampling"].append(case)
    x = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    coords = torch.tensor([[[.1, .2], [.5, .5], [.5, .5], [0., 0.], [1., 1.], [-.1, .4], [1.1, .8]]], dtype=torch.float64)
    report["gradcheck"] = bool(torch.autograd.gradcheck(lambda value: sample_point(value, coords, align_corners=False), (x,), **THRESHOLDS["gradcheck"]))
    if not report["gradcheck"]:
        raise AssertionError("CPU double gradcheck failed")
    # The actual criterion target construction is checked, including weighted no-object normalization.
    criterion = criterion_new("cpu").double()
    labels = [torch.tensor([0, 3, 8]), torch.tensor([], dtype=torch.long)]
    indices = [(torch.tensor([0, 2, 4]), torch.arange(3)), (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))]
    logits = torch.randn(2, 5, 10, dtype=torch.float64, requires_grad=True)
    actual = criterion.loss_labels(logits, labels, indices)["loss_cross_entropy"]
    target = torch.full((2, 5), 9, dtype=torch.long)
    target[0, [0, 2, 4]] = labels[0]
    expected_logits = logits.detach().clone().requires_grad_(True)
    expected = F.cross_entropy(expected_logits.reshape(-1, 10), target.reshape(-1), weight=criterion.empty_weight)
    actual.backward(); expected.backward()
    report["weighted_ce"] = {"loss": math_comparison(actual, expected), "gradient": math_comparison(logits.grad, expected_logits.grad)}
    report["passed"] = True
    return report


def repeated_run(run, repeats: int, *, device: str) -> tuple[dict, dict]:
    observations = []
    first = None
    times = []
    for index in range(repeats):
        start = time.perf_counter()
        result = run()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        record = {key: digest_tree(value) for key, value in result.items() if key != "math_values"}
        observations.append(record)
        if first is None:
            first = result
        elif record != observations[0]:
            raise AssertionError(f"Repeat {index + 1} differs from repeat 1; exact hashes required")
    return {"passed": True, "repeats": repeats, "identical": True, "first": observations[0],
            "all_repeat_observation_hashes": [canonical_hash(item) for item in observations],
            "seconds": times, "first_seconds": times[0], "warm_median_seconds": float(np.median(times[1:]))}, first


def ce_gpu_case(dtype, compiled: bool, repeats: int, all_no_object: bool, non_contiguous: bool) -> dict:
    criterion = criterion_new("cuda")
    logits = torch.randn(16, 16, 10, device="cuda", dtype=dtype)
    if non_contiguous:
        logits = logits.transpose(0, 1).contiguous().transpose(0, 1)
    logits = logits.detach().requires_grad_(True)
    labels = [torch.tensor([0, 2, 8], device="cuda") for _ in range(16)]
    matches = 0 if all_no_object else 3
    indices = [(torch.arange(matches, device="cuda"), torch.arange(matches, device="cuda")) for _ in range(16)]
    fn = lambda value: criterion.loss_labels(value, labels, indices)["loss_cross_entropy"]
    if compiled:
        fn = torch.compile(fn)
    state = rng_state()
    def run():
        logits.grad = None
        restore_rng(state, "cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = fn(logits)
        loss.backward()
        return {"loss": loss.detach(), "gradient": logits.grad.detach(), "rng_after": digest_tree(rng_state())}
    repeated, first = repeated_run(run, repeats, device="cuda")
    target = torch.full((16, 16), 9, dtype=torch.long)
    if matches:
        target[:, :3] = torch.tensor([0, 2, 8])
    reference_logits = logits.detach().cpu().double().requires_grad_(True)
    reference_loss = F.cross_entropy(reference_logits.reshape(-1, 10), target.reshape(-1), weight=criterion.empty_weight.cpu().double())
    reference_loss.backward()
    repeated.update(dtype=str(dtype), compiled=compiled, all_no_object=all_no_object, non_contiguous=non_contiguous,
                    loss_reference=math_comparison(first["loss"], reference_loss),
                    gradient_reference=math_comparison(first["gradient"], reference_logits.grad))
    return repeated


def sample_gpu_case(dtype, compiled: bool, repeats: int, grid_4d: bool) -> dict:
    from training.deterministic_point_sampling import sample_point
    features = torch.randn(4, 2, 24, 24, device="cuda", dtype=dtype).transpose(2, 3).detach().requires_grad_(True)
    coords = coords_with_boundaries(4, 12544, "cuda", dtype)
    if grid_4d:
        coords = coords.reshape(4, 112, 112, 2)
    fn = lambda value: sample_point(value, coords, align_corners=False)
    if compiled:
        fn = torch.compile(fn)
    with torch.autocast("cuda", dtype=torch.float16):
        first_output = fn(features)
        expected_native = native_sample(features, coords)
    if tensor_hash(first_output) != tensor_hash(expected_native):
        raise AssertionError("Sampling candidate native forward hash mismatch")
    grad_output = torch.linspace(-.8, 1.2, first_output.numel(), device="cuda", dtype=first_output.dtype).reshape(first_output.shape)
    state = rng_state()
    def run():
        features.grad = None
        restore_rng(state, "cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            output = fn(features)
            loss = (output * grad_output).sum()
        loss.backward()
        return {"output": output.detach(), "loss": loss.detach(), "gradient": features.grad.detach(), "rng_after": digest_tree(rng_state())}
    repeated, first = repeated_run(run, repeats, device="cuda")
    expected, expected_grad = cpu_reference_sample(features, coords, grad_output)
    repeated.update(dtype=str(dtype), compiled=compiled, points=12544, grid_4d=grid_4d, non_contiguous=True,
                    native_forward_exact=True, output_reference=math_comparison(first["output"], expected),
                    gradient_reference=math_comparison(first["gradient"], expected_grad))
    return repeated


def operator_checks(repeats: int) -> dict:
    report = {"scope": "G1/G2 isolated CUDA operators, not full model training", "ce": [], "sampling": []}
    torch.manual_seed(731)
    # Every path gets 30 exact repeated losses and gradients, including FP16 autocast/default compile.
    for dtype in (torch.float32, torch.float16):
        for compiled in (False, True):
            for all_no_object in (False, True):
                report["ce"].append(ce_gpu_case(dtype, compiled, repeats, all_no_object, non_contiguous=True))
            for grid_4d in (False, True):
                report["sampling"].append(sample_gpu_case(dtype, compiled, repeats, grid_4d))
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    report["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
    report["passed"] = True
    return report


def to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def observe_criterion(criterion, observations: list) -> None:
    # Instance-local wrappers only; no global Transformers/PyTorch monkeypatch.
    original_match = criterion.matcher.forward
    def match(*args, **kwargs):
        result = original_match(*args, **kwargs)
        observations.append({"kind": "matcher", "indices": digest_tree(result)})
        return result
    criterion.matcher.forward = match
    original_uncertainty = criterion.sample_points_using_uncertainty
    def uncertainty(*args, **kwargs):
        result = original_uncertainty(*args, **kwargs)
        observations.append({"kind": "final_coords", "coords": digest_tree(result)})
        return result
    criterion.sample_points_using_uncertainty = uncertainty
    for owner_name, owner in (("criterion", criterion), ("matcher", criterion.matcher)):
        if hasattr(owner, "_sample_point"):
            original = owner._sample_point
            def sample(*args, _original=original, _owner_name=owner_name, **kwargs):
                result = _original(*args, **kwargs)
                observations.append({"kind": "sample", "owner": _owner_name, "coords": digest_tree(args[1]),
                                     "output": digest_tree(result), "grad_enabled": torch.is_grad_enabled()})
                return result
            owner._sample_point = sample


def replay_checks(path: Path, repeats: int, *, include_native_reference: bool) -> dict:
    from training.mask_classification_loss import MaskClassificationLoss
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kwargs = to_device(payload["kwargs"], "cuda")
    args = to_device(payload.get("args", ()), "cuda")
    if args or not kwargs:
        raise RuntimeError("Replay expects the saved real keyword-based criterion input; refusing inferred bindings")
    masks = kwargs["masks_queries_logits"].detach().requires_grad_(True)
    classes = kwargs["class_queries_logits"].detach().requires_grad_(True)
    kwargs["masks_queries_logits"] = masks
    kwargs["class_queries_logits"] = classes
    report = {"scope": "G3 real saved criterion input; no ViT or optimizer", "input": str(path),
              "input_sha256": file_hash(path), "input_metadata": digest_tree(kwargs), "paths": []}
    native_first = None
    if include_native_reference:
        # Native CUDA backward is an explicit mathematical reference, not a candidate fallback.
        native = MaskClassificationLoss(12544, 3., .75, 5., 5., 2., 9, .1).cuda()
        events = []
        observe_criterion(native, events)
        masks.grad = None; classes.grad = None
        restore_rng(payload["rng"], "cuda")
        torch.use_deterministic_algorithms(False)
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                losses = native(**kwargs)
                total = native.loss_total(losses, lambda *args, **kwargs: None)
            total.backward()
            native_first = {"losses": {key: value.detach().cpu().clone() for key, value in losses.items()},
                            "total": total.detach().cpu().clone(), "mask_gradient": masks.grad.detach().cpu().clone(),
                            "class_gradient": classes.grad.detach().cpu().clone(), "events": list(events), "rng_after": digest_tree(rng_state())}
        finally:
            torch.use_deterministic_algorithms(True, warn_only=False)
        report["native_reference"] = {"strict": False, "reason": "Original known nondeterministic CUDA reference only", "observations": digest_tree(native_first)}
        del native
    for compiled in (False, True):
        criterion = criterion_new("cuda")
        events = []
        observe_criterion(criterion, events)
        call = torch.compile(criterion) if compiled else criterion
        def run():
            masks.grad = None; classes.grad = None
            events.clear()
            restore_rng(payload["rng"], "cuda")
            before_rng = digest_tree(rng_state())
            with torch.autocast("cuda", dtype=torch.float16):
                losses = call(**kwargs)
                total = criterion.loss_total(losses, lambda *args, **kwargs: None)
            total.backward()
            return {"losses": {key: value.detach().cpu().clone() for key, value in losses.items()},
                    "total": total.detach().cpu().clone(), "mask_gradient": masks.grad.detach().cpu().clone(),
                    "class_gradient": classes.grad.detach().cpu().clone(), "events": list(events),
                    "rng_before": before_rng, "rng_after": digest_tree(rng_state())}
        repeated, first = repeated_run(run, repeats, device="cuda")
        repeated.update(compiled_wrapper=compiled, criterion_forward_compiler_disabled=True,
                        compile_note="Original criterion forward is compiler-disabled; this verifies compatibility with the default compiled outer call, not an all-captured loss graph.")
        if native_first is not None:
            shared_events = lambda value: [event for event in value if event["kind"] in ("matcher", "final_coords")]
            if shared_events(first["events"]) != shared_events(native_first["events"]):
                raise AssertionError("Candidate matcher or final uncertainty sampling coordinates differ from native")
            if first["rng_after"] != native_first["rng_after"]:
                raise AssertionError("Candidate criterion consumed a different RNG sequence from native")
            repeated["native_math_comparison"] = {
                "losses": {key: math_comparison(value, native_first["losses"][key]) for key, value in first["losses"].items()},
                "total": math_comparison(first["total"], native_first["total"]),
                "mask_gradient": math_comparison(first["mask_gradient"], native_first["mask_gradient"]),
                "class_gradient": math_comparison(first["class_gradient"], native_first["class_gradient"]),
                "matcher_final_coords_exact": True, "rng_after_exact": True,
            }
        report["paths"].append(repeated)
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    report["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
    report["passed"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "operators", "replay", "all"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--replay-input", type=Path, default=ROOT / "outputs/audit/baseline_repeat_diagnosis/20261027T023508Z/trajectory_old/trajectory_first_criterion_inputs.pt")
    parser.add_argument("--min-free-mib", type=int, default=6144)
    parser.add_argument("--no-native-reference", action="store_true", help="Repeatability only; does not pass native mathematical agreement gate")
    options = parser.parse_args()
    if options.repeats < THRESHOLDS["repeatability"]["minimum_repeats"]:
        parser.error("At least 30 repeats required; no reduced-repeat passed reports")
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    register_thresholds(output)  # Must precede torch imports and all experimental outcomes.
    report_path = output / f"{options.mode}_validation.json"
    if report_path.exists():
        raise RuntimeError(f"Refusing to overwrite prior evidence: {report_path}")
    started = time.time()
    report = {"protocol": PROTOCOL, "mode": options.mode, "thresholds_sha256": file_hash(output / "validation_thresholds.json"),
              "script_sha256": file_hash(Path(__file__)), "started_unix": started, "passed": False}
    try:
        global torch, np, F
        import torch
        import numpy as np
        from torch.nn import functional as F
        device = "cpu" if options.mode == "cpu" else "cuda"
        if device == "cuda":
            report["admission"] = gpu_admission(options.min_free_mib)
        report["environment"] = strict_environment(device)
        report["sources"] = {str(path.relative_to(ROOT)): file_hash(path) for path in (
            ROOT / "training/deterministic_mask_loss.py", ROOT / "training/deterministic_point_sampling.py",
            ROOT / "training/mask_classification_loss.py")}
        if options.mode in ("cpu", "all"):
            report["cpu"] = cpu_checks()
        if options.mode in ("operators", "all"):
            report["operators"] = operator_checks(options.repeats)
        if options.mode in ("replay", "all"):
            report["replay"] = replay_checks(options.replay_input, options.repeats, include_native_reference=not options.no_native_reference)
        report["passed"] = not options.no_native_reference
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report["finished_unix"] = time.time()
    report["wall_seconds"] = report["finished_unix"] - started
    json_write(report_path, report)
    print(json.dumps({"report": str(report_path), "passed": report["passed"], "wall_seconds": report["wall_seconds"], "error": report.get("error", {}).get("message")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
