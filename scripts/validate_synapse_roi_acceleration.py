"""CPU-only full-case ROI comparison on identical saved predictions; no training."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from training.synapse_physical_geometry import SpacingManifest, binary_metrics_mm
from training.synapse_roi_metrics import binary_metrics_mm_roi, union_roi_slices


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "outputs/ablations/journal_20261025"
    with (base / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = base / "orchestration_state.json"
        state = json.loads(state_path.read_text())
        if state["stop"]["status"] != "stopped_by_user":
            raise RuntimeError("Original run has not been confirmed stopped")
        if state.get("active_experiment") is not None:
            raise RuntimeError("Another experiment is registered")
        stat = Path(f"/proc/{os.getpid()}/stat").read_text()
        ticks = int(stat[stat.rfind(")") + 2:].split()[19])
        owner = {"pid": os.getpid(), "start_ticks": ticks, "command": sys.argv,
                 "started_at_utc": now(), "stage": "cpu_full_case_validation"}
        write(output / "validation_owner.json", owner)
        state["phase"] = "optimization_validating"
        state["optimization"].update(status="validating", validation_owner=owner)
        state["last_update_utc"] = now()
        write(state_path, state)
        case_list = ROOT / "project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt"
        cases = case_list.read_text().splitlines()
        if len(cases) != 12 or len(set(cases)) != 12:
            raise RuntimeError("Expected exactly 12 unique fixed cases")
        geometry = SpacingManifest("priors/synapse_spacing.verified.json")
        report = {"status": "running", "started_at_utc": now(), "owner": owner,
                  "repeats": args.repeats, "cases": cases, "comparisons": [],
                  "numeric_rule": "exact dictionary equality; no tolerance",
                  "timing_scope": "cached complete 12-case evaluation: disk read, full geometry verification, eight original binary metrics per case; GPU inference excluded",
                  "num_cpu_workers": 1, "gpu_used": False,
                  "numpy": np.__version__, "case_prediction_hashes": {}}
        try:
            raw_order = ("original", "roi")
            for repeat in range(args.repeats):
                for index, case in enumerate(cases):
                    start = time.perf_counter()
                    with np.load(args.cache_dir / f"{case}.npz") as cache:
                        prediction, gt = cache["prediction"], cache["gt"]
                    if prediction.shape != gt.shape or prediction.ndim != 3:
                        raise ValueError(f"Bad cached volume: {case}")
                    if not np.isin(prediction, range(9)).all():
                        raise ValueError(f"Unexpected prediction classes: {case}")
                    verified = geometry.for_case(case, gt)
                    spacing = verified.spacing_zyx_mm
                    signature = {"prediction_sha256": array_hash(prediction),
                                 "gt_sha256": array_hash(gt), "shape": list(gt.shape),
                                 "spacing_zyx_mm": list(spacing)}
                    if case in report["case_prediction_hashes"] and signature != report["case_prediction_hashes"][case]:
                        raise RuntimeError("Cached prediction changed between repeats")
                    report["case_prediction_hashes"][case] = signature
                    load_geometry_seconds = time.perf_counter() - start
                    order = raw_order if (repeat + index) % 2 == 0 else raw_order[::-1]
                    modes = {}
                    for mode in order:
                        fn = binary_metrics_mm if mode == "original" else binary_metrics_mm_roi
                        mode_start = time.perf_counter()
                        # Each timed method reloads and validates the complete cache.
                        with np.load(args.cache_dir / f"{case}.npz") as cache:
                            prediction, gt = cache["prediction"], cache["gt"]
                        geometry.for_case(case, gt)
                        if array_hash(prediction) != signature["prediction_sha256"] or array_hash(gt) != signature["gt_sha256"]:
                            raise RuntimeError("Method input changed within comparison")
                        results = []
                        begin = time.perf_counter()
                        for label in range(1, 9):
                            pred_mask, gt_mask = prediction == label, gt == label
                            tick = time.perf_counter()
                            metrics = fn(pred_mask, gt_mask, spacing, compute_hd95_voxel=True)
                            duration = time.perf_counter() - tick
                            results.append({"label_id": label, "metrics": metrics,
                                            "metric_seconds": duration})
                        modes[mode] = {"organs": results,
                                       "all_organs_seconds": time.perf_counter() - begin,
                                       "cached_case_total_seconds": time.perf_counter() - mode_start}
                    for old, new in zip(modes["original"]["organs"], modes["roi"]["organs"]):
                        if old["metrics"] != new["metrics"]:
                            raise AssertionError(f"Metric mismatch: {case}/class{old['label_id']}: {old['metrics']} != {new['metrics']}")
                    item = {"repeat": repeat, "case": case, "execution_order": order,
                            "load_geometry_seconds": load_geometry_seconds,
                            "original": modes["original"], "roi": modes["roi"],
                            "exact": True, "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                    report["comparisons"].append(item)
                    write(output / "cpu_validation_progress.json", report)
                    print(json.dumps({"repeat": repeat, "case": case, "exact": True,
                                      "old_seconds": modes["original"]["all_organs_seconds"],
                                      "roi_seconds": modes["roi"]["all_organs_seconds"]}), flush=True)
            totals = []
            for repeat in range(args.repeats):
                items = [x for x in report["comparisons"] if x["repeat"] == repeat]
                totals.append({mode: sum(x[mode]["cached_case_total_seconds"] for x in items)
                               for mode in raw_order})
            medians = {mode: statistics.median(x[mode] for x in totals) for mode in raw_order}
            report.update(status="passed", finished_at_utc=now(), repeated_full_case_seconds=totals,
                          median_seconds=medians, speedup=medians["original"] / medians["roi"],
                          compared_case_classes=len(report["comparisons"]) * 8,
                          rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            write(output / "cpu_validation.json", report)
            state["phase"] = "optimization_awaiting_integration_checks"
            state["optimization"].update(status="cpu_validation_passed",
                                          cpu_report=str((output / "cpu_validation.json").relative_to(ROOT)))
        except BaseException as error:
            report.update(status="failed", finished_at_utc=now(),
                          error=type(error).__name__ + ": " + str(error))
            write(output / "cpu_validation_failed.json", report)
            state["phase"] = "optimization_validation_failed"
            state["optimization"].update(status="failed", error=report["error"])
            raise
        finally:
            state["optimization"]["validation_owner"]["finished_at_utc"] = now()
            state["last_update_utc"] = now()
            write(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 2:
        raise SystemExit("At least two alternating complete evaluations required")
    run(args)
