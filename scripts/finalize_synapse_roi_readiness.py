"""Finalize ROI readiness only after every required audit has actually passed."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/ablations/journal_20261025"
OUT = BASE / "optimization"
SOURCES = ["training/synapse_roi_metrics.py", "training/synapse_roi_evaluation.py",
           "scripts/run_synapse_roi.py", "scripts/validate_synapse_roi_acceleration.py",
           "scripts/finalize_synapse_roi_readiness.py",
           "tests/test_synapse_roi_metrics.py", "tests/test_synapse_roi_evaluation.py",
           "tests/test_synapse_roi_readiness.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((OUT / name).read_text())


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def _finite_positive(value, name):
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise RuntimeError("Invalid positive finite timing: " + name)


def validate_reports(cpu, gpu, fit, expected_cases):
    import math
    import statistics
    expected = set(expected_cases)
    if len(expected_cases) != 12 or len(expected) != 12:
        raise RuntimeError("Expected fixed twelve-case list")
    if cpu.get("cases") != expected_cases or cpu.get("repeats", 0) < 2:
        raise RuntimeError("CPU repeat/list contract mismatch")
    if cpu.get("num_cpu_workers") != 1 or cpu.get("gpu_used") is not False:
        raise RuntimeError("CPU validation was not the selected single-worker CPU path")
    comparisons = cpu.get("comparisons", [])
    if len(comparisons) != 12 * cpu["repeats"]:
        raise RuntimeError("Incomplete CPU comparisons")
    totals = []
    valid_statuses = {"nonempty", "empty_gt_legacy_zero", "empty_pred_legacy_zero", "both_empty_legacy_zero"}
    for repeat in range(cpu["repeats"]):
        items = [x for x in comparisons if x["repeat"] == repeat]
        if len(items) != 12 or {x["case"] for x in items} != expected:
            raise RuntimeError("Missing or duplicate CPU case")
        sums = {"original": 0.0, "roi": 0.0}
        for item in items:
            if item.get("exact") is not True:
                raise RuntimeError("A CPU comparison was not exact")
            by_mode = {}
            for mode in sums:
                entry = item[mode]
                _finite_positive(entry["cached_case_total_seconds"], mode)
                _finite_positive(entry["all_organs_seconds"], mode)
                values = entry["organs"]
                if len(values) != 8 or {x["label_id"] for x in values} != set(range(1, 9)):
                    raise RuntimeError("Missing or duplicate CPU organ")
                by_mode[mode] = {x["label_id"]: x["metrics"] for x in values}
                for metrics in by_mode[mode].values():
                    if metrics.get("status") not in valid_statuses:
                        raise RuntimeError("Unknown metric status")
                    for key in ("dice", "hd95_mm", "hd95_voxel"):
                        if not math.isfinite(metrics[key]) or metrics[key] < 0:
                            raise RuntimeError("Nonfinite or negative metric")
                    if metrics["dice"] > 1:
                        raise RuntimeError("Dice outside 0..1")
                sums[mode] += entry["cached_case_total_seconds"]
            if by_mode["original"] != by_mode["roi"]:
                raise RuntimeError("CPU report contains unequal actual metric dictionaries")
        totals.append(sums)
    medians = {mode: statistics.median(x[mode] for x in totals) for mode in ("original", "roi")}
    if cpu.get("median_seconds") != medians:
        raise RuntimeError("CPU aggregate timing does not match individual complete cases")
    speedup = medians["original"] / medians["roi"]
    _finite_positive(cpu["speedup"], "speedup")
    if cpu["speedup"] != speedup or speedup <= 1:
        raise RuntimeError("Invalid ROI timing benefit")
    if cpu.get("compared_case_classes") != len(comparisons) * 8:
        raise RuntimeError("Incorrect numeric comparison count")
    results = gpu.get("cases", [])
    if len(results) != 12 or [x["case"] for x in results] != expected_cases:
        raise RuntimeError("Incomplete or out-of-order GPU cases")
    if gpu.get("precision") != "16-mixed" or gpu.get("compile_test") is not True or gpu.get("compiled_forward_observed") is not True:
        raise RuntimeError("Wrong GPU precision/compile path")
    if gpu.get("tta_views") != 8 or gpu.get("num_cases") != 12:
        raise RuntimeError("Wrong GPU TTA contract")
    _finite_positive(gpu["seconds"], "GPU total")
    for row in results:
        if row.get("tta_views") != 8 or row.get("tta_scheme") != "scale_hflip":
            raise RuntimeError("Wrong case TTA scheme")
        for key in ("dice_per_class", "hd95_mm_per_class", "hd95_voxel_per_class"):
            values = row.get(key, [])
            if len(values) != 8 or any(not math.isfinite(v) or v < 0 for v in values):
                raise RuntimeError("Incomplete/nonfinite GPU metrics")
        if any(v > 1 for v in row["dice_per_class"]):
            raise RuntimeError("GPU Dice outside 0..1")
    if fit.get("attempted_train_batches") != 3 or len(fit.get("finite_losses", [])) != 3 or any(not math.isfinite(x) for x in fit["finite_losses"]):
        raise RuntimeError("Incomplete/nonfinite fit smoke")
    for key, value in {"batch_size": 16, "num_workers": 8, "img_size": [384, 384], "patch_size": 16, "precision": "16-mixed", "compile_disabled": False, "evaluation_cases": 12}.items():
        if fit.get(key) != value:
            raise RuntimeError("Fit contract changed: " + key)
    _finite_positive(fit["seconds"], "fit smoke time")


def finalize():
    with (BASE / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = read("original_source_hashes.json")
        changed = [name for name, expected in original.items() if sha(ROOT / name) != expected]
        if changed:
            raise RuntimeError("Original protected files changed: " + repr(changed))
        cpu = read("cpu_validation.json")
        cpu_exit = read("cpu_validation_exit.json")
        gpu = read("gpu_eight_view_validation.json")
        gpu_exit = read("gpu_eight_view_validation_exit.json")
        fit = read("gpu_fit_smoke.json")
        fit_exit = read("gpu_fit_smoke_exit.json")
        replay = read("capture_replay_verification.json")
        regress = read("cpu_regression_exit.json")
        expected_cases = (ROOT / "project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt").read_text().splitlines()
        validate_reports(cpu, gpu, fit, expected_cases)
        import csv
        reference_path = ROOT / "outputs/tta/innovation2_best_scale_hflip8/20261024T123754Z/independent_test/20261024T123808Z_bfa7fe37/physical_case_organ_metrics.csv"
        reference = {(x["case"], int(x["label_id"])): x for x in csv.DictReader(reference_path.open())}
        if len(reference) != 96:
            raise RuntimeError("Historical eight-view reference incomplete")
        for row in gpu["cases"]:
            for label in range(1, 9):
                old = reference[(row["case"], label)]
                for new_key, old_key in [("dice_per_class", "dice"), ("hd95_mm_per_class", "hd95_mm"), ("hd95_voxel_per_class", "hd95_voxel")]:
                    if row[new_key][label - 1] != float(old[old_key]):
                        raise RuntimeError("GPU metrics do not exactly match the historical reference")
        frozen = read("validated_implementation_hashes.json")
        for name, expected in frozen["hashes"].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError("Tested implementation changed after audits: " + name)
        if not all(x["status"] == "passed" for x in (cpu, gpu, fit, replay)):
            raise RuntimeError("A validation stage has not passed")
        if any(x["exit_code"] != 0 for x in (cpu_exit, gpu_exit, fit_exit, regress)):
            raise RuntimeError("A validation command exited nonzero")
        if len(cpu["cases"]) != 12 or cpu["repeats"] < 2 or cpu["compared_case_classes"] < 192:
            raise RuntimeError("Incomplete repeated CPU full-case validation")
        if len(cpu["comparisons"]) != 12 * cpu["repeats"] or not all(x["exact"] for x in cpu["comparisons"]):
            raise RuntimeError("Incomplete exact comparisons")
        if gpu["num_cases"] != 12 or gpu["tta_views"] != 8 or not gpu["compiled_forward_observed"]:
            raise RuntimeError("GPU integration scope incomplete")
        if fit["attempted_train_batches"] != 3 or fit["evaluation_cases"] != 12:
            raise RuntimeError("Fit integration smoke incomplete")
        if replay["num_cases"] != 12 or replay["exact_dice_values"] != 96:
            raise RuntimeError("Original prediction replay incomplete")
        if regress["tests"] < 212:
            raise RuntimeError("Incomplete CPU regression suite")
        if cpu["speedup"] <= 1:
            raise RuntimeError("ROI did not improve repeated complete CPU evaluation")
        hashes = {**original, **{name: sha(ROOT / name) for name in SOURCES}}
        provenance_names = ["priors/synapse_asee_v1.pt", "priors/synapse_spacing.verified.json", "priors/synapse_label_map.verified.json",
                            "project_TransUNet/TransUNet/lists/lists_Synapse/train.txt",
                            "project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt"]
        hashes.update({name: sha(ROOT / name) for name in provenance_names})
        reports = ["cpu_validation.json", "cpu_validation_exit.json", "gpu_eight_view_validation.json",
                   "gpu_eight_view_validation_exit.json", "gpu_fit_smoke.json",
                   "gpu_fit_smoke_exit.json", "cpu_regression_exit.json", "capture_replay_verification.json"]
        report_hashes = {str((OUT / name).relative_to(ROOT)): sha(OUT / name) for name in reports}
        ready = {"schema_version": 1, "status": "passed", "passed": True,
                 "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                 "entry_point": "scripts/run_synapse_roi.py",
                 "implementation": "union ROI + halo1 + original MedPy; single CPU worker",
                 "cpu_workers": 1, "cpu_parallel_pool_enabled": False,
                 "geometry": "full-case validation before ROI; verified ZYX spacing",
                 "empty_policy": "unchanged user-selected legacy",
                 "prediction_and_training_variables_changed": False,
                 "original_source_hashes_unchanged": len(original),
                 "hashes": hashes, "validation_report_hashes": report_hashes,
                 "cpu_regression_tests": regress["tests"], "new_tests": regress["tests"] - 153, "new_roi_tests": 38,
                 "cpu_complete_case_repeats": cpu["repeats"],
                 "exact_compared_case_classes": cpu["compared_case_classes"],
                 "cached_cpu_evaluation_median_seconds": cpu["median_seconds"],
                 "cached_cpu_evaluation_speedup": cpu["speedup"],
                 "gpu_compiled_fp16_twelve_case_eight_view_seconds": gpu["seconds"],
                 "gpu_eight_view_exact_numeric_values": 288,
                 "gpu_fit_smoke": "3 attempted training batches plus complete 12-case selection",
                 "timing_limits": "CPU pair timings include fresh cache I/O and full geometry, exclude GPU. GPU full eight-view time measured with ROI only; historical old full-run timing is not a paired repeated GPU benchmark.",
                 "current_augmentation_run_restarted": False, "ablation_started": False}
        existing = OUT / "readiness.json"
        if existing.exists() and json.loads(existing.read_text()).get("status") == "passed":
            old = json.loads(existing.read_text())
            if old["hashes"] != hashes:
                raise RuntimeError("Passed readiness exists for different source hashes")
            print("Readiness already passed for unchanged sources")
            return
        state_path = BASE / "orchestration_state.json"
        state = json.loads(state_path.read_text())
        state["phase"] = "ready_for_scheduled_ablations"
        state["optimization"].update(status="passed", finished_at_utc=ready["finished_at_utc"],
                                     readiness_path=str(existing.relative_to(ROOT)))
        state["last_update_utc"] = ready["finished_at_utc"]
        write(state_path, state)
        plan_path = ROOT / ".codex/synapse_journal_ablation_plan.md"
        text = plan_path.read_text()
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("| 前置 | OPT |"):
                lines[index] = "| 前置 | OPT | 指标等价加速及完整回归 | 已完成 | outputs/ablations/journal_20261025/optimization/readiness.json；完整结果见同目录summary.md |"
        plan_path.write_text("\n".join(lines) + "\n")
        old_seconds, new_seconds = cpu["median_seconds"]["original"], cpu["median_seconds"]["roi"]
        summary = [
            "# Synapse ROI评估加速验证结果", "",
            "全部验证通过；仅未来独立入口启用。原默认入口、模型、训练超参数、增强、种子、数据、预测和评分政策均保持。",
            "", f"- 原有{len(original)}份源码/配置/清单hash全部一致；{regress['tests']}项CPU回归通过，含38项新增ROI指标/接口测试及额外就绪门槛测试。",
            f"- 同一保存预测的12病例×8器官、{cpu['repeats']}轮逐值比较全部一致；无容差放宽。",
            f"- 完整缓存CPU评估中位数：原{old_seconds:.3f}秒，ROI {new_seconds:.3f}秒，{cpu['speedup']:.2f}倍。",
            f"- 真实GPU FP16/compile固定8-view完整12例：{gpu['seconds']:.3f}秒，288个数值精确匹配旧同权重历史CSV。",
            "- 默认FP16/compile、batch16/workers8三步fit冒烟及完整12例选择链路通过；仅审计覆盖max_epochs=1，不是正式220轮或消融成绩。",
            "- 固定历史best187的1-view捕获96个Dice精确匹配原在线CSV，预测来源权重hash见capture_replay_verification.json。",
            "", "## 计时与范围", "",
            "CPU原/ROI各自重新加载相同缓存、核验完整几何并执行全部器官指标，交替顺序重复。此加速比只用于该CPU完整评估阶段；GPU推理未计入。",
            "真实GPU完整8-view时间包含推理和ROI指标，但没有重新测量原入口的配对重复GPU完整耗时，不把局部CPU倍数当作整场训练倍数。",
            "本次选择已充分验证的单进程ROI；没有启用2-worker指标池或跨病例流水。",
            "", "## 使用方式", "",
            "未来独立fit：scripts/run_synapse_roi.py fit --config 独立消融YAML（物理包装类）--tta_views=1；保持原最后50轮1-view选择best。",
            "未来独立test：同入口test --config 对应独立评估YAML --ckpt_path=该模型best --tta_views=8 --compile_test。",
            "正式训练仍为原增强/seed0/220轮，所有参数以独立YAML和实际CLI核验；本次未启动M0/M1。",
            "capture仅用于显式1-view审计，不覆盖缓存；teacher接口独立而且不在此ROI支持范围。",
            "", "## 保留的审计失败", "",
            "第一次GPU审计把CLI limit_test_batches=1误当作整数1；实际解析为1.0并完成全部12例，随后审计的单例断言失败。原日志与退出1保留，未改评分或生产推理；后续改为明确完整12例的审计后通过。",
            "", "正式消融以readiness.json的源码与验证hash为门槛。已完成本阶段，M0等待16:30起的预定推进任务。",
        ]
        (OUT / "summary.md").write_text("\n".join(summary) + "\n")
        book = ROOT / ".codex/synapse_ablation_scheduled_runbook.md"
        body = book.read_text()
        body += f"\n## 15:30优化验证完成\n\n{ready['finished_at_utc']}：独立入口scripts/run_synapse_roi.py验证通过；CPU完整缓存评估{cpu['speedup']:.2f}倍，12例8类精确一致，{regress['tests']}项CPU回归、完整GPU8-view和三步fit链路通过。单CPU ROI启用，未实现并行池/流水，不改变原入口。详见{str((OUT/'summary.md').relative_to(ROOT))}。本阶段未启动M0/M1。\n"
        book.write_text(body)
        agents = ROOT / "AGENTS.md"
        body = agents.read_text()
        body += "\n## 2026-10-25 独立ROI评估入口已验证\n\n- 后续消融使用scripts/run_synapse_roi.py，原main.py及原指标默认路径保留。fit仍1-view选择best，test固定8-view；要求明确物理包装类与verified spacing，teacher接口不支持此ROI范围。\n- 优化门槛为outputs/ablations/journal_20261025/optimization/readiness.json及hash，详情summary.md。使用单CPU联合前景ROI+halo1调用原MedPy，未改变原legacy空mask、Dice、mm/voxel或科学训练变量。不要把CPU阶段加速倍数声称为整场训练倍数；未启用CPU并行池/流水。\n"
        agents.write_text(body)
        # Publish the passed marker last, after all reports and state/docs are persisted.
        write(existing, ready)
        print(json.dumps({"status": "passed", "cpu_speedup": cpu["speedup"],
                          "readiness": str(existing)}, ensure_ascii=False))


if __name__ == "__main__":
    finalize()
