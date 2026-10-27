"""Idempotent journal-ablation launcher: existing M0/M1 checkpoints, fixed 8-view.

Original model/training/configuration files stay unchanged. Components require
separate prepared configurations; this launcher never invents or skips them.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "outputs/ablations/journal_20261025"
PYTHON = "/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python"
ENV = {"HF_HUB_OFFLINE": "1", "TORCH_HOME": "/Disk/HDD/Sk/.cache/torch",
       "TIMM_USE_OLD_CACHE": "1", "WANDB_MODE": "offline"}
PROFILES = {
    "M0": {"network": "models.eoseg.EoSegMultiQSegFusion", "asee": False,
           "epoch": 177, "source": "outputs/baseline/20261021T132635Z",
           "config": "configs/synapse/ablations_journal/m0_baseline_eight_view.yaml"},
    "M1": {"network": "models.eoseg_asee.EoSegMultiQSegFusionASEE", "asee": True,
           "epoch": 196, "source": "outputs/innovation1/asee_plan/20261022T044551Z",
           "config": "configs/synapse/ablations_journal/m1_asee_eight_view.yaml"},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def environment():
    result = {**os.environ, **ENV}
    result.pop("CUDA_VISIBLE_DEVICES", None)
    return result


def identity(pid):
    try:
        path = Path("/proc") / str(pid)
        raw = (path / "stat").read_text()
        tail = raw[raw.rfind(")") + 2:].split()
        return {"pid": int(pid), "state": tail[0], "start_ticks": int(tail[19]),
                "command": (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
                "cwd": os.readlink(path / "cwd")}
    except (FileNotFoundError, ProcessLookupError):
        return None


def wait_identity(pid, command_fragment):
    for _ in range(40):
        current = identity(pid)
        if current and command_fragment in current["command"]:
            return current
        time.sleep(0.05)
    raise RuntimeError("Could not verify freshly launched process identity")


def alive(owner):
    if not owner:
        return False
    current = identity(owner["pid"])
    return bool(current and current["state"] != "Z" and
                current["start_ticks"] == owner["start_ticks"] and
                current["command"] == owner["command"] and current["cwd"] == str(ROOT))


def checked_root(run):
    run = Path(run).resolve()
    if not run.is_relative_to(BASE.resolve()) or run.parent.name not in PROFILES:
        raise ValueError("Supervisor target outside prepared M0/M1 roots")
    return run


def preflight(experiment):
    import yaml
    profile = PROFILES[experiment]
    state = read(BASE / "orchestration_state.json")
    stop = read(BASE / "stop_report.json")
    if state["stop"]["status"] != "stopped_by_user" or stop["state"] != "stopped_by_user":
        raise RuntimeError("Original run not confirmed stopped")
    for old in (645665, 645616):
        current = identity(old)
        if current and str(ROOT / state["stop"]["target_run"]) in current["command"]:
            raise RuntimeError("Original run still exists")
    readiness_path = BASE / "optimization/readiness.json"
    readiness = read(readiness_path)
    if readiness.get("status") != "passed" or readiness.get("passed") is not True:
        raise RuntimeError("ROI readiness not passed")
    for relative, expected in readiness["hashes"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("Readiness source changed: " + relative)
    for relative, expected in readiness["validation_report_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("Readiness report changed: " + relative)
    config_path = ROOT / profile["config"]
    config = yaml.safe_load(config_path.read_text())
    model = config["model"]
    if model["class_path"] != "training.synapse_physical_semantic.SynapsePhysicalHD95Semantic":
        raise RuntimeError("Explicit physical evaluator required")
    args = model["init_args"]
    if args["network"]["class_path"] != profile["network"] or args["img_size"] != [384, 384] or args["num_classes"] != 9:
        raise RuntimeError("Wrong experiment model profile")
    data = config["data"]
    if data["class_path"] != "datasets.synapse_semantic.SynapseSemantic":
        raise RuntimeError("Original Synapse data interface required")
    for key, expected in {"batch_size": 16, "num_workers": 8, "img_size": [384, 384], "num_classes": 9,
                          "train_split": "train", "test_split": "test_vol"}.items():
        if data["init_args"].get(key) != expected:
            raise RuntimeError("Data configuration changed: " + key)
    if config.get("seed_everything", 0) != 0 or config["trainer"]["precision"] != "16-mixed":
        raise RuntimeError("Wrong seed/precision")
    checkpoints = list((ROOT / profile["source"] / "synapse_test_selection").glob("*/best_test.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError("Expected one original best checkpoint")
    checkpoint = checkpoints[0]
    selection = read(checkpoint.parent / "selection_summary.json")
    if selection["best_epoch"] != profile["epoch"] or selection["last_evaluated_epoch"] != 220:
        raise RuntimeError("Historical selected checkpoint/window mismatch")
    import torch
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    hp = saved["hyper_parameters"]
    network = hp["network"]
    names = [key.removeprefix("_orig_mod.") for key in saved["state_dict"]]
    if network["class_path"] != profile["network"] or saved["epoch"] + 1 != profile["epoch"]:
        raise RuntimeError("Checkpoint network/epoch mismatch")
    if hp.get("_class_path") != "training.mask_classification_semantic.MaskClassificationSemantic":
        raise RuntimeError("Unexpected historical checkpoint training class")
    if any(".asee." in name for name in names) != profile["asee"] or any(".selector." in name for name in names):
        raise RuntimeError("Checkpoint ASEE/CCPS profile mismatch")
    encoder = network["init_args"]["encoder"]["init_args"]
    if encoder["patch_size"] != 16 or encoder["backbone_name"] != "vit_large_patch14_reg4_dinov2":
        raise RuntimeError("Wrong historical backbone/patch")
    checkpoint_info = {"path": str(checkpoint), "sha256": sha(checkpoint),
                       "epoch": saved["epoch"] + 1, "global_step": saved["global_step"],
                       "model_state_entry_count": len(saved["state_dict"]),
                       "model_tensor_count": sum(torch.is_tensor(value) for value in saved["state_dict"].values()), "network_class": network["class_path"]}
    del saved
    compatibility_path = BASE / "preparation/model_compatibility.json"
    compatibility = read(compatibility_path)[experiment]
    if compatibility.get("status") != "passed" or compatibility.get("strict_loading") is not True or compatibility.get("all_tensor_values_equal") is not True:
        raise RuntimeError("CPU strict model compatibility has not passed")
    for key, value in {"config": profile["config"], "config_sha256": sha(config_path),
                       "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": checkpoint_info["sha256"],
                       "epoch": profile["epoch"], "global_step": checkpoint_info["global_step"],
                       "model_class": model["class_path"], "network_class": profile["network"]}.items():
        if compatibility.get(key) != value:
            raise RuntimeError("CPU compatibility evidence mismatch: " + key)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if apps:
        raise RuntimeError("GPU busy; no resource contention")
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError("Insufficient free disk")
    probe = "import json,sys,torch; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); print(json.dumps({'python':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':p.name,'total_memory_bytes':p.total_memory}))"
    gpu = json.loads(subprocess.check_output([PYTHON, "-c", probe], env=environment(), cwd=ROOT, text=True))
    if "5090" not in gpu["gpu"]:
        raise RuntimeError("Unexpected GPU")
    return {"checkpoint": checkpoint_info, "configuration": profile["config"],
            "configuration_sha256": sha(config_path), "readiness_sha256": sha(readiness_path),
            "launcher_sha256": sha(Path(__file__)), "environment": ENV, "hardware": gpu,
            "model_compatibility": compatibility, "model_compatibility_sha256": sha(compatibility_path),
            "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "hostname": subprocess.check_output(["hostname"], text=True).strip()}


def set_plan_status(experiment, status, evidence):
    path = ROOT / ".codex/synapse_journal_ablation_plan.md"
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("|") and ("| " + experiment + " |") in line:
            columns = line.split("|")
            columns[4], columns[5] = " " + status + " ", " " + evidence + " "
            lines[index] = "|".join(columns)
    path.write_text("\n".join(lines) + "\n")


def complete(run, launch):
    import numpy as np
    from training.synapse_selection import summarize_cases
    evaluation = list((run / "evaluation" / "independent_test").glob("*"))
    if len(evaluation) != 1:
        raise RuntimeError("Expected one evaluation output")
    folder = evaluation[0]
    with (folder / "physical_case_organ_metrics.csv").open(newline="", encoding="utf-8") as stream:
        case_rows = list(csv.DictReader(stream))
    with (folder / "physical_epoch_organ_metrics.csv").open(newline="", encoding="utf-8") as stream:
        organ_rows = list(csv.DictReader(stream))
    names = {int(x["label_id"]): x for x in read(ROOT / "priors/synapse_label_map.verified.json")["classes"]}
    expected = (ROOT / "project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt").read_text().splitlines()
    expected_pairs = {(case, label) for case in expected for label in range(1, 9)}
    if len(case_rows) != 96 or {(x["case"], int(x["label_id"])) for x in case_rows} != expected_pairs:
        raise RuntimeError("Incomplete or duplicate case/organ CSV")
    if len(organ_rows) != 8 or {int(x["label_id"]) for x in organ_rows} != set(range(1, 9)):
        raise RuntimeError("Incomplete organ summary")
    epoch = launch["preflight"]["checkpoint"]["epoch"]
    for row in case_rows + organ_rows:
        label = int(row["label_id"])
        if row["organ"] != names[label]["name"] or row["organ_zh"] != names[label]["name_zh"] or int(row["epoch"]) != epoch:
            raise RuntimeError("Organ name/epoch mismatch")
        for key in ("dice", "hd95_mm", "hd95_voxel"):
            value = float(row[key])
            if not math.isfinite(value) or value < 0 or (key == "dice" and value > 1):
                raise RuntimeError("Invalid numeric metric")
    indexed = {(x["case"], int(x["label_id"])): x for x in case_rows}
    results = [{"case": case, "dice_per_class": [float(indexed[case, c]["dice"]) for c in range(1, 9)],
                "hd95_per_class": [float(indexed[case, c]["hd95_mm"]) for c in range(1, 9)]} for case in expected]
    recomputed = summarize_cases(results, 12, 8)
    for row in organ_rows:
        c = int(row["label_id"])
        if int(row["num_cases"]) != 12 or float(row["dice"]) != recomputed[f"dice_class_{c}"] or float(row["hd95_mm"]) != recomputed[f"hd95_class_{c}"]:
            raise RuntimeError("Organ summary differs from raw cases")
        if float(row["hd95_voxel"]) != sum(float(indexed[case, c]["hd95_voxel"]) for case in expected) / 12:
            raise RuntimeError("Voxel organ summary differs")
    physical = read(folder / "physical_test_summary.json")
    if physical.get("num_cases") != 12 or physical.get("tta_views") != 8 or physical.get("tta_scheme") != "scale_hflip" or physical.get("scales") != [1.0, 0.875, 1.125, 0.75]:
        raise RuntimeError("Wrong final physical/TTA summary scope")
    if physical["mean_dice"] != recomputed["mean_dice"] or physical["mean_hd95_mm"] != recomputed["mean_hd95"]:
        raise RuntimeError("Final mean differs")
    checkpoint = launch["preflight"]["checkpoint"]
    source = read(folder / "checkpoint_source.json")
    for key, value in {"checkpoint_epoch": checkpoint["epoch"], "checkpoint_global_step": checkpoint["global_step"],
                       "checkpoint_sha256": checkpoint["sha256"], "ckpt_path": checkpoint["path"]}.items():
        if source[key] != value or physical[key] != value:
            raise RuntimeError("Checkpoint source mismatch")
    if sha(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise RuntimeError("Original checkpoint changed")
    contract_path = folder / "tta_contract.json"
    contract = read(contract_path)
    from training.synapse_multiscale_tta import multiscale_contract
    if contract != multiscale_contract(8, (384, 384)):
        raise RuntimeError("Wrong eight-view inference contract")
    manifest_path = folder / "run_manifest.json"
    manifest = read(manifest_path)
    if manifest["tta_contract"] != contract or manifest["checkpoint_source"] != source:
        raise RuntimeError("Manifest source/contract mismatch")
    if manifest["precision"] != "16-mixed" or not manifest["device"].startswith("cuda"):
        raise RuntimeError("Wrong evaluation precision/device")
    if sha(ROOT / launch["preflight"]["configuration"]) != launch["preflight"]["configuration_sha256"]:
        raise RuntimeError("Configuration changed during run")
    archive = folder / "tta_contract.generic.json"
    if archive.exists():
        raise RuntimeError("Contract adaptation already exists; inspect before rewriting")
    archive.write_bytes(contract_path.read_bytes())
    contract.update(ccps="not_present", model_profile=PROFILES[launch["experiment"]]["network"])
    write(contract_path, contract)
    manifest["tta_contract"] = contract
    manifest["artifact_sha256"][contract_path.name] = sha(contract_path)
    manifest["artifact_sha256"][archive.name] = sha(archive)
    manifest["model_profile"] = {"experiment": launch["experiment"], "network": PROFILES[launch["experiment"]]["network"],
                                 "asee": PROFILES[launch["experiment"]]["asee"], "ccps": False,
                                 "metadata_only_adaptation": "generic ccps wording corrected; predictions unchanged"}
    roi_snapshot = run / "optimization_readiness_snapshot.json"
    if sha(roi_snapshot) != launch["preflight"]["readiness_sha256"]:
        raise RuntimeError("Launch ROI snapshot changed")
    manifest["roi_optimization_readiness_snapshot_sha256"] = sha(roi_snapshot)
    manifest["roi_evaluation_implementation"] = "instance-scoped union ROI+halo1 original MedPy; single CPU worker"
    write(manifest_path, manifest)
    summary = {"status": "completed", "experiment": launch["experiment"],
               "checkpoint": checkpoint, "tta_views": 8, "tta_scheme": "scale_hflip",
               "scales": [1.0, 0.875, 1.125, 0.75], "num_cases": 12,
               "mean_dice": physical["mean_dice"], "mean_dice_percent": 100 * physical["mean_dice"],
               "mean_hd95_mm": physical["mean_hd95_mm"], "mean_hd95_voxel": physical["mean_hd95_voxel"],
               "organs": sorted(organ_rows, key=lambda x: int(x["label_id"])),
               "evaluation_folder": str(folder.relative_to(ROOT)), "manifest_sha256": sha(manifest_path),
               "completed_at_utc": now(), "training_performed": False,
               "metric_policy": "verified ZYX mm and voxel; unchanged legacy empty-mask rule"}
    write(run / "summary.json", summary)
    lines = [f"# {launch['experiment']} 固定8-view结果", "",
             f"已有best epoch={checkpoint['epoch']}，12病例；未重新训练。网络：{checkpoint['network_class']}。",
             "", "| 标签ID | 器官 | Dice（%） | HD95（mm） | HD95（voxel） |",
             "|---|---|---:|---:|---:|"]
    for row in summary["organs"]:
        lines.append(f"| {row['label_id']} | {row['organ_zh']}（{row['organ']}） | {float(row['dice_percent']):.6f} | {float(row['hd95_mm']):.6f} | {float(row['hd95_voxel']):.6f} |")
    lines.extend([f"| — | 8器官平均 | {summary['mean_dice_percent']:.6f} | {summary['mean_hd95_mm']:.6f} | {summary['mean_hd95_voxel']:.6f} |",
                  "", "原legacy空mask口径保留，不整理背景；本组固定历史best，没有按TTA重新选权重。",
                  "新run的通用contract中CCPS文案已归档后纠正为not_present；仅报告元数据适配，原推理和旧文件未修改。", ""])
    (run / "summary.md").write_text("\n".join(lines))
    return summary


def supervise(run):
    run = checked_root(run)
    with (BASE / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        launch = read(run / "launch.json")
        experiment = launch["experiment"]
        status = read(run / "status.json")
        registry_path, state_path = BASE / "experiment_registry.json", BASE / "orchestration_state.json"
        registry, state = read(registry_path), read(state_path)
        entry = next(x for x in registry["queue"] if x["id"] == experiment)
        active = state.get("active_experiment") or {}
        receipt = read(BASE / experiment / "launch_receipt.json")
        relative_run = str(run.relative_to(ROOT))
        if (status.get("state") != "starting" or entry.get("status") != "starting" or
                active.get("id") != experiment or active.get("run_dir") != relative_run or
                active.get("supervisor_pid") != os.getpid() or receipt.get("run_dir") != relative_run):
            raise RuntimeError("Supervisor intent guard rejected duplicate/stale invocation; existing artifacts unchanged")
        try:
            verified = preflight(experiment)
            if verified["checkpoint"] != launch["preflight"]["checkpoint"] or verified["configuration_sha256"] != launch["preflight"]["configuration_sha256"]:
                raise RuntimeError("Source changed after launch intent")
            command = launch["command"]
            with (run / "test.log").open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT)
                status.update(state="running", stage="eight_view_evaluation", child_pid=child.pid,
                              child_identity=wait_identity(child.pid, "scripts/run_synapse_roi.py test"), supervisor_identity=identity(os.getpid()),
                              actual_started_at_utc=now())
                write(run / "status.json", status)
                registry = read(registry_path)
                next(x for x in registry["queue"] if x["id"] == experiment).update(status="running", child_pid=child.pid, supervisor_pid=os.getpid())
                write(registry_path, registry)
                state = read(state_path)
                state["active_experiment"].update(stage="eight_view_evaluation", child_pid=child.pid,
                                                   child_identity=status["child_identity"], supervisor_identity=status["supervisor_identity"])
                state["last_update_utc"] = now()
                write(state_path, state)
                code = child.wait()
            status["test_exit_code"] = code
            if code != 0:
                raise RuntimeError(f"Evaluation exited {code}; no automatic restart")
            summary = complete(run, launch)
            registry = read(registry_path)
            entry = next(x for x in registry["queue"] if x["id"] == experiment)
            entry.update(status="completed", completed_at_utc=summary["completed_at_utc"],
                         checkpoint_sha256=summary["checkpoint"]["sha256"],
                         summary=str((run / "summary.json").relative_to(ROOT)))
            write(registry_path, registry)
            state = read(state_path)
            state.update(phase="ready_for_next_ablation", active_experiment=None, last_update_utc=now())
            write(state_path, state)
            set_plan_status(experiment, "已完成", str((run / "summary.md").relative_to(ROOT)))
            status.update(state="completed", stage="finished", exit_code=0, finished_at_utc=now())
        except BaseException as error:
            status.update(state="failed", stage="needs_review", exit_code=1, error=type(error).__name__ + ": " + str(error), finished_at_utc=now())
            registry = read(registry_path)
            entry = next(x for x in registry["queue"] if x["id"] == experiment)
            entry.update(status="failed", error=status["error"])
            write(registry_path, registry)
            state = read(state_path)
            state.update(phase="ablation_failed", active_experiment={"id": experiment, "run_dir": str(run.relative_to(ROOT)), "status": "failed"}, last_update_utc=now())
            write(state_path, state)
            set_plan_status(experiment, "失败待核查", str((run / "status.json").relative_to(ROOT)))
        write(run / "status.json", status)
        return 0 if status["state"] == "completed" else 1


def advance():
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / "queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"state": "busy", "reason": "existing optimization/evaluation owns shared queue lock"}))
            return
        state_path, registry_path = BASE / "orchestration_state.json", BASE / "experiment_registry.json"
        state, registry = read(state_path), read(registry_path)
        if state.get("active_experiment") is not None:
            active = state["active_experiment"]
            print(json.dumps({"state": "already_registered", "active": active}, ensure_ascii=False))
            return
        rows = [x for x in registry["queue"] if x["status"] != "completed"]
        if not rows:
            print(json.dumps({"state": "all_completed"}))
            return
        entry = rows[0]
        if entry["status"] != "planned":
            print(json.dumps({"state": "needs_review", "entry": entry}, ensure_ascii=False))
            return
        experiment = entry["id"]
        if experiment not in PROFILES:
            print(json.dumps({"state": "not_prepared", "next_experiment": experiment, "reason": "This launcher supports existing M0/M1 best checkpoints only; component training requires separate prepared configs and validation."}))
            return
        profile = preflight(experiment)
        parent = BASE / experiment
        parent.mkdir(exist_ok=True)
        receipt_path = parent / "launch_receipt.json"
        if receipt_path.exists():
            print(json.dumps({"state": "already_launched", "receipt": read(receipt_path)}))
            return
        run = parent / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run.mkdir(exist_ok=False)
        callback = [{"class_path": "training.synapse_multiscale_tta.SynapseMultiScaleTTASelection",
                     "init_args": {"expected_test_cases": 12, "output_dir": str(run / "evaluation")}}]
        command = [PYTHON, "-u", "scripts/run_synapse_roi.py", "test", "--config", profile["configuration"],
                   "--ckpt_path=" + profile["checkpoint"]["path"], "--tta_views=8", "--compile_test",
                   "--trainer.accelerator=gpu", "--trainer.devices=1", "--trainer.logger=false",
                   "--trainer.callbacks=" + json.dumps(callback), "--trainer.default_root_dir=" + str(run)]
        launch = {"experiment": experiment, "created_at_utc": now(), "command": command, "preflight": profile,
                  "training_performed": False, "seed": 0, "tta_views": 8, "output_root": str(run)}
        write(run / "launch.json", launch)
        shutil.copy2(BASE / "optimization/readiness.json", run / "optimization_readiness_snapshot.json")
        shutil.copy2(BASE / "preparation/model_compatibility.json", run / "model_compatibility_snapshot.json")
        write(run / "status.json", {"state": "starting", "experiment": experiment, "output_root": str(run), "created_at_utc": now()})
        write(receipt_path, {"experiment": experiment, "run_dir": str(run.relative_to(ROOT)), "created_at_utc": now()})
        with (run / "supervisor.log").open("w") as log:
            supervisor = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), "--supervise", str(run)],
                                          cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, start_new_session=True)
        owner = wait_identity(supervisor.pid, "--supervise " + str(run))
        entry.update(status="starting", output_dir=str(run.relative_to(ROOT)), checkpoint_sha256=profile["checkpoint"]["sha256"])
        state.update(phase="ablation_running", active_experiment={"id": experiment, "run_dir": str(run.relative_to(ROOT)), "supervisor_pid": supervisor.pid, "supervisor_identity": owner}, last_update_utc=now())
        write(registry_path, registry)
        write(state_path, state)
        (parent / "latest_path.txt").write_text(str(run.relative_to(ROOT)) + "\n")
        set_plan_status(experiment, "运行中", str((run / "status.json").relative_to(ROOT)))
        print(json.dumps({"state": "launched", "experiment": experiment, "run_dir": str(run.relative_to(ROOT)), "supervisor_pid": supervisor.pid}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--next", action="store_true")
    parser.add_argument("--supervise", type=Path)
    args = parser.parse_args()
    if args.supervise is not None:
        raise SystemExit(supervise(args.supervise))
    else:
        advance()
