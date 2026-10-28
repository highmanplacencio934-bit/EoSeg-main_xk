"""Independent component fit and same-best eight-view supervisor.

Original training variables remain fixed; one explicit component flag changes.
"""
from __future__ import annotations
import argparse
import copy
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

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "main.py").is_file() and (parent / "AGENTS.md").is_file())
sys.path.insert(0, str(ROOT))
from scripts import launch_synapse_journal_ablation as shared
BASE, PYTHON, ENV = shared.BASE, shared.PYTHON, shared.ENV
read, write, sha, now = shared.read, shared.write, shared.sha, shared.now
voxel_mode, validate_voxel_value = shared.voxel_mode, shared.validate_voxel_value
validate_metric_row, format_voxel = shared.validate_metric_row, shared.format_voxel
validate_voxel_row_mode = shared.validate_voxel_row_mode
identity, wait_identity, alive = shared.identity, shared.wait_identity, shared.alive
environment, set_plan_status = shared.environment, shared.set_plan_status

ASEE = "models.eoseg_asee.EoSegMultiQSegFusionASEE"
CCPS = "models.eoseg_asee_ccps.EoSegMultiQSegFusionASEECCPS"
PROFILES = {
    "S1": {"config": "s1_asee_text_only.yaml", "flags": {"use_text": True, "use_probability": False}},
    "S2": {"config": "s2_asee_probability_only.yaml", "flags": {"use_text": False, "use_probability": True}},
    "S3": {"config": "s3_asee_patch_only.yaml", "flags": {"enhance_patch": True, "enhance_query": False}},
    "S4": {"config": "s4_asee_query_only.yaml", "flags": {"enhance_patch": False, "enhance_query": True}},
    "A1": {"config": "a1_global.yaml", "flags": {"selector_mode": "global", "selector_on_patch": True, "selector_on_query": False, "patch_gate_effect": "both"}},
    "A2": {"config": "a2_visual.yaml", "flags": {"selector_mode": "visual", "selector_on_patch": True, "selector_on_query": False, "patch_gate_effect": "both"}},
    "A10": {"config": "a10_context_only.yaml", "flags": {"selector_mode": "visual_prior", "selector_on_patch": True, "selector_on_query": False, "patch_gate_effect": "context_only"}},
    "A11": {"config": "a11_residual_only.yaml", "flags": {"selector_mode": "visual_prior", "selector_on_patch": True, "selector_on_query": False, "patch_gate_effect": "residual_only"}},
}
for _key, _profile in PROFILES.items():
    _profile.update(network=CCPS if _key.startswith("A") else ASEE, ccps=_key.startswith("A"), asee=True)
    _profile["configuration"] = "configs/synapse/ablations_journal/" + _profile["config"]


def checked_run(run):
    run = Path(run).resolve()
    if not run.is_relative_to(BASE.resolve()) or run.parent.name not in PROFILES:
        raise ValueError("Component run outside allowed roots")
    return run


def gpu_free():
    return not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()


def check_no_checkpoint(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "ckpt_path" and item is not None:
                raise RuntimeError("Component fit must not load a trained checkpoint")
            check_no_checkpoint(item)
    elif isinstance(value, list):
        for item in value:
            check_no_checkpoint(item)


def preflight(experiment, require_gpu=True):
    import yaml
    profile = PROFILES[experiment]
    state = read(BASE / "orchestration_state.json")
    registry = read(BASE / "experiment_registry.json")
    if any(next(x for x in registry["queue"] if x["id"] == key)["status"] != "completed" for key in ("M0", "M1")):
        raise RuntimeError("M0 and M1 must complete first")
    if state["stop"]["status"] != "stopped_by_user":
        raise RuntimeError("Original augmentation run not confirmed stopped")
    ready_path = BASE / "optimization/readiness.json"
    ready = read(ready_path)
    if ready.get("status") != "passed" or ready.get("passed") is not True:
        raise RuntimeError("ROI readiness not passed")
    for name, expected in {**ready["hashes"], **ready["validation_report_hashes"]}.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("ROI evidence/source changed: " + name)
    preparation_path = BASE / "preparation/component_configs.json"
    preparation = read(preparation_path)
    rows = [x for x in preparation["configurations"] if x["id"] == experiment]
    if len(rows) != 1:
        raise RuntimeError("Component configuration is not prepared")
    record = rows[0]
    if record.get("cpu_parse_status") != "passed" or record.get("print_config_exit_code") != 0 or record.get("data_dictionary_identical") is not True:
        raise RuntimeError("Component CPU configuration preparation not passed")
    config_path = ROOT / profile["configuration"]
    if record["configuration"] != profile["configuration"] or sha(config_path) != record["sha256"]:
        raise RuntimeError("Component configuration/hash mismatch")
    source_path = ROOT / record["source_configuration"]
    if sha(source_path) != record["source_sha256"]:
        raise RuntimeError("Original comparison configuration changed")
    config, original = yaml.safe_load(config_path.read_text()), yaml.safe_load(source_path.read_text())
    check_no_checkpoint(config)
    if config.get("seed_everything") != 0 or config.get("tta_views") != 1 or config.get("compile_disabled", False):
        raise RuntimeError("Seed/view/compile training contract changed")
    trainer = config["trainer"]
    if trainer["max_epochs"] != 220 or trainer["precision"] != "16-mixed":
        raise RuntimeError("Training budget/precision changed")
    ignore = {"default_root_dir", "logger", "callbacks"}
    if {k: v for k, v in trainer.items() if k not in ignore} != {k: v for k, v in original["trainer"].items() if k not in ignore}:
        raise RuntimeError("Unexpected scientific trainer override")
    if config["data"] != original["data"] or config["data"]["class_path"] != "datasets.synapse_semantic.SynapseSemantic":
        raise RuntimeError("Original augmentation/data configuration changed")
    for key, expected in {"batch_size": 16, "num_workers": 8, "img_size": [384, 384], "num_classes": 9}.items():
        if config["data"]["init_args"].get(key) != expected:
            raise RuntimeError("Wrong data setting: " + key)
    model = config["model"]
    expected_wrapper = "training.mask_classification_ccps.MaskClassificationSemanticCCPS" if profile["ccps"] else "training.synapse_physical_semantic.SynapsePhysicalHD95Semantic"
    if model["class_path"] != expected_wrapper:
        raise RuntimeError("Wrong explicit physical training wrapper")
    args, source_args = model["init_args"], original["model"]["init_args"]
    for key, expected in {"num_classes": 9, "img_size": [384, 384]}.items():
        if args.get(key) != expected:
            raise RuntimeError("Wrong model dimensions")
    if float(args["lr"]) != 2e-4 or args.get("presence_loss_weight", 0.0) != 0.0:
        raise RuntimeError("Unexpected LR/presence supervision")
    excluded = {"network", "spacing_manifest"}
    if {k: v for k, v in args.items() if k not in excluded} != {k: v for k, v in source_args.items() if k not in excluded}:
        raise RuntimeError("Unexpected model/loss/optimizer setting")
    expected_network = copy.deepcopy(source_args["network"])
    expected_network["init_args"].update(record["requested_component_flags"])
    if args["network"] != expected_network or args["network"]["class_path"] != profile["network"]:
        raise RuntimeError("Network differs beyond declared component")
    for key, expected in profile["flags"].items():
        if args["network"]["init_args"].get(key) != expected:
            raise RuntimeError("Incorrect study flag: " + key)
    parsed = record["parsed_scientific_values"]
    for key, expected in {"lr": 2e-4, "llrd": 0.8, "weight_decay": 0.05, "num_points": 12544, "max_epochs": 220,
                          "precision": "16-mixed", "gradient_clip_val": 0.01, "patch_size": 16,
                          "scale_range": [0.5, 2.0], "color_jitter_enabled": True, "rotation_present": False}.items():
        if parsed.get(key) != expected:
            raise RuntimeError("Resolved scientific setting mismatch: " + key)
    prior = args["network"]["init_args"]["prior_path"]
    if prior != "priors/synapse_asee_v1.pt" or sha(ROOT / prior) != ready["hashes"][prior]:
        raise RuntimeError("Verified training-only prior changed")
    hardware = None
    if require_gpu:
        if not gpu_free():
            raise RuntimeError("GPU busy; do not contend")
        probe = "import json,sys,torch; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); print(json.dumps({'python':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':p.name,'total_memory_bytes':p.total_memory}))"
        hardware = json.loads(subprocess.check_output([PYTHON, "-c", probe], env=environment(), cwd=ROOT, text=True))
        if "5090" not in hardware["gpu"]:
            raise RuntimeError("Unexpected GPU")
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise RuntimeError("Insufficient component-run disk")
    return {"configuration": profile["configuration"], "configuration_sha256": sha(config_path),
            "component_record": record, "preparation_sha256": sha(preparation_path),
            "readiness_sha256": sha(ready_path), "prior_sha256": sha(ROOT / prior),
            "launcher_sha256": sha(Path(__file__)), "shared_launcher_sha256": sha(Path(shared.__file__)), "network_class": profile["network"],
            "model_class": model["class_path"], "seed": 0, "max_epochs": 220,
            "initialization": "original offline timm pretrained flow; no trained segmentation checkpoint",
            "disk_free_bytes": shutil.disk_usage(ROOT).free, "environment": ENV,
            "hardware": hardware, "hostname": subprocess.check_output(["hostname"], text=True).strip()}


def fit_command(experiment, run, proof):
    callback = [{"class_path": "training.synapse_physical_selection.SynapsePhysicalHD95Selection",
                 "init_args": {"last_n_epochs": 50, "expected_test_cases": 12, "output_dir": str(run / "training")}}]
    return [PYTHON, "-u", "scripts/run_synapse_roi.py", "fit", "--config", proof["configuration"],
            "--seed_everything=0", "--tta_views=1", "--trainer.accelerator=gpu", "--trainer.devices=1",
            "--trainer.default_root_dir=" + str(run / "training"), "--trainer.callbacks=" + json.dumps(callback),
            "--trainer.logger.init_args.save_dir=" + str(run / "wandb"),
            "--trainer.logger.init_args.name=journal_" + experiment.lower() + "_seed0_" + run.name]


def validate_training(experiment, run):
    import torch
    profile = PROFILES[experiment]
    training = read(run / "training_exit.json")
    if training["exit_code"] != 0:
        raise RuntimeError("Training did not exit successfully")
    folders = list((run / "training" / "synapse_test_selection").glob("*"))
    if len(folders) != 1:
        raise RuntimeError("Expected one training selection output")
    folder = folders[0]
    voxel_enabled = voxel_mode(folder)
    selection = read(folder / "selection_summary.json")
    if selection["max_epochs"] != 220 or selection["first_evaluation_epoch"] != 171 or selection["last_evaluated_epoch"] != 220:
        raise RuntimeError("Incomplete 220/final50 training budget")
    with (folder / "test_epoch_metrics.csv").open(newline="", encoding="utf-8") as stream:
        epochs = list(csv.DictReader(stream))
    with (folder / "test_case_metrics.csv").open(newline="", encoding="utf-8") as stream:
        cases = list(csv.DictReader(stream))
    with (folder / "physical_epoch_organ_metrics.csv").open(newline="", encoding="utf-8") as stream:
        organs = list(csv.DictReader(stream))
    with (folder / "physical_case_organ_metrics.csv").open(newline="", encoding="utf-8") as stream:
        case_organs = list(csv.DictReader(stream))
    expected = set((ROOT / "project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt").read_text().splitlines())
    if [int(x["epoch"]) for x in epochs] != list(range(171, 221)):
        raise RuntimeError("Missing or duplicate candidate epochs")
    if len(cases) != 600 or len(organs) != 400 or len(case_organs) != 4800:
        raise RuntimeError("Incomplete final50 CSV records")
    for epoch in range(171, 221):
        cr = [x for x in cases if int(x["epoch"]) == epoch]
        oc = [x for x in organs if int(x["epoch"]) == epoch]
        co = [x for x in case_organs if int(x["epoch"]) == epoch]
        if len(cr) != 12 or {x["case"] for x in cr} != expected:
            raise RuntimeError("Incomplete/duplicate epoch cases")
        if len(oc) != 8 or {int(x["label_id"]) for x in oc} != set(range(1, 9)):
            raise RuntimeError("Incomplete/duplicate epoch organs")
        if len(co) != 96 or {(x["case"], int(x["label_id"])) for x in co} != {(c, i) for c in expected for i in range(1, 9)}:
            raise RuntimeError("Incomplete case/organ epoch pairs")
    for row in cases + organs + case_organs:
        validate_voxel_row_mode(row, voxel_enabled)
        for key, value in row.items():
            if key == "hd95_voxel":
                validate_voxel_value(value, voxel_enabled)
                continue
            if key in {"mean_dice", "mean_hd95", "dice", "hd95_mm"} or key.startswith("dice_class_") or key.startswith("hd95_class_"):
                number = float(value)
                if not math.isfinite(number) or number < 0 or ((key == "dice" or key == "mean_dice" or key.startswith("dice_class_")) and number > 1):
                    raise RuntimeError("Nonfinite/out-of-range training selection metrics")
    best = folder / "best_test.ckpt"
    last = folder / "last.ckpt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError("Missing original best/last weights")
    final = torch.load(last, map_location="cpu", mmap=True, weights_only=False)
    if final["epoch"] != 219 or final["global_step"] != 30360:
        raise RuntimeError("Last checkpoint did not complete 220 full epochs")
    del final
    ckpt = torch.load(best, map_location="cpu", mmap=True, weights_only=False)
    hp = ckpt["hyper_parameters"]
    if hp.get("compute_hd95_voxel", True) != voxel_enabled:
        raise RuntimeError("Checkpoint HD95 voxel mode differs from selection metadata")
    if hp["network"]["class_path"] != profile["network"] or ckpt["epoch"] + 1 != selection["best_epoch"]:
        raise RuntimeError("Selected checkpoint model/epoch mismatch")
    saved_init = hp["network"]["init_args"]
    for flag, value in profile["flags"].items():
        if saved_init.get(flag) != value:
            raise RuntimeError("Saved checkpoint study flag mismatch: " + flag)
    if hp.get("_class_path") != ("training.mask_classification_ccps.MaskClassificationSemanticCCPS" if profile["ccps"] else "training.synapse_physical_semantic.SynapsePhysicalHD95Semantic"):
        raise RuntimeError("Saved training wrapper mismatch")
    first_best = max(epochs, key=lambda x: float(x["mean_dice"]))
    if int(first_best["epoch"]) != selection["best_epoch"] or float(first_best["mean_dice"]) != selection["best_mean_dice"]:
        raise RuntimeError("Checkpoint was not chosen by original mean Dice")
    checkpoint = {"path": str(best), "sha256": sha(best), "epoch": ckpt["epoch"] + 1,
                  "global_step": ckpt["global_step"], "network_class": profile["network"],
                  "model_state_entry_count": len(ckpt["state_dict"]),
                  "model_tensor_count": sum(torch.is_tensor(x) for x in ckpt["state_dict"].values())}
    del ckpt
    manifest = read(folder / "run_manifest.json")
    if str(manifest.get("device", "")).split(":")[0] != "cuda":
        raise RuntimeError("Training did not use CUDA")
    if read(folder / "metric_units.json")["hd95_unit"] != "mm":
        raise RuntimeError("Training selection did not preserve physical HD95")
    proof = {"selection_folder": str(folder.relative_to(ROOT)), "selection": selection,
             "checkpoint": checkpoint, "last_checkpoint_sha256": sha(last),
             "manifest_sha256": sha(folder / "run_manifest.json"),
             "candidate_epochs": 50, "case_records": 600, "case_organ_records": 4800,
             "hd95_voxel_enabled": voxel_enabled, "validated_at_utc": now()}
    write(run / "training_completion.json", proof)
    return proof


def evaluation_command(run, proof, training):
    callback = [{"class_path": "training.synapse_multiscale_tta.SynapseMultiScaleTTASelection",
                 "init_args": {"expected_test_cases": 12, "output_dir": str(run / "evaluation")}}]
    return [PYTHON, "-u", "scripts/run_synapse_roi.py", "test", "--config", proof["configuration"],
            "--ckpt_path=" + training["checkpoint"]["path"], "--seed_everything=0",
            "--tta_views=8", "--compile_test", "--trainer.accelerator=gpu", "--trainer.devices=1",
            "--trainer.logger=false", "--trainer.callbacks=" + json.dumps(callback),
            "--trainer.default_root_dir=" + str(run / "evaluation")]


def complete_component(run, launch):
    import numpy as np
    from training.synapse_selection import summarize_cases
    evaluation = list((run / "evaluation" / "independent_test").glob("*"))
    if len(evaluation) != 1:
        raise RuntimeError("Expected one evaluation output")
    folder = evaluation[0]
    voxel_enabled = voxel_mode(folder)
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
        validate_metric_row(row, voxel_enabled)
    indexed = {(x["case"], int(x["label_id"])): x for x in case_rows}
    results = [{"case": case, "dice_per_class": [float(indexed[case, c]["dice"]) for c in range(1, 9)],
                "hd95_per_class": [float(indexed[case, c]["hd95_mm"]) for c in range(1, 9)]} for case in expected]
    recomputed = summarize_cases(results, 12, 8)
    for row in organ_rows:
        c = int(row["label_id"])
        if int(row["num_cases"]) != 12 or float(row["dice"]) != recomputed[f"dice_class_{c}"] or float(row["hd95_mm"]) != recomputed[f"hd95_class_{c}"]:
            raise RuntimeError("Organ summary differs from raw cases")
        if voxel_enabled and float(row["hd95_voxel"]) != sum(float(indexed[case, c]["hd95_voxel"]) for case in expected) / 12:
            raise RuntimeError("Voxel organ summary differs")
    physical = read(folder / "physical_test_summary.json")
    voxel_mode(folder, physical)
    if voxel_enabled:
        validate_voxel_value(physical.get("mean_hd95_voxel"), True)
        voxel_mean = sum(float(np.mean([float(indexed[case, c]["hd95_voxel"]) for c in range(1, 9)])) for case in expected) / 12
        if physical["mean_hd95_voxel"] != voxel_mean:
            raise RuntimeError("Final voxel mean differs from raw cases")
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
    if contract != multiscale_contract(8, (384, 384), compute_hd95_voxel=voxel_enabled):
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
    contract.update(ccps=contract["ccps"] if PROFILES[launch["experiment"]]["ccps"] else "not_present",
                    model_profile=PROFILES[launch["experiment"]]["network"])
    write(contract_path, contract)
    manifest["tta_contract"] = contract
    manifest["artifact_sha256"][contract_path.name] = sha(contract_path)
    manifest["artifact_sha256"][archive.name] = sha(archive)
    manifest["model_profile"] = {"experiment": launch["experiment"], "network": PROFILES[launch["experiment"]]["network"],
                                 "asee": PROFILES[launch["experiment"]]["asee"], "ccps": PROFILES[launch["experiment"]]["ccps"],
                                 "metadata_only_adaptation": "model-specific CCPS applicability recorded; predictions unchanged"}
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
               "hd95_voxel_enabled": voxel_enabled,
               "organs": sorted(organ_rows, key=lambda x: int(x["label_id"])),
               "evaluation_folder": str(folder.relative_to(ROOT)), "manifest_sha256": sha(manifest_path),
               "completed_at_utc": now(), "training_performed": True,
               "metric_policy": ("verified ZYX mm and voxel; unchanged legacy empty-mask rule" if voxel_enabled else
                                 "verified ZYX mm only; voxel disabled; unchanged legacy empty-mask rule"),
               "training_completion": str((run / "training_completion.json").relative_to(ROOT)),
               "training_max_epochs": 220, "training_candidate_epochs": list(range(171, 221)),
               "training_exit_code": 0, "scientific_change": PROFILES[launch["experiment"]]["flags"]}
    write(run / "summary.json", summary)
    lines = [f"# {launch['experiment']} 固定8-view结果", "",
             f"独立220轮训练best epoch={checkpoint['epoch']}，12病例；训练完成后固定同一best测试。网络：{checkpoint['network_class']}。",
             "", "| 标签ID | 器官 | Dice（%） | HD95（mm） | HD95（voxel） |",
             "|---|---|---:|---:|---:|"]
    for row in summary["organs"]:
        lines.append(f"| {row['label_id']} | {row['organ_zh']}（{row['organ']}） | {float(row['dice_percent']):.6f} | {float(row['hd95_mm']):.6f} | {format_voxel(row['hd95_voxel'])} |")
    lines.extend([f"| — | 8器官平均 | {summary['mean_dice_percent']:.6f} | {summary['mean_hd95_mm']:.6f} | {format_voxel(summary['mean_hd95_voxel'])} |",
                  "", "原legacy空mask口径保留，不整理背景；本组使用最后50轮1-view选择的同一best，没有按TTA重新选权重。",
                  "新run的通用contract已归档并标记当前模型的CCPS适用性；仅报告元数据适配，原推理和旧文件未修改。", ""])
    (run / "summary.md").write_text("\n".join(lines))
    return summary


def save_runtime(registry, state, experiment, run, status):
    entry = next(x for x in registry["queue"] if x["id"] == experiment)
    entry.update(status=status["state"], output_dir=str(run.relative_to(ROOT)))
    if status.get("state") in ("completed", "failed"):
        entry["finished_at_utc"] = status.get("finished_at_utc")
    write(BASE / "experiment_registry.json", registry)
    state.update(phase=status["state"], active_experiment=None if status["state"] in ("completed", "training_completed_tta_deferred") else
                 {"id": experiment, "run_dir": str(run.relative_to(ROOT)), "supervisor_pid": os.getpid(), "status": status["state"]},
                 last_update_utc=now())
    write(BASE / "orchestration_state.json", state)
    write(run / "status.json", status)


def run_child(command, run, filename, status, state, registry, experiment):
    with (run / filename).open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT)
        status.update(child_pid=child.pid, child_identity=wait_identity(child.pid, "scripts/run_synapse_roi.py"),
                      supervisor_identity=identity(os.getpid()))
        save_runtime(registry, state, experiment, run, status)
        return child.wait()


def supervise(run, test_only=False):
    run = checked_run(run)
    with (BASE / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        launch, status = read(run / "launch.json"), read(run / "status.json")
        experiment = launch["experiment"]
        registry, state = read(BASE / "experiment_registry.json"), read(BASE / "orchestration_state.json")
        entry = next(x for x in registry["queue"] if x["id"] == experiment)
        active = state.get("active_experiment") or {}
        required = "tta_resume_starting" if test_only else "starting"
        receipt = read(BASE / experiment / "component_launch_receipt.json")
        if (status.get("state") != required or entry.get("status") != required or
                active.get("id") != experiment or active.get("run_dir") != str(run.relative_to(ROOT)) or
                active.get("supervisor_pid") != os.getpid() or receipt.get("run_dir") != str(run.relative_to(ROOT))):
            raise RuntimeError("Supervisor intent rejected duplicate/stale run")
        try:
            current = preflight(experiment, require_gpu=not test_only)
            for key in ("configuration_sha256", "prior_sha256", "readiness_sha256", "launcher_sha256", "shared_launcher_sha256", "preparation_sha256"):
                if current[key] != launch["preflight"][key]:
                    raise RuntimeError("Source changed after launch intent: " + key)
            if not test_only:
                status.update(state="training", stage="fit", training_started_at_utc=now())
                write(run / "fit_launch.json", {"command": launch["fit_command"], "preflight": current, "started_at_utc": now()})
                code = run_child(launch["fit_command"], run, "train.log", status, state, registry, experiment)
                write(run / "training_exit.json", {"exit_code": code, "finished_at_utc": now()})
                status["training_exit_code"] = code
                if code != 0:
                    raise RuntimeError("Training exited nonzero; no automatic retraining")
            training = validate_training(experiment, run)
            status["training_completed"] = True
            if not gpu_free():
                status.update(state="training_completed_tta_deferred", stage="awaiting_gpu_for_same_best_test",
                              reason="Other GPU processes active; weights retained; resume-test only", finished_at_utc=now())
                save_runtime(registry, state, experiment, run, status)
                return 0
            command = evaluation_command(run, current, training)
            evaluation_launch = {"experiment": experiment, "command": command,
                                 "preflight": {**current, "checkpoint": training["checkpoint"]},
                                 "training_performed": True, "source_training": training}
            write(run / "evaluation_launch.json", evaluation_launch)
            status.update(state="evaluating_tta", stage="same_best_eight_view", test_started_at_utc=now())
            code = run_child(command, run, "test.log", status, state, registry, experiment)
            status["test_exit_code"] = code
            if code != 0:
                raise RuntimeError("Final eight-view test failed; retain training, no automatic rerun")
            summary = complete_component(run, evaluation_launch)
            registry, state = read(BASE / "experiment_registry.json"), read(BASE / "orchestration_state.json")
            entry = next(x for x in registry["queue"] if x["id"] == experiment)
            entry.update(status="completed", completed_at_utc=summary["completed_at_utc"],
                         checkpoint_sha256=summary["checkpoint"]["sha256"], summary=str((run / "summary.json").relative_to(ROOT)))
            status.update(state="completed", stage="finished", exit_code=0, finished_at_utc=now())
            save_runtime(registry, state, experiment, run, status)
            set_plan_status(experiment, "已完成", str((run / "summary.md").relative_to(ROOT)))
            return 0
        except BaseException as error:
            status.update(state="failed", stage="needs_review", exit_code=1,
                          error=type(error).__name__ + ": " + str(error), finished_at_utc=now())
            save_runtime(registry, state, experiment, run, status)
            set_plan_status(experiment, "失败待核查", str((run / "status.json").relative_to(ROOT)))
            return 1


def advance(resume_test=None):
    with (BASE / "queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"state": "busy", "reason": "existing pipeline owns shared lock"}))
            return
        registry, state = read(BASE / "experiment_registry.json"), read(BASE / "orchestration_state.json")
        if any(next(x for x in registry["queue"] if x["id"] == key)["status"] != "completed" for key in ("M0", "M1")):
            print(json.dumps({"state": "not_prepared", "reason": "M0/M1 must finish; components not started"}))
            return
        remaining = [x for x in registry["queue"] if x["status"] != "completed"]
        if not remaining:
            print(json.dumps({"state": "all_completed"}))
            return
        entry = remaining[0]
        experiment = entry["id"]
        if experiment not in PROFILES:
            print(json.dumps({"state": "not_prepared", "next_experiment": experiment}))
            return
        if state.get("active_experiment") is not None:
            print(json.dumps({"state": "already_registered", "active": state["active_experiment"]}))
            return
        is_resume = entry["status"] == "training_completed_tta_deferred"
        if entry["status"] != "planned" and not is_resume:
            print(json.dumps({"state": "needs_review", "entry": entry}))
            return
        if resume_test is not None and not is_resume:
            raise RuntimeError("Explicit resume-test requires retained deferred training")
        try:
            proof = preflight(experiment, require_gpu=True)
        except (FileNotFoundError, KeyError) as error:
            print(json.dumps({"state": "not_prepared", "next_experiment": experiment, "reason": str(error)}))
            return
        parent = BASE / experiment
        parent.mkdir(exist_ok=True)
        receipt_path = parent / "component_launch_receipt.json"
        if is_resume:
            run = checked_run(ROOT / entry["output_dir"])
            if resume_test is not None and checked_run(resume_test) != run:
                raise RuntimeError("Only the first queued deferred run may resume test")
            status = read(run / "status.json")
            if status["state"] != "training_completed_tta_deferred":
                raise RuntimeError("Deferred-state mismatch")
            write(run / ("status_before_tta_resume_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"), status)
            status.update(state="tta_resume_starting", stage="reserving_same_best_test")
            write(run / "status.json", status)
        else:
            if receipt_path.exists():
                print(json.dumps({"state": "already_launched", "receipt": read(receipt_path)}))
                return
            run = parent / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run.mkdir(exist_ok=False)
            launch = {"experiment": experiment, "created_at_utc": now(), "preflight": proof,
                      "fit_command": fit_command(experiment, run, proof), "max_epochs": 220,
                      "seed": 0, "scientific_change": PROFILES[experiment]["flags"],
                      "output_root": str(run)}
            write(run / "launch.json", launch)
            write(run / "status.json", {"state": "starting", "stage": "reserving_fit", "experiment": experiment})
            shutil.copy2(BASE / "optimization/readiness.json", run / "optimization_readiness_snapshot.json")
            shutil.copy2(BASE / "preparation/component_configs.json", run / "component_configuration_snapshot.json")
            write(receipt_path, {"experiment": experiment, "run_dir": str(run.relative_to(ROOT)), "created_at_utc": now()})
        command = [PYTHON, str(Path(__file__).resolve()), "--supervise", str(run)]
        if is_resume:
            command.append("--test-only")
        supervisor_log = run / ("tta_resume_supervisor.log" if is_resume else "supervisor.log")
        with supervisor_log.open("a" if is_resume else "w") as log:
            worker = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, start_new_session=True)
        owner = wait_identity(worker.pid, "--supervise " + str(run))
        entry.update(status="tta_resume_starting" if is_resume else "starting", output_dir=str(run.relative_to(ROOT)))
        state.update(phase="component_running", active_experiment={"id": experiment, "run_dir": str(run.relative_to(ROOT)),
                     "supervisor_pid": worker.pid, "supervisor_identity": owner}, last_update_utc=now())
        write(BASE / "experiment_registry.json", registry)
        write(BASE / "orchestration_state.json", state)
        (parent / "latest_path.txt").write_text(str(run.relative_to(ROOT)) + "\n")
        set_plan_status(experiment, "运行中", str((run / "status.json").relative_to(ROOT)))
        print(json.dumps({"state": "launched", "experiment": experiment, "resume_test_only": is_resume,
                          "run_dir": str(run.relative_to(ROOT)), "supervisor_pid": worker.pid}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--next", action="store_true")
    parser.add_argument("--supervise", type=Path)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--resume-test", type=Path)
    args = parser.parse_args()
    if "preparation" in Path(__file__).parts:
        raise SystemExit("Draft is intentionally non-executable; stabilize and test under scripts first")
    if args.supervise:
        raise SystemExit(supervise(args.supervise, test_only=args.test_only))
    else:
        advance(args.resume_test)
