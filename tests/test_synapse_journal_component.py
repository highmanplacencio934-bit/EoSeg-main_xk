"""Component-launcher safety checks; temp state, mocked processes, CPU only."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import launch_synapse_journal_component as component


class ComponentLauncherSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "outputs/ablations/journal_20261025"
        self.base.mkdir(parents=True)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.temp.cleanup)
        self.stack.enter_context(patch.object(component, "ROOT", self.root))
        self.stack.enter_context(patch.object(component, "BASE", self.base))
        self.popen = self.stack.enter_context(patch.object(
            component.subprocess, "Popen", side_effect=AssertionError("No real process")))
        self.stack.enter_context(patch.object(
            component.subprocess, "check_output", side_effect=AssertionError("No GPU/process probe")))
        self.preflight = self.stack.enter_context(patch.object(
            component, "preflight", side_effect=AssertionError("Unexpected preflight")))
        self.identity = self.stack.enter_context(patch.object(component, "identity", return_value=None))
        self.wait_identity = self.stack.enter_context(patch.object(component, "wait_identity", return_value={
            "pid": 123, "state": "S", "start_ticks": 1, "command": "fake supervisor", "cwd": str(self.root)}))
        self.plan = self.stack.enter_context(patch.object(component, "set_plan_status"))
        self.state = {"active_experiment": None}
        self.registry = {"queue": [{"id": "M0", "status": "completed"},
                                   {"id": "M1", "status": "completed"},
                                   {"id": "S1", "status": "planned"},
                                   {"id": "S2", "status": "planned"}]}
        self.persist()

    def save(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def persist(self):
        self.save(self.base / "orchestration_state.json", self.state)
        self.save(self.base / "experiment_registry.json", self.registry)

    def advance(self, resume=None, may_spawn=False):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            component.advance(resume_test=resume)
        if not may_spawn:
            self.popen.assert_not_called()
        return json.loads(output.getvalue())

    def test_occupied_lock_does_not_start_or_change_registry(self):
        import fcntl
        before = (self.base / "experiment_registry.json").read_bytes()
        with (self.base / "queue.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.advance()["state"], "busy")
        self.assertEqual((self.base / "experiment_registry.json").read_bytes(), before)
        self.preflight.assert_not_called()

    def test_m0_or_m1_incomplete_prevents_components(self):
        self.registry["queue"][1]["status"] = "failed"
        self.persist()
        self.assertEqual(self.advance()["state"], "not_prepared")
        self.assertFalse((self.base / "S1").exists())
        self.preflight.assert_not_called()

    def test_existing_active_is_not_replaced(self):
        self.state["active_experiment"] = {"id": "S1", "status": "training", "run_dir": "retained"}
        self.persist()
        before = (self.base / "orchestration_state.json").read_bytes()
        self.assertEqual(self.advance()["state"], "already_registered")
        self.assertEqual((self.base / "orchestration_state.json").read_bytes(), before)
        self.preflight.assert_not_called()

    def test_failed_first_component_is_not_retrained_or_skipped(self):
        self.registry["queue"][2]["status"] = "failed"
        self.persist()
        self.assertEqual(self.advance()["state"], "needs_review")
        self.assertFalse((self.base / "S2").exists())
        self.preflight.assert_not_called()

    def test_existing_component_receipt_is_not_overwritten(self):
        receipt = self.base / "S1/component_launch_receipt.json"
        self.save(receipt, {"experiment": "S1", "run_dir": "retained", "evidence": "original"})
        before = receipt.read_bytes()
        self.preflight.side_effect = None
        self.preflight.return_value = {}
        self.assertEqual(self.advance()["state"], "already_launched")
        self.assertEqual(receipt.read_bytes(), before)

    def test_unknown_first_component_is_not_skipped(self):
        self.registry["queue"][2]["id"] = "UNSUPPORTED"
        self.persist()
        result = self.advance()
        self.assertEqual(result["state"], "not_prepared")
        self.assertEqual(result["next_experiment"], "UNSUPPORTED")
        self.preflight.assert_not_called()

    def test_explicit_resume_rejects_untrained_planned_run(self):
        with self.assertRaisesRegex(RuntimeError, "resume-test"):
            self.advance(self.base / "S1/does_not_exist")
        self.popen.assert_not_called()
        self.preflight.assert_not_called()

    def deferred(self):
        run = self.base / "S1/retained"
        run.mkdir(parents=True)
        self.registry["queue"][2].update(status="training_completed_tta_deferred",
                                          output_dir=str(run.relative_to(self.root)))
        self.persist()
        self.save(run / "status.json", {"state": "training_completed_tta_deferred"})
        self.save(self.base / "S1/component_launch_receipt.json",
                  {"experiment": "S1", "run_dir": str(run.relative_to(self.root))})
        self.preflight.side_effect = None
        self.preflight.return_value = {}
        return run

    def test_resume_cannot_target_a_different_retained_run(self):
        run = self.deferred()
        before = (run / "status.json").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "first queued deferred"):
            self.advance(self.base / "S2/another")
        self.popen.assert_not_called()
        self.assertEqual((run / "status.json").read_bytes(), before)

    def test_deferred_resume_launches_same_run_test_only(self):
        run = self.deferred()
        receipt = (self.base / "S1/component_launch_receipt.json").read_bytes()
        self.popen.side_effect = None
        self.popen.return_value = SimpleNamespace(pid=123)
        result = self.advance(run, may_spawn=True)
        self.assertEqual(result["state"], "launched")
        self.assertTrue(result["resume_test_only"])
        args = self.popen.call_args.args[0]
        self.assertIn("--test-only", args)
        self.assertIn("--supervise", args)
        self.assertIn(str(run), args)
        self.assertNotIn("fit", args)
        self.assertFalse((run / "fit_launch.json").exists())
        self.assertEqual((self.base / "S1/component_launch_receipt.json").read_bytes(), receipt)

    def test_test_only_supervisor_never_enters_fit_when_gpu_is_busy(self):
        run = self.deferred()
        relative = str(run.relative_to(self.root))
        proof = {"configuration_sha256": "config", "prior_sha256": "prior",
                 "readiness_sha256": "ready", "launcher_sha256": "launcher",
                 "preparation_sha256": "preparation", "shared_launcher_sha256": "shared"}
        self.save(run / "launch.json", {"experiment": "S1", "preflight": proof,
                                        "fit_command": ["DO_NOT_EXECUTE_FIT"]})
        self.save(run / "status.json", {"state": "tta_resume_starting"})
        self.registry["queue"][2]["status"] = "tta_resume_starting"
        self.state["active_experiment"] = {"id": "S1", "run_dir": relative, "supervisor_pid": 321}
        self.persist()
        self.preflight.return_value = proof
        with patch.object(component.os, "getpid", return_value=321), \
             patch.object(component, "validate_training", return_value={"checkpoint": {"epoch": 180}}) as training, \
             patch.object(component, "gpu_free", return_value=False), \
             patch.object(component, "run_child", side_effect=AssertionError("Fit/test child forbidden")) as child:
            self.assertEqual(component.supervise(run, test_only=True), 0)
            training.assert_called_once_with("S1", run)
            self.preflight.assert_called_once_with("S1", require_gpu=False)
            child.assert_not_called()
        self.popen.assert_not_called()
        self.assertFalse((run / "fit_launch.json").exists())
        self.assertEqual(json.loads((run / "status.json").read_text())["state"],
                         "training_completed_tta_deferred")

    def test_component_path_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            component.checked_run(self.root / "outside/S1/run")
        with self.assertRaises(ValueError):
            component.checked_run(self.base / "M0/run")

    def test_nested_trained_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "must not load"):
            component.check_no_checkpoint({"model": {"network": [{"ckpt_path": "trained.ckpt"}]}})
        component.check_no_checkpoint({"model": {"network": [{"ckpt_path": None}]}})


class VoxelTrainingValidationTests(unittest.TestCase):
    def exercise(self, module, experiment, *, enabled=False, fake_zero=False, mixed_mode=False):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            folder = run / "training/synapse_test_selection/fixture"
            folder.mkdir(parents=True)
            def save(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            def save_csv(name, rows):
                with (folder / name).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            names = [f"case{i:04d}" for i in range(1, 13)]
            lists = root / "project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt"
            lists.parent.mkdir(parents=True)
            lists.write_text("\n".join(names) + "\n")
            save(run / "training_exit.json", {"exit_code": 0})
            save(folder / "selection_summary.json", {
                "max_epochs": 220, "first_evaluation_epoch": 171, "last_evaluated_epoch": 220,
                "best_epoch": 171, "best_mean_dice": .5})
            mode = {} if enabled else {"hd95_voxel_enabled": False}
            save(folder / "metric_units.json", {"hd95_unit": "mm", **mode})
            save(folder / "run_manifest.json", {"device": "cuda:0"})
            save_csv("test_epoch_metrics.csv", [{"epoch": e, "mean_dice": .5} for e in range(171, 221)])
            save_csv("test_case_metrics.csv", [
                {"epoch": e, "case": case, "mean_dice": .5, "mean_hd95": 2., **mode}
                for e in range(171, 221) for case in names])
            save_csv("physical_epoch_organ_metrics.csv", [
                {"epoch": e, "label_id": i, "dice": .5, "hd95_mm": 2.,
                 "hd95_voxel": 3. if enabled else None, **mode}
                for e in range(171, 221) for i in range(1, 9)])
            rows = [{"epoch": e, "case": case, "label_id": i, "dice": .5, "hd95_mm": 2.,
                     "hd95_voxel": 3. if enabled else None, **mode}
                    for e in range(171, 221) for case in names for i in range(1, 9)]
            if fake_zero:
                rows[0]["hd95_voxel"] = 0.
            if mixed_mode:
                rows[0]["hd95_voxel_enabled"] = True
            save_csv("physical_case_organ_metrics.csv", rows)
            (folder / "last.ckpt").write_bytes(b"last fixture")
            (folder / "best_test.ckpt").write_bytes(b"best fixture")
            profile = module.PROFILES[experiment]
            hp = {"network": {"class_path": profile["network"], "init_args": profile["flags"]},
                  "_class_path": "training.synapse_physical_semantic.SynapsePhysicalHD95Semantic"}
            if not enabled:
                hp["compute_hd95_voxel"] = False
            checkpoint = {"epoch": 170, "global_step": 23598, "hyper_parameters": hp,
                          "state_dict": {"fixture": torch.ones(1)}}
            def load(path, **kwargs):
                return {"epoch": 219, "global_step": 30360} if Path(path).name == "last.ckpt" else checkpoint
            with patch.object(module, "ROOT", root), patch("torch.load", side_effect=load),                  patch.object(module.subprocess, "Popen") as spawn:
                proof = module.validate_training(experiment, run)
                spawn.assert_not_called()
            return proof

    def launchers(self):
        from scripts import launch_synapse_s2_s3_scheduled
        return [(component, "S1"), (launch_synapse_s2_s3_scheduled, "S2")]

    def test_mm_only_training_and_legacy_hparam_omission_are_accepted(self):
        for module, experiment in self.launchers():
            for enabled in (False, True):
                with self.subTest(module=module.__name__, enabled=enabled):
                    proof = self.exercise(module, experiment, enabled=enabled)
                    self.assertEqual(proof["hd95_voxel_enabled"], enabled)
                    self.assertEqual(proof["case_organ_records"], 4800)

    def test_disabled_training_rejects_fake_zero_and_mixed_row_mode(self):
        for module, experiment in self.launchers():
            for kwargs in ({"fake_zero": True}, {"mixed_mode": True}):
                with self.subTest(module=module.__name__, kwargs=kwargs):
                    with self.assertRaisesRegex(RuntimeError, "Disabled voxel metric|mode mismatch"):
                        self.exercise(module, experiment, **kwargs)


if __name__ == "__main__":
    unittest.main()
