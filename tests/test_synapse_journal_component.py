"""Component-launcher safety checks; temp state, mocked processes, CPU only."""
import contextlib
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


if __name__ == "__main__":
    unittest.main()
