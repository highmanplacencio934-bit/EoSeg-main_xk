"""CPU safety tests for the independent journal launcher; all state is temporary."""
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import launch_synapse_journal_ablation as launcher


class JournalLauncherFixtures:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root / "outputs/ablations/journal_20261025"
        self.base.mkdir(parents=True)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.temporary.cleanup)
        self.stack.enter_context(patch.object(launcher, "ROOT", self.root))
        self.stack.enter_context(patch.object(launcher, "BASE", self.base))
        self.popen = self.stack.enter_context(patch.object(
            launcher.subprocess, "Popen", side_effect=AssertionError("No real process may start")))
        self.stack.enter_context(patch.object(
            launcher.subprocess, "check_output", side_effect=AssertionError("No GPU/process probe")))
        self.preflight = self.stack.enter_context(patch.object(
            launcher, "preflight", side_effect=AssertionError("Unexpected preflight")))
        self.state = {"active_experiment": None}
        self.registry = {"queue": [{"id": "M0", "status": "planned"},
                                   {"id": "M1", "status": "planned"}]}
        self.persist()

    def save(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def persist(self):
        self.save(self.base / "orchestration_state.json", self.state)
        self.save(self.base / "experiment_registry.json", self.registry)

    def advance(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            launcher.advance()
        self.popen.assert_not_called()
        return json.loads(out.getvalue())

class JournalLauncherSafetyTests(JournalLauncherFixtures, unittest.TestCase):
    def test_occupied_queue_lock_does_not_start_or_modify_state(self):
        import fcntl
        before = (self.base / "orchestration_state.json").read_bytes()
        with (self.base / "queue.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.advance()["state"], "busy")
        self.assertEqual((self.base / "orchestration_state.json").read_bytes(), before)
        self.preflight.assert_not_called()

    def test_existing_active_is_not_replaced_or_relaunched(self):
        self.state["active_experiment"] = {
            "id": "M0", "run_dir": "outputs/ablations/journal_20261025/M0/existing",
            "supervisor_pid": 123, "supervisor_identity": {
                "pid": 123, "start_ticks": 4, "command": "fake supervisor", "cwd": str(self.root)}}
        self.persist()
        before = (self.base / "orchestration_state.json").read_bytes()
        current = {"pid": 123, "start_ticks": 4, "command": "fake supervisor",
                   "cwd": str(self.root), "state": "S"}
        with patch.object(launcher, "identity", return_value=current):
            self.assertNotEqual(self.advance()["state"], "launched")
        self.assertEqual((self.base / "orchestration_state.json").read_bytes(), before)
        self.preflight.assert_not_called()

    def test_failed_first_entry_does_not_retrain_or_skip_to_next(self):
        self.registry["queue"][0]["status"] = "failed"
        self.persist()
        self.assertEqual(self.advance()["state"], "needs_review")
        self.assertFalse((self.base / "M1").exists())
        self.preflight.assert_not_called()

    def test_existing_receipt_is_never_overwritten(self):
        parent = self.base / "M0"
        receipt = parent / "launch_receipt.json"
        self.save(receipt, {"experiment": "M0", "run_dir": "old", "evidence": "retain"})
        old = receipt.read_bytes()
        self.preflight.side_effect = None
        self.preflight.return_value = {}
        self.assertEqual(self.advance()["state"], "already_launched")
        self.assertEqual(receipt.read_bytes(), old)

    def test_unprepared_component_is_not_silently_launched(self):
        self.registry["queue"] = [{"id": "M0", "status": "completed"},
                                  {"id": "M1", "status": "completed"},
                                  {"id": "S1", "status": "planned"}]
        self.persist()
        output = self.advance()
        self.assertEqual(output["state"], "not_prepared")
        self.assertEqual(output["next_experiment"], "S1")
        self.preflight.assert_not_called()

    def test_checked_root_rejects_escape_and_wrong_profile_parent(self):
        with self.assertRaises(ValueError):
            launcher.checked_root(self.root / "outside" / "M0" / "run")
        with self.assertRaises(ValueError):
            launcher.checked_root(self.base / "S1" / "run")
        with self.assertRaises(ValueError):
            launcher.checked_root(self.base / "M0" / ".." / "escaped")
        valid = self.base / "M0" / "valid"
        self.assertEqual(launcher.checked_root(valid), valid.resolve())

    def test_checked_root_rejects_symlink_outside_workspace(self):
        outside = self.root / "outside"
        outside.mkdir()
        parent = self.base / "M0"
        parent.mkdir(exist_ok=True)
        link = parent / "link"
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            launcher.checked_root(link / "run")

    def test_environment_removes_hidden_gpu_without_mutating_parent(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "0"}):
            actual = launcher.environment()
            self.assertNotIn("CUDA_VISIBLE_DEVICES", actual)
            self.assertEqual(actual["HF_HUB_OFFLINE"], "1")
            self.assertEqual(actual["TORCH_HOME"], "/Disk/HDD/Sk/.cache/torch")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")



class JournalPlanStatusTests(JournalLauncherFixtures, unittest.TestCase):
    def plan_path(self):
        path = self.root / ".codex/synapse_journal_ablation_plan.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def test_only_section11_status_row_changes_other_bytes_preserved(self):
        old = """# 论文消融
## 2. 主模型对照
| ID | 模型 | 训练增强 | 论文推理 | 工作内容 |
|---|---|---|---|---|
| M0 | baseline | 原增强 | 8-view | 补测 |
| M1 | baseline+ASEE | 原增强 | 8-view | 补测 |
## 4. ASEE机制
| ID | 因素 | 解释 |
|---|---|---|
| M1 | 双分支 | 机制对照 |
## 11. 实验状态登记
| 顺序 | ID | 内容 | 状态 | 结果与证据 |
|---|---|---|---|---|
| 1 | M0 | baseline已有best + 8-view | 已完成 | old_m0.md |
| 2 | M1 | baseline+ASEE已有best + 8-view | 待执行 | — |
| 3 | M10 | 相似ID不能修改 | 待执行 | — |
## 12. 后文
| 2 | M1 | 同格式其他节 | 待执行 | — |
"""
        path = self.plan_path()
        path.write_text(old, encoding="utf-8")
        launcher.set_plan_status("M1", "已完成", "new_m1.md")
        expected = old.replace(
            "| 2 | M1 | baseline+ASEE已有best + 8-view | 待执行 | — |",
            "| 2 | M1 | baseline+ASEE已有best + 8-view | 已完成 | new_m1.md |")
        self.assertEqual(path.read_bytes(), expected.encode("utf-8"))

    def test_missing_section11_target_raises_without_writing(self):
        old = """# 论文消融
## 2. 主模型对照
| M1 | 原表不该匹配 | 固定8-view |
## 4. ASEE机制
| M1 | 机制行不该匹配 | 原描述 |
## 11. 实验状态登记
| 顺序 | ID | 内容 | 状态 | 结果与证据 |
| 1 | M0 | baseline | 已完成 | evidence |
## 12. 后文
| 2 | M1 | 其他节五列表 | 待执行 | — |
"""
        path = self.plan_path()
        path.write_text(old, encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "section 11"):
            launcher.set_plan_status("M1", "已完成", "must_not_write.md")
        self.assertEqual(path.read_bytes(), before)


class JournalCompletionRejectionTests(JournalLauncherFixtures, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.run = self.base / "M0" / "example"
        self.folder = self.run / "evaluation/independent_test/example"
        self.folder.mkdir(parents=True)
        names = [{"label_id": i, "name": "Organ" + str(i), "name_zh": "器官" + str(i)}
                 for i in range(1, 9)]
        self.save(self.root / "priors/synapse_label_map.verified.json", {"classes": names})
        self.expected_cases = ["case%04d" % i for i in range(1, 13)]
        path = self.root / "project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(self.expected_cases) + "\n")
        self.case_rows = [
            {"epoch": "177", "case": case, "label_id": str(i),
             "organ": names[i-1]["name"], "organ_zh": names[i-1]["name_zh"],
             "dice": "0.5", "dice_percent": "50.0", "hd95_mm": "2.0", "hd95_voxel": "3.0"}
            for case in self.expected_cases for i in range(1, 9)]
        self.organ_rows = [
            {"epoch": "177", "label_id": str(i), "organ": names[i-1]["name"],
             "organ_zh": names[i-1]["name_zh"], "num_cases": "12", "dice": "0.5",
             "dice_percent": "50.0", "hd95_mm": "2.0", "hd95_voxel": "3.0"}
            for i in range(1, 9)]
        self.launch = {"experiment": "M0", "preflight": {"checkpoint": {"epoch": 177}}}

    def write_csv(self, name, rows):
        path = self.folder / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def reject(self, message):
        self.write_csv("physical_case_organ_metrics.csv", self.case_rows)
        self.write_csv("physical_epoch_organ_metrics.csv", self.organ_rows)
        with self.assertRaisesRegex(RuntimeError, message):
            launcher.complete(self.run, self.launch)
        self.assertFalse((self.run / "summary.json").exists())
        self.popen.assert_not_called()

    def test_complete_rejects_missing_case_class(self):
        self.case_rows.pop()
        self.reject("Incomplete or duplicate")

    def test_complete_rejects_duplicate_case_class(self):
        self.case_rows[-1] = copy.deepcopy(self.case_rows[-2])
        self.reject("Incomplete or duplicate")

    def test_complete_rejects_missing_organ_summary(self):
        self.organ_rows.pop()
        self.reject("Incomplete organ summary")

    def test_complete_rejects_wrong_organ_name(self):
        self.case_rows[0]["organ"] = "Wrong organ"
        self.reject("Organ name/epoch mismatch")

    def test_complete_rejects_wrong_epoch(self):
        self.case_rows[0]["epoch"] = "176"
        self.reject("Organ name/epoch mismatch")

    def test_complete_rejects_nonfinite_or_invalid_metrics(self):
        for field, value in [("dice", "nan"), ("hd95_mm", "inf"), ("hd95_voxel", "-1"), ("dice", "1.01")]:
            with self.subTest(field=field, value=value):
                old = self.case_rows[0][field]
                self.case_rows[0][field] = value
                self.reject("Invalid numeric metric")
                self.case_rows[0][field] = old


if __name__ == "__main__":
    unittest.main()
