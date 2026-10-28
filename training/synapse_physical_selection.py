"""Named millimetre/voxel reports without changing Dice checkpoint selection."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

from training.synapse_selection import append_csv, summarize_cases
from training.synapse_test_selection import SynapseTestSelection
from training.synapse_physical_geometry import project_path


class SynapsePhysicalHD95Selection(SynapseTestSelection):
    def __init__(self, last_n_epochs: int = 50, expected_test_cases: int = 12,
                 output_dir: str = "outputs/physical_hd95",
                 label_map: str = "priors/synapse_label_map.verified.json"):
        super().__init__(last_n_epochs, expected_test_cases, output_dir)
        self.label_map_path = project_path(label_map)
        self._test_results = []

    def _prepare_reports(self, pl_module):
        geometry = getattr(pl_module, "physical_geometry", None)
        if geometry is None or not getattr(pl_module, "physical_preflight", None):
            raise ValueError("Use SynapsePhysicalHD95Semantic with complete geometry preflight")
        self.hd95_voxel_enabled = getattr(pl_module, "compute_hd95_voxel", True)
        raw_map = self.label_map_path.read_bytes()
        mapping = json.loads(raw_map)
        if mapping.get("verified") is not True:
            raise ValueError("A verified Synapse organ-name mapping is required")
        self.names = {row["label_id"]: row for row in mapping["classes"]}
        if len(self.names) != len(mapping["classes"]) or set(self.names) != set(range(1, pl_module.num_classes)):
            raise ValueError("Organ names must exactly cover foreground class IDs")
        (self.run_dir / "spacing_manifest.json").write_bytes(geometry.raw_bytes)
        (self.run_dir / "label_map.json").write_bytes(raw_map)
        metadata = {
            "hd95_unit": "mm", "legacy_hd95_unit": "voxel_grid",
            "hd95_voxel_enabled": self.hd95_voxel_enabled,
            "voxelspacing_order": "zyx", "spacing_manifest_sha256": geometry.sha256,
            "label_map_sha256": hashlib.sha256(raw_map).hexdigest(),
            "empty_mask_policy": "unchanged baseline legacy Dice/HD95 zero branches; flagged per row",
            "selection_metric": "mean Dice; unchanged", "geometry_preflight": pl_module.physical_preflight,
            "csv_contract": "test_*metrics.csv hd95_* fields are mm in this opt-in run only; named files explicitly separate mm and voxel",
        }
        (self.run_dir / "metric_units.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        self._prepare_reports(pl_module)

    def _write_named(self, results, summary, epoch):
        rows = []
        enabled = self.hd95_voxel_enabled
        for result in results:
            if result.get("hd95_voxel_enabled", enabled) is not enabled:
                raise ValueError("Inconsistent HD95 voxel computation mode")
            values = result["hd95_voxel_per_class"]
            if enabled and any(v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("Enabled voxel HD95 must contain finite nonnegative values")
            if len(values) != len(self.names) or (not enabled and (any(v is not None for v in values) or result.get("mean_hd95_voxel") is not None)):
                raise ValueError("Disabled voxel HD95 must remain uncomputed/null")
            if result.get("hd95_unit") != "mm":
                raise ValueError("Refusing to mix physical and voxel HD95 results")
            for c in sorted(self.names):
                name = self.names[c]
                rows.append({"epoch": epoch, "case": result["case"], "label_id": c,
                             "organ": name["name"], "organ_zh": name["name_zh"],
                             "dice": result["dice_per_class"][c-1],
                             "dice_percent": 100 * result["dice_per_class"][c-1],
                             "hd95_mm": result["hd95_mm_per_class"][c-1],
                             "hd95_voxel": result["hd95_voxel_per_class"][c-1],
                             "status": result["metric_status_per_class"][c-1],
                             "spacing_z_mm": result["spacing_zyx_mm"][0],
                             "spacing_y_mm": result["spacing_zyx_mm"][1],
                             "spacing_x_mm": result["spacing_zyx_mm"][2]})
        append_csv(self.run_dir / "physical_case_organ_metrics.csv", rows)
        organ_rows = []
        for c in sorted(self.names):
            organ_rows.append({"epoch": epoch, "label_id": c,
                               "organ": self.names[c]["name"], "organ_zh": self.names[c]["name_zh"],
                               "num_cases": len(results), "dice": summary[f"dice_class_{c}"],
                               "dice_percent": 100 * summary[f"dice_class_{c}"],
                               "hd95_mm": summary[f"hd95_class_{c}"],
                               "hd95_voxel": (sum(r["hd95_voxel_per_class"][c-1] for r in results)/len(results) if enabled else None),
                               "legacy_empty_cases": sum(r["metric_status_per_class"][c-1] != "nonempty" for r in results)})
        append_csv(self.run_dir / "physical_epoch_organ_metrics.csv", organ_rows)

    def _evaluate(self, trainer, pl_module):
        results, summary = super()._evaluate(trainer, pl_module)
        self._write_named(results, summary, trainer.current_epoch + 1)
        return results, summary

    def on_test_start(self, trainer, pl_module):
        if trainer.world_size != 1:
            raise ValueError("Physical Synapse reporting supports a single device")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = project_path(self.output_dir) / "independent_test" / f"{stamp}_{uuid4().hex[:8]}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._test_results = []
        self._test_checkpoint_epoch = getattr(pl_module, "physical_checkpoint_epoch", None)
        ckpt = getattr(trainer, "ckpt_path", None)
        self._test_checkpoint_info = {"ckpt_path": str(ckpt) if ckpt else None,
                                      "checkpoint_epoch": self._test_checkpoint_epoch,
                                      "checkpoint_global_step": getattr(pl_module, "physical_checkpoint_global_step", None),
                                      "checkpoint_sha256": None}
        if ckpt and Path(ckpt).is_file():
            digest = hashlib.sha256()
            with Path(ckpt).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._test_checkpoint_info["checkpoint_sha256"] = digest.hexdigest()
        self._prepare_reports(pl_module)
        (self.run_dir / "checkpoint_source.json").write_text(
            json.dumps(self._test_checkpoint_info, indent=2), encoding="utf-8")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not isinstance(outputs, dict) or outputs.get("hd95_unit") != "mm":
            raise ValueError("Expected physical case metrics from test_step")
        self._test_results.append(outputs)

    def on_test_epoch_end(self, trainer, pl_module):
        summary = summarize_cases(self._test_results, self.expected_test_cases, pl_module.num_classes-1)
        self._write_named(self._test_results, summary, self._test_checkpoint_epoch)
        report = {"hd95_unit": "mm", "mean_dice": summary["mean_dice"],
                  "mean_hd95_mm": summary["mean_hd95"],
                  "mean_hd95_voxel": (sum(r["mean_hd95_voxel"] for r in self._test_results)/len(self._test_results) if self.hd95_voxel_enabled else None),
                  "hd95_voxel_enabled": self.hd95_voxel_enabled,
                  "num_cases": len(self._test_results), **self._test_checkpoint_info}
        (self.run_dir / "physical_test_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
