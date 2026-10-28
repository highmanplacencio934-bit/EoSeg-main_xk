"""Retain the existing physical reports and add exact teacher-source provenance."""
import json
from training.synapse_physical_selection import SynapsePhysicalHD95Selection

class SynapseTeacherHD95Selection(SynapsePhysicalHD95Selection):
    def _prepare_reports(self, pl_module):
        super()._prepare_reports(pl_module)
        if not getattr(pl_module, "teacher_source", None):
            raise ValueError("Use the reviewed teacher evaluation module")
        contract = {**pl_module.teacher_source,
                    "teacher_functions_executed": True,
                    "network_adapter": "gray repeat to RGB; EoSeg query-to-pixel scores; bilinear resize; FP32 log-score adapter before teacher softmax/argmax",
                    "extra_teacher_softmax_dtype": "float32; model/decoder stay 16-mixed",
                    "cudnn_flags": "EoSeg defaults preserved; teacher deterministic toggle not applied",
                    "image_export": "disabled; avoids teacher's hardcoded XY spacing=1",
                    "report_hd95_mm": "added MedPy physical distance on same masks",
                    "report_hd95_voxel": "actual teacher calculate_metric_percase result",
                    "empty_mask_policy": "teacher and existing baseline legacy rules preserved"}
        (self.run_dir / "teacher_metric_contract.json").write_text(json.dumps(contract, indent=2))
