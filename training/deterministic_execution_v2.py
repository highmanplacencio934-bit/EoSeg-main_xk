"""Independent execution v2: unchanged v1 baseline/ASEE, local CCPS compile boundary."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib,json
import torch
import torch._functorch.config as aot_config
from training import deterministic_protocol as v1
from training.deterministic_protocol import backend_state,configure_strict,restore_backend,install_criterion,write_json

PROTOCOL_ID="ce2d_stableseg_execution_v2"
SOURCE_FILES=v1.SOURCE_FILES+("training/deterministic_execution_v2.py","scripts/run_synapse_deterministic_v2.py")
def source_hashes(root):
    root=Path(root);result=v1.source_hashes(root)
    for name in SOURCE_FILES:result[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    return result

@contextmanager
def execution_context(profile,*,enabled=True):
    if profile not in ("baseline","asee","ccps"):raise ValueError("Unknown execution profile")
    if enabled and profile=="ccps":
        with aot_config.patch(backward_pass_autocast="off"):
            yield
    else:
        yield

def install_execution_boundary(model,profile,*,enabled=True):
    if profile not in ("baseline","asee","ccps"):raise ValueError("Unknown execution profile")
    if not enabled or profile!="ccps":return False
    raw=getattr(model,"_orig_mod",model)
    if not hasattr(raw,"network") or not hasattr(raw.network,"asee") or getattr(raw.network.asee,"selector",None) is None:
        raise TypeError("Explicit CCPS profile requires an enabled selector")
    if getattr(raw,"_ccps_execution_v2_boundary",False):return False
    bn=raw.network.feature_upscale[10]
    if not isinstance(bn,torch.nn.BatchNorm2d) or bn.num_features!=256:
        raise ValueError("Decoder differs from validated CCPS boundary")
    bn.forward=torch.compiler.disable(bn.forward)
    raw._ccps_execution_v2_boundary=True
    return True

def require_readiness(path,root,profile):
    path=Path(path)
    if not path.is_file():raise RuntimeError("Execution v2 readiness is missing; formal fit blocked")
    d=json.loads(path.read_text())
    if d.get("protocol")!=PROTOCOL_ID or d.get("passed") is not True:raise RuntimeError("Execution v2 readiness is not passed")
    if profile not in d.get("validated_profiles",[]):raise RuntimeError("Model profile not validated")
    if d.get("sources")!=source_hashes(root):raise RuntimeError("Execution v2 source hashes changed")
    if any(d.get("gates",{}).get("G"+str(i))!="passed" for i in range(6)):raise RuntimeError("All G0-G5 gates required")

class DeterministicProtocolRecord(v1.DeterministicProtocolRecord):
    def on_fit_start(self,trainer,pl_module):
        super().on_fit_start(trainer,pl_module)
        raw=getattr(pl_module,"_orig_mod",pl_module)
        if self.profile=="ccps" and (not getattr(raw,"_ccps_execution_v2_boundary",False) or aot_config.backward_pass_autocast!="off"):
            raise RuntimeError("CCPS execution controls did not take effect")
        path=self.run_dir/"protocol_manifest.json"
        d=json.loads(path.read_text())
        d.update(protocol=PROTOCOL_ID,sources=source_hashes(self.root),
            execution_profile=self.profile,
            ccps_local_bn_eager_boundary=bool(getattr(raw,"_ccps_execution_v2_boundary",False)),
            compiler_backward_assumption=aot_config.backward_pass_autocast,
            global_outer_compile_preserved=True,
            baseline_asee_execution_unchanged_from_v1=self.profile!="ccps")
        write_json(path,d)
