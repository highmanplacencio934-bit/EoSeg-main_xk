"""Independent deterministic G0/G4/G5 probe, adapted from the archived diagnosis.
Original archived script/evidence and production entries are never modified.

Run prepare and CPU self-test first; old/roi modes need the configured GPU reserve.
Default scientific configuration remains 220 epochs / 30360 estimated steps.
Both modes default to native DataLoader batches and preserve native AMP overflow handling.
"""
from __future__ import annotations
import argparse, functools, hashlib, json, os, pickle, random, runpy, subprocess, sys
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
BASE = None
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "roi", "deterministic", "compare", "self-test"])
    p.add_argument("--tag")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--protocol-disabled", action="store_true", help="Test recovery to the original criterion without strict settings")
    p.add_argument("--profile", choices=["baseline","asee","ccps"], default="baseline")
    p.add_argument("--batch-target", type=int, default=0, help="G5: 20-100 native batches; zero selects actual-update target")
    p.add_argument("--left", default="old")
    p.add_argument("--right", default="roi")
    p.add_argument("--required-free-mib", type=int, default=22000)
    p.add_argument("--max-attempts", type=int, default=16)
    p.add_argument("--target-updates", type=int, default=2)
    p.add_argument("--native-batches", dest="native_batches", action="store_true", default=True)
    p.add_argument("--fixed-batch", dest="native_batches", action="store_false")
    p.add_argument("--cpu-interface-check", action="store_true", help="Intercept actual Trainer.fit before device transfer; no training")
    p.add_argument("--pair-rng", action="store_true",
                   help="Restore old on-train-start CPU/CUDA RNG for roi; report native states first.")
    return p.parse_args()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")

def main():
    args = parse_args()
    global BASE
    BASE=args.output_dir.resolve()
    BASE.mkdir(parents=True,exist_ok=True)
    if args.batch_target and not 20<=args.batch_target<=100:
        raise SystemExit("G5 requires 20-100 native batches")
    if args.mode in ("deterministic", "roi") and not args.cpu_interface_check:
        if args.required_free_mib < 21606:
            raise SystemExit("Do not lower the measured baseline peak+headroom below21606MiB")
        free = int(subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free",
                      "--format=csv,noheader,nounits"], text=True).strip())
        if free < args.required_free_mib:
            raise SystemExit(f"GPU diagnostic deferred: free={free}MiB, require={args.required_free_mib}MiB")
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            raise SystemExit("Formal GPU probe cannot inherit CPU-only CUDA_VISIBLE_DEVICES")
    import numpy as np
    import torch
    from lightning.pytorch import Callback, Trainer, seed_everything
    from lightning.fabric.utilities.data import _auto_add_worker_init_fn
    torch.set_num_threads(1)

    def tensor_hash(t):
        if not isinstance(t,torch.Tensor):
            return hashlib.sha256(json.dumps(pack(t),sort_keys=True,default=str).encode()).hexdigest()
        x=t.detach().cpu().contiguous()
        header=json.dumps({"dtype":str(x.dtype),"shape":list(x.shape)},sort_keys=True).encode()
        return hashlib.sha256(header+x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

    def rng_snapshot():
        state = {"cpu": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
        if torch.cuda.is_initialized():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def rng_hashes(s):
        result = {"cpu": tensor_hash(s["cpu"]),
                  "python": hashlib.sha256(pickle.dumps(s["python"])).hexdigest(),
                  "numpy": hashlib.sha256(pickle.dumps(s["numpy"])).hexdigest()}
        if "cuda" in s:
            result["cuda"] = [tensor_hash(x) for x in s["cuda"]]
        return result

    def pack(value):
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_hash(value)}
        if isinstance(value, dict):
            return {str(k): pack(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [pack(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def important(name):
        return any(x in name for x in ("query_embed", "mask_embed", "class_embed", "mask_head", "class_head", "feature_upscale", "layer_gate", "layer_fusion_bias"))

    if args.mode == "prepare":
        from datasets.synapse_semantic import SynapseSemantic
        target = BASE / "trajectory_fixed_batch.pt"
        if target.exists():
            raise SystemExit("Fixed batch already exists; never overwrite silently")
        seed_everything(0, workers=True)
        data = SynapseSemantic(batch_size=16, num_workers=8)
        data.setup("fit")
        loader = data.train_dataloader()
        _auto_add_worker_init_fn(loader, 0)
        it = iter(loader)
        try:
            batch = next(it)
        finally:
            it._shutdown_workers()
        assert batch[0].shape == (16, 3, 384, 384)
        torch.save(batch, target)
        report = {"batch": pack(batch), "provenance": "original SynapseSemantic seed0/16/workers8",
                  "scope": "controlled shared batch; historical batches were not archived"}
        write(BASE / "trajectory_fixed_batch.json", report)
        print(json.dumps({"state": "prepared", "file": str(target), "shape": list(batch[0].shape)}))
        return

    if args.mode == "self-test":
        # Verify the snapshot operation and sampling observer consume no RNG.
        seed_everything(0, workers=True)
        x = torch.randn(3, 4)
        state = torch.get_rng_state().clone()
        assert pack(x) == pack(x.clone())
        assert torch.equal(state, torch.get_rng_state())
        original_rand = torch.rand
        torch.set_rng_state(state)
        expected = original_rand(8)
        end = torch.get_rng_state().clone()
        torch.set_rng_state(state)
        seen = []
        def observed(*a, **kw):
            v = original_rand(*a, **kw)
            seen.append(pack(v))
            return v
        torch.rand = observed
        try:
            actual = torch.rand(8)
        finally:
            torch.rand = original_rand
        assert torch.equal(expected, actual) and torch.equal(end, torch.get_rng_state())
        # Real optimizer hooks only fire when step really runs; GradScaler omission
        # is detected by their absence, not by Lightning global_step.
        layer = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=2e-4)
        events = []
        handles = [optimizer.register_step_pre_hook(lambda *a: events.append("pre")),
                   optimizer.register_step_post_hook(lambda *a: events.append("post"))]
        layer(x).sum().backward()
        optimizer.step()
        assert events == ["pre", "post"]
        for h in handles:
            h.remove()
        from inspect import signature
        @functools.wraps(Trainer.fit)
        def wrapped(self, *a, **kw):
            return None
        assert signature(wrapped) == signature(Trainer.fit)
        write(BASE / "trajectory_cpu_self_test.json",
              {"status": "passed", "snapshot_rng_preserved": True, "rand_observer_exact": True,
               "optimizer_hook_real_step": True, "trainer_fit_signature_preserved": True,
               "cuda_initialized": torch.cuda.is_initialized()})
        print("trajectory CPU interface self-test passed")
        return

    if args.mode == "compare":
        from scripts.validate_synapse_deterministic_trace import compare_pair
        left=BASE/("trajectory_"+args.left);right=BASE/("trajectory_"+args.right)
        a,b=[json.loads((directory/"trajectory_trace.json").read_text()) for directory in (left,right)]
        ia,ib=[json.loads((directory/"trajectory_initialization.json").read_text()) for directory in (left,right)]
        report=compare_pair(a,b,ia,ib,batch_target=args.batch_target,target_updates=args.target_updates)
        report.update(left=str(left),right=str(right),attempt_counts=[len(a["attempts"]),len(b["attempts"])],
                      actual_updates=[a["actual_optimizer_updates"],b["actual_optimizer_updates"]])
        write(BASE/("trajectory_comparison_"+args.left+"_"+args.right+".json"),report)
        print(json.dumps(report,indent=2))
        if not report["exact_equality"]:
            raise SystemExit("Candidate trajectories differ; gate not passed")
        return

    run = BASE / ("trajectory_" + (args.tag or (args.mode + "_cpu_interface" if args.cpu_interface_check else args.mode)))
    assert args.target_updates>=1 and args.max_attempts>=args.target_updates
    run.mkdir(exist_ok=False)
    shared_batch = None if args.native_batches else torch.load(BASE / "trajectory_fixed_batch.pt", map_location="cpu", weights_only=False)
    expected_batch = None if shared_batch is None else pack(shared_batch)
    trace = {"mode": args.mode, "attempts": [], "actual_optimizer_updates": 0,
             "native_rng_hashes": None, "effective_rng_hashes": None,
             "gpu_peak_allocated_mib": None, "pair_rng": args.pair_rng, "target_updates": args.target_updates, "native_batches": args.native_batches,
             "note": "independent controlled candidate protocol; original scientific settings and native AMP skips retained"}
    initial_state = None

    class FirstUpdateCapture(Callback):
        def __init__(self):
            self.current = None
            self.handles = []
            self.original_rand = None
            self.original_sample_point = None
            self.sample_module = None

        def flush(self):
            write(run / "trajectory_trace.json", trace)

        def gradients(self, module, label):
            hashes = {}
            selected = {}
            for name, param in module.named_parameters():
                if param.grad is None:
                    hashes[name] = None
                else:
                    hashes[name] = tensor_hash(param.grad)
                    if important(name):
                        selected[name] = param.grad.detach().cpu().clone()
            if selected:
                torch.save(selected, run / f"trajectory_attempt{len(trace['attempts'])}_{label}_selected.pt")
            return hashes

        def on_train_start(self, trainer, pl_module):
            raw = getattr(pl_module, "_orig_mod", pl_module)
            assert int(trainer.estimated_stepping_batches) == 30360
            assert trainer.max_epochs == 220 and trainer.precision == "16-mixed"
            assert trainer.gradient_clip_val == .01 and trainer.gradient_clip_algorithm.value == "norm"
            assert len(trainer.optimizers) == 1
            trace["estimated_steps"] = int(trainer.estimated_stepping_batches)
            trace["precision"] = trainer.precision
            trace["optimizer_groups"] = [{k: v for k, v in g.items() if k != "params"} for g in trainer.optimizers[0].param_groups]
            native = rng_snapshot()
            trace["native_rng_hashes"] = rng_hashes(native)
            torch.save(native, run / "trajectory_native_rng.pt")
            if args.pair_rng:
                old = torch.load(BASE / ("trajectory_" + args.left) / "trajectory_native_rng.pt",
                                 map_location="cpu", weights_only=False)
                torch.set_rng_state(old["cpu"])
                random.setstate(old["python"])
                np.random.set_state(old["numpy"])
                torch.cuda.set_rng_state_all(old["cuda"])
            trace["effective_rng_hashes"] = rng_hashes(rng_snapshot())
            # Observe the exact sample_point call, after top-k point selection.
            import importlib
            self.sample_module = importlib.import_module("transformers.models.mask2former.modeling_mask2former")
            self.original_sample_point = self.sample_module.sample_point
            original_sample_point = self.original_sample_point
            def observed_sample_point(input_features, point_coordinates, *a, **kw):
                result = original_sample_point(input_features, point_coordinates, *a, **kw)
                if self.current is not None:
                    self.current["sample_point_calls"].append({"input": pack(input_features), "coordinates": pack(point_coordinates), "output": pack(result), "kwargs": pack(kw)})
                return result
            if args.mode == "roi":
                self.sample_module.sample_point = observed_sample_point
            criterion = raw.criterion
            if args.mode == "deterministic":
                for owner in (criterion,criterion.matcher):
                    original_local_sample=owner._sample_point
                    def local_sample(*a,_original=original_local_sample,**kw):
                        result=_original(*a,**kw)
                        if self.current is not None:
                            self.current["sample_point_calls"].append({"input":pack(a[0]),"coordinates":pack(a[1]),"output":pack(result),"kwargs":pack(kw)})
                        return result
                    owner._sample_point=local_sample
            original_forward = criterion.forward
            original_matcher = criterion.matcher.forward

            def matcher(*a, **kw):
                result = original_matcher(*a, **kw)
                if self.current is not None:
                    self.current["matcher"] = pack(result)
                return result
            criterion.matcher.forward = matcher

            @torch.compiler.disable
            def forward(*a, **kw):
                if self.current is None:
                    return original_forward(*a, **kw)
                self.current["forward"] = pack({"args": a, "kwargs": kw})
                if not (run / "trajectory_first_criterion_inputs.pt").exists():
                    def cpu_copy(v):
                        if isinstance(v, torch.Tensor):return v.detach().cpu().clone()
                        if isinstance(v, dict):return {k:cpu_copy(x) for k,x in v.items()}
                        if isinstance(v, tuple):return tuple(cpu_copy(x) for x in v)
                        if isinstance(v, list):return [cpu_copy(x) for x in v]
                        return v
                    torch.save({"args":cpu_copy(a),"kwargs":cpu_copy(kw),"rng":rng_snapshot()}, run / "trajectory_first_criterion_inputs.pt")
                original_rand = torch.rand
                def observed_rand(*a, **kw):
                    value = original_rand(*a, **kw)
                    self.current["random_samples"].append(pack(value))
                    return value
                torch.rand = observed_rand
                try:
                    losses = original_forward(*a, **kw)
                finally:
                    torch.rand = original_rand
                self.current["loss_components"] = {k: float(v.detach().cpu()) for k, v in losses.items()}
                return losses
            criterion.forward = forward

            def before_step(optimizer, positional, keyword):
                if self.current is None:
                    raise RuntimeError("Optimizer step without a captured batch")
                self.current["optimizer_called"] = True
                self.current["optimizer_step_lrs"] = [g["lr"] for g in optimizer.param_groups]
                self.current["clipped_gradients"] = self.gradients(raw, "clipped_gradient")
            def after_step(optimizer, positional, keyword):
                trace["actual_optimizer_updates"] += 1
                self.current["weights_after_step"] = {k: tensor_hash(v) for k, v in raw.state_dict().items()}
                self.current["state_keys_changed_from_initial"] = [k for k,v in self.current["weights_after_step"].items() if initial_state[k]!=v]
                parameter_names={id(p):name for name,p in raw.named_parameters()}
                self.current["optimizer_state_after"]={parameter_names[id(p)]:pack(state) for p,state in optimizer.state.items()}
                self.current["parameter_keys_changed_from_initial"]=[name for name,p in raw.named_parameters() if tensor_hash(p)!=initial_state[name]]
                selected = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items() if important(k)}
                torch.save(selected, run / f"trajectory_attempt{len(trace['attempts'])}_updated_selected_weights.pt")
            optimizer = trainer.optimizers[0]
            self.handles = [optimizer.register_step_pre_hook(before_step),
                            optimizer.register_step_post_hook(after_step)]
            # Strategy.pre_backward is called on the original closure loss before
            # MixedPrecision.pre_backward scales it and calls on_before_backward.
            native_pre_backward = trainer.strategy.pre_backward
            @functools.wraps(native_pre_backward)
            def observe_native_loss(loss, *a, **kw):
                if self.current is not None:
                    self.current["loss_total"] = float(loss.detach().cpu())
                return native_pre_backward(loss, *a, **kw)
            trainer.strategy.pre_backward = observe_native_loss
            trace["loss_semantics"] = {"loss_total": "original closure loss at Strategy.pre_backward, before AMP scaling",
                                       "loss_total_scaled": "on_before_backward after MixedPrecision.scaler.scale"}
            self.flush()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.current = {"batch_index": batch_idx, "batch": pack(batch),
                            "rng_before_batch": rng_hashes(rng_snapshot()), "random_samples": [], "sample_point_calls": [],
                            "optimizer_called": False}
            if expected_batch is not None:
                assert self.current["batch"] == expected_batch, "Transferred fixed batch does not match shared CPU input"
            scaler = trainer.precision_plugin.scaler
            self.current["scaler_before"] = scaler.state_dict() if scaler else None
            trace["attempts"].append(self.current)

        def on_before_backward(self, trainer, pl_module, loss):
            self.current["loss_total_scaled"] = float(loss.detach().cpu())

        def on_after_backward(self, trainer, pl_module):
            raw = getattr(pl_module, "_orig_mod", pl_module)
            self.current["scaled_gradients"] = self.gradients(raw, "scaled_gradient")

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            raw = getattr(pl_module, "_orig_mod", pl_module)
            self.current["unscaled_gradients"] = self.gradients(raw, "unscaled_gradient")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            scaler = trainer.precision_plugin.scaler
            self.current["scaler_after"] = scaler.state_dict() if scaler else None
            self.current["lightning_global_step_after"] = trainer.global_step
            self.flush()
            if (len(trace["attempts"]) >= args.batch_target if args.batch_target else trace["actual_optimizer_updates"] >= args.target_updates or len(trace["attempts"]) >= args.max_attempts):
                trainer.should_stop = True

        def on_train_end(self, trainer, pl_module):
            trace["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024 ** 2
            trace["state"] = ("captured_native_batch_target" if len(trace["attempts"]) == args.batch_target else "incomplete_batch_target") if args.batch_target else ("captured_target_optimizer_updates" if trace["actual_optimizer_updates"] == args.target_updates else "no_actual_update_within_attempt_limit")
            self.flush()
            for h in self.handles:
                h.remove()
            if self.sample_module is not None:
                self.sample_module.sample_point = self.original_sample_point

        def on_exception(self, trainer, pl_module, exception):
            trace["state"] = "diagnostic_failed"
            trace["exception"] = repr(exception)
            self.flush()

    real_fit = Trainer.fit

    @functools.wraps(real_fit)
    def intercepted_fit(self, model, *positional, **kwargs):
        raw = getattr(model, "_orig_mod", model)
        nonlocal initial_state
        initial_state = {k: tensor_hash(v) for k, v in raw.state_dict().items()}
        write(run / "trajectory_initialization.json",
              {"state": initial_state, "rng": rng_hashes(rng_snapshot()),
               "compiled": hasattr(model, "_orig_mod"), "config": self.baseline_resolved_config})
        if args.cpu_interface_check:
            from types import SimpleNamespace
            original_trainer=raw._trainer
            raw._trainer=SimpleNamespace(estimated_stepping_batches=30360)
            before=rng_hashes(rng_snapshot())
            configured=raw.configure_optimizers()
            optimizer=configured["optimizer"]
            names={id(p):n for n,p in raw.named_parameters()}
            groups=[{**{k:v for k,v in g.items() if k!="params"},"parameter_names":[names[id(p)] for p in g["params"]]} for g in optimizer.param_groups]
            raw._trainer=original_trainer
            assert before==rng_hashes(rng_snapshot()), "optimizer construction consumed RNG"
            write(run/"optimizer_groups.json",{"groups":groups,"scheduler":configured["lr_scheduler"]["scheduler"].state_dict()})
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized()
            print(json.dumps({"state": "cpu_interface_passed", "mode": args.mode, "state_tensors": len(initial_state), "compiled": hasattr(model, "_orig_mod")}))
            return None
        self.callbacks.append(FirstUpdateCapture())
        # Preserve the full loader length and its RNG consumption. Only replace the
        # batch at the standard data-transfer hook. Both entries see one same batch.
        dm = kwargs.get("datamodule")
        if dm is None:
            raise RuntimeError("Actual entry did not provide datamodule")
        def fixed_transfer(self, batch, dataloader_idx):
            return shared_batch
        if not args.native_batches:
            dm.on_before_batch_transfer = MethodType(fixed_transfer, dm)
        return real_fit(self, model, *positional, **kwargs)
    Trainer.fit = intercepted_fit
    entry=ROOT/("scripts/run_synapse_deterministic.py" if args.mode=="deterministic" else "scripts/run_synapse_roi.py")
    sources={"baseline":"configs/synapse/baseline_2_roi.yaml",
             "asee":"configs/synapse/vit_query_mul_scale_fusion_asee_plan_hd95_mm.yaml",
             "ccps":"configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml"}
    config=("configs/synapse/deterministic/"+args.profile+".yaml") if args.mode=="deterministic" else sources[args.profile]
    cb="training.synapse_physical_selection.SynapsePhysicalHD95Selection"
    sys.argv=[str(entry),"fit","--config",config,"--seed_everything=0",
              "--trainer.default_root_dir="+str(run),"--trainer.logger=false",
              "--trainer.enable_progress_bar=false",
              "--trainer.callbacks="+json.dumps([{"class_path":cb,"init_args":{
                  "last_n_epochs":50,"expected_test_cases":12,"output_dir":str(run)}}])]
    if args.mode=="deterministic":
        sys.argv += ["--deterministic_probe_batches=100","--deterministic_output_dir="+str(run/"protocol_records")]
        if args.protocol_disabled:sys.argv += ["--deterministic_enabled=false"]
    runpy.run_path(str(entry), run_name="__main__")
    if not args.cpu_interface_check and trace.get("state") != ("captured_native_batch_target" if args.batch_target else "captured_target_optimizer_updates"):
        raise SystemExit("No actual optimizer update captured; inspect trace, do not claim equality")

if __name__ == "__main__":
    main()
