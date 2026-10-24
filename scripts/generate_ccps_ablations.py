"""Generate independent CCPS configs without touching the original ASEE config.
Run in repo root: python scripts/generate_ccps_ablations.py
Requires PyYAML already used by project configs.
"""
from pathlib import Path
import copy
import argparse
import yaml

BASE = Path("configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml")
DEST = Path("configs/synapse/ccps_ablations")

EXPERIMENTS = {
    "a0_off": {"selector_mode": "off"},
    "a1_global_patch": {"selector_mode": "global"},
    "a2_visual_patch": {"selector_mode": "visual"},
    "a3_visual_prior_patch": {"selector_mode": "visual_prior"},
    "a4_visual_prior_query": {"selector_mode": "visual_prior", "selector_on_patch": False, "selector_on_query": True},
    "a5_visual_prior_both": {"selector_mode": "visual_prior", "selector_on_patch": True, "selector_on_query": True},
    "a6_visual_prior_patch_presence": {"selector_mode": "visual_prior", "presence_loss_weight": 0.1},
    "a7_visual_prior_both_presence": {"selector_mode": "visual_prior", "selector_on_patch": True, "selector_on_query": True, "presence_loss_weight": 0.1},
    "a8_visual_prior_text_only": {"selector_mode": "visual_prior", "use_text": True, "use_probability": False},
    "a10_context_only": {"patch_gate_effect": "context_only"},
    "a11_residual_only": {"patch_gate_effect": "residual_only"},
    "a9_visual_prior_prob_only": {"selector_mode": "visual_prior", "use_text": False, "use_probability": True},
}


def generate(base=BASE, dest=DEST, overwrite=False):
    base, dest = Path(base), Path(dest)
    collisions = [dest / f"{name}.yaml" for name in EXPERIMENTS if (dest / f"{name}.yaml").exists()]
    if collisions and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {collisions}; pass --overwrite explicitly")
    data = yaml.safe_load(base.read_text(encoding="utf-8"))
    dest.mkdir(parents=True, exist_ok=True)
    for name, overrides in EXPERIMENTS.items():
        cfg = copy.deepcopy(data)
        net = cfg["model"]["init_args"]["network"]["init_args"]
        mdl = cfg["model"]["init_args"]
        for k, v in overrides.items():
            if k in {"presence_loss_weight", "presence_pos_weight"}:
                mdl[k] = v
            else:
                net[k] = v
        out = f"outputs/innovation2/{name}"
        cfg["trainer"]["default_root_dir"] = out
        cfg["trainer"]["callbacks"][0]["init_args"]["output_dir"] = out
        cfg["trainer"]["logger"]["init_args"]["name"] = "ccps_" + name
        (dest / f"{name}.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"wrote {dest / (name + '.yaml')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(args.base, args.dest, args.overwrite)


if __name__ == "__main__":
    main()
