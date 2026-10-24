# CCPS reference code

Use the accompanying `EoSeg_CCPS_Implementation_Plan.md` for full implementation and ablation instructions. Copy only the new relative source paths into the existing EoSeg repository; do not overwrite the original baseline or ASEE files. No checkpoints or data included.

CPU tests: `pytest -q tests/test_ccps.py`

Config generation: `python scripts/generate_ccps_ablations.py`
