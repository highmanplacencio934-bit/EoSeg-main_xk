"""EoSeg adaptation of the uploaded teacher HD95 entry; no work at import.

The original AgileFormer-only entry is archived under outputs/audit. The
actual teacher volume/metric functions remain unchanged in AgileFormer-main.
Usage: python HD95_test.py --ckpt_path ... --compile_test [Lightning options]
"""
import sys
from pathlib import Path


def main():
    from main import cli_main
    default = Path(__file__).resolve().parent / "configs/synapse/vit_query_mul_scale_fusion_asee_teacher_hd95.yaml"
    sys.argv = [sys.argv[0], "test", "--config", str(default), *sys.argv[1:]]
    cli_main()


if __name__ == "__main__":
    main()
