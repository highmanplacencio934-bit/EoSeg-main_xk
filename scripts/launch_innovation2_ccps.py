"""One-shot, locked CCPS launch after readiness; never schedules or changes parameters."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/innovation2/ccps_plan"
PYTHON="/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python"
ENV={"HF_HUB_OFFLINE":"1","TORCH_HOME":"/Disk/HDD/Sk/.cache/torch","TIMM_USE_OLD_CACHE":"1","WANDB_MODE":"offline"}

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)

def preflight():
    os.chdir(ROOT)
    if os.path.realpath(sys.executable)!=os.path.realpath(PYTHON):raise RuntimeError("Wrong interpreter")
    if socket.gethostname()!="5090":raise RuntimeError("Wrong host")
    ready=json.loads((ROOT/"outputs/innovation2/ccps_readiness.json").read_text())
    if ready.get("state")!="passed":raise RuntimeError("Readiness not passed")
    for rel,digest in ready["sha256"].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("Readiness source/artifact changed: "+rel)
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
    if apps:raise RuntimeError("GPU has compute processes; refusing contention: "+apps)
    info=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.free,driver_version","--format=csv,noheader,nounits"],text=True).strip()
    parts=[x.strip() for x in info.split(",")]
    if len(parts)!=3 or "5090" not in parts[0] or int(parts[1])<28000:raise RuntimeError("Unexpected GPU/free memory: "+info)
    if shutil.disk_usage(ROOT).free<15*2**30:raise RuntimeError("Less than 15GiB free disk")
    import torch
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    return {"hostname":socket.gethostname(),"python":sys.executable,"torch":torch.__version__,
            "cuda_runtime":torch.version.cuda,"nvidia_smi":info,"disk_free_bytes":shutil.disk_usage(ROOT).free,
            "readiness_sha256":hashlib.sha256((ROOT/"outputs/innovation2/ccps_readiness.json").read_bytes()).hexdigest()}

def supervise(run):
    status={"state":"starting","supervisor_pid":os.getpid(),"started_at_utc":now(),"output_root":str(run)}
    write(run/"status.json",status)
    try:
        env_report=preflight()
        cb=[{"class_path":"training.synapse_physical_selection.SynapsePhysicalHD95Selection",
             "init_args":{"last_n_epochs":50,"expected_test_cases":12,"output_dir":str(run)}}]
        cmd=[PYTHON,"-u","main.py","fit","--config","configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml",
             "--trainer.accelerator=gpu","--trainer.devices=1","--trainer.default_root_dir="+str(run),
             "--trainer.callbacks="+json.dumps(cb)]
        write(run/"launch.json",{"command":cmd,"environment":ENV,"preflight":env_report,
             "started_at_utc":now(),"initialization":"original offline timm pretrain; no innovation1 checkpoint",
             "hd95_unit":"mm; named reports retain voxel","selection":"mean Dice, original legacy empty rule"})
        with (run/"train.log").open("w") as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,**ENV},stdout=log,stderr=subprocess.STDOUT)
            status.update(state="running",training_pid=child.pid)
            write(run/"status.json",status)
            code=child.wait()
        status.update(state="completed" if code==0 else "failed",exit_code=code,finished_at_utc=now())
    except Exception as exc:
        status.update(state="failed",exit_code=1,error=str(exc),traceback=traceback.format_exc(),finished_at_utc=now())
    write(run/"status.json",status)

def launch():
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/"launch.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=BASE/"launch_receipt.json"
        if receipt.exists():
            old=json.loads(receipt.read_text())
            path=Path(old["output_root"])/"status.json"
            print(json.dumps({"already_launched":True,"receipt":old,"status":json.loads(path.read_text()) if path.exists() else "pending"},ensure_ascii=False))
            return
        preflight()
        stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run=BASE/stamp;run.mkdir(exist_ok=False)
        write(receipt,{"output_root":str(run),"requested_at_utc":now()})
        (BASE/"latest_path.txt").write_text(str(run.relative_to(ROOT)))
        with (run/"supervisor.log").open("w") as log:
            proc=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),"--supervise",str(run)],cwd=ROOT,
                env={**os.environ,**ENV},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({"output_root":str(run),"supervisor_pid":proc.pid},ensure_ascii=False))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--supervise",type=Path);parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    if args.supervise:supervise(args.supervise)
    elif args.check_only:print(json.dumps(preflight(),ensure_ascii=False))
    else:launch()
