"""Authorized two-repeat G6 supervisor. Never restarts an existing failed training run."""
from __future__ import annotations
import argparse,csv,fcntl,hashlib,json,math,os,pickle,shutil,subprocess,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/"outputs/baseline-repair"
PY="/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python"
CONFIG="configs/synapse/deterministic_v2/baseline.yaml"
READY=ROOT/"outputs/determinism/ce2d_stableseg_execution_v2/readiness.json"
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+"\n");temp.replace(path)
def read(p):return json.loads(Path(p).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(pid):
 path=Path("/proc")/str(pid)
 try:
  text=(path/"stat").read_text();tail=text[text.rfind(")")+2:].split()
  return {"pid":int(pid),"start_ticks":tail[19],"state":tail[0],"cmdline":(path/"cmdline").read_bytes().replace(b"\0",b" ").decode(),"cwd":str((path/"cwd").resolve())}
 except (FileNotFoundError,ProcessLookupError,PermissionError):return None
def same_process(record):
 current=identity(record["pid"]) if record else None
 return bool(current and current["state"]!="Z" and current["start_ticks"]==record["start_ticks"] and current["cwd"]==str(ROOT) and current["cmdline"]==record["cmdline"])
def tensors_digest(state):
 import torch
 result={}
 for key,value in sorted(state.items()):
  if isinstance(value,torch.Tensor):
   x=value.detach().cpu().contiguous()
   h=hashlib.sha256(json.dumps({"dtype":str(x.dtype),"shape":list(x.shape)},sort_keys=True).encode())
   h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes());result[key]=h.hexdigest()
  else:result[key]=hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
 return result
from lightning.pytorch import Callback
def compiled_model_active(module):
 return hasattr(module,"_orig_mod") or (getattr(module,"_compiler_ctx",None) or {}).get("compiler")=="dynamo"
class G6InitialState(Callback):
 def __init__(self,output_dir):self.output_dir=Path(output_dir)
 def on_train_start(self,trainer,module):
  import torch,numpy as np,random
  raw=getattr(module,"_orig_mod",module)
  if next(raw.parameters()).device.type!="cuda" or hasattr(raw.network,"asee"):raise RuntimeError("G6 must be CUDA pure baseline")
  write(self.output_dir/"callback_context.json",{"compiled":compiled_model_active(module),"seed":torch.initial_seed(),"max_epochs":trainer.max_epochs,"estimated_steps":int(trainer.estimated_stepping_batches)})
  assert compiled_model_active(module) and torch.initial_seed()==0 and trainer.max_epochs==220 and int(trainer.estimated_stepping_batches)==30360
  assert len(trainer.optimizers[0].param_groups)==373
  rng={"torch":sha_bytes(torch.get_rng_state().numpy().tobytes()),"python":sha_bytes(pickle.dumps(random.getstate())),"numpy":sha_bytes(pickle.dumps(np.random.get_state())),"cuda":[sha_bytes(x.cpu().numpy().tobytes()) for x in torch.cuda.get_rng_state_all()]}
  write(self.output_dir/"startup_verification.json",{"cuda":True,"network":type(raw.network).__module__+"."+type(raw.network).__name__,"profile":"baseline","seed":0,"max_epochs":trainer.max_epochs,"estimated_steps":trainer.estimated_stepping_batches,"initial_state":tensors_digest(raw.state_dict()),"rng":rng,"optimizer_groups":len(trainer.optimizers[0].param_groups),"compute_hd95_voxel":raw.compute_hd95_voxel,"compiled":compiled_model_active(module),"device":str(next(raw.parameters()).device)})
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def coverage(run):
 import torch
 run=Path(run);folders=list((run/"synapse_test_selection").glob("*"));assert len(folders)==1,"Selection directory count"
 folder=folders[0];selection=read(folder/"selection_summary.json")
 assert selection["max_epochs"]==220 and selection["last_evaluated_epoch"]==220
 def rows(n):
  with (folder/n).open(newline="") as f:return list(csv.DictReader(f))
 epochs,cases,organs=[rows(n) for n in ("test_epoch_metrics.csv","test_case_metrics.csv","physical_case_organ_metrics.csv")]
 expected=set((ROOT/"project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt").read_text().splitlines())
 assert len(epochs)==50 and {int(x["epoch"]) for x in epochs}==set(range(171,221))
 assert len(cases)==600 and len(organs)==4800
 for e in range(171,221):
  cr=[x for x in cases if int(x["epoch"])==e];orr=[x for x in organs if int(x["epoch"])==e]
  assert len(cr)==12 and {x["case"] for x in cr}==expected
  assert len(orr)==96 and {(x["case"],int(x["label_id"])) for x in orr}=={(c,k) for c in expected for k in range(1,9)}
 for row in organs:
  assert math.isfinite(float(row["dice"])) and 0<=float(row["dice"])<=1
  assert math.isfinite(float(row["hd95_mm"])) and float(row["hd95_mm"])>=0
  assert row["hd95_voxel"]=="","Voxel mode changed"
 last=torch.load(folder/"last.ckpt",map_location="cpu",weights_only=False,mmap=True)
 assert last["epoch"]+1==220 and last["global_step"]==30360
 best=torch.load(folder/"best_test.ckpt",map_location="cpu",weights_only=False,mmap=True)
 assert best["epoch"]+1==selection["best_epoch"]
 mapped=read(ROOT/"priors/synapse_label_map.verified.json")["classes"];bestrows=[x for x in organs if int(x["epoch"])==selection["best_epoch"]]
 output=[]
 for name in mapped:
  label=name["label_id"];rr=[x for x in bestrows if int(x["label_id"])==label]
  assert len(rr)==12
  output.append({"label_id":label,"organ":name["name_zh"],"dice":sum(float(x["dice"]) for x in rr)/12,"hd95_mm":sum(float(x["hd95_mm"]) for x in rr)/12})
 summary={"run":str(run),"protocol":"ce2d_stableseg_execution_v2","max_epochs":220,"global_step":30360,"test_epochs":50,"cases":600,"case_organs":4800,"best_epoch":selection["best_epoch"],"organs":output,"mean_dice":sum(x["dice"] for x in output)/8,"mean_hd95_mm":sum(x["hd95_mm"] for x in output)/8,"best_checkpoint_sha256":sha(folder/"best_test.ckpt"),"best_state_hashes":tensors_digest(best["state_dict"]),"last_state_hashes":tensors_digest(last["state_dict"]),"selection_dir":str(folder)}
 assert abs(summary["mean_dice"]-selection["best_mean_dice"])<1e-12
 write(run/"summary.json",summary)
 lines=["# G6 pure baseline","",f"best epoch: {summary['best_epoch']}; Dice%: {summary['mean_dice']*100:.6f}; HD95_mm: {summary['mean_hd95_mm']:.6f}","","|label|organ|Dice%|HD95_mm|","|---|---|---|---|"]
 for x in output:lines.append(f"|{x['label_id']}|{x['organ']}|{x['dice']*100:.6f}|{x['hd95_mm']:.6f}|")
 (run/"summary.md").write_text("\n".join(lines)+"\n")
 return summary
def existing(repeat):
 p=BASE/f"repeat_{repeat}.json"
 return read(p) if p.exists() else None
def preflight(repeat):
 import yaml
 from training.deterministic_execution_v2 import require_readiness
 require_readiness(READY,ROOT,"baseline")
 preparation=read(BASE/"preparation.json")
 assert preparation.get("passed") is True and preparation.get("launcher_sha256")==sha(__file__),"G6 launcher preparation not passed/current"
 assert Path(sys.executable).resolve()==Path(PY).resolve()
 d=yaml.safe_load((ROOT/CONFIG).read_text())
 assert d["seed_everything"]==0 and d["trainer"]["max_epochs"]==220 and d["trainer"]["precision"]=="16-mixed"
 assert d["model"]["init_args"]["network"]["class_path"]=="models.eoseg.EoSegMultiQSegFusion"
 assert d["data"]["init_args"]["batch_size"]==16 and d["data"]["init_args"]["num_workers"]==8
 assert d["model"]["init_args"]["compute_hd95_voxel"] is False
 if repeat==2:
  first=existing(1)
  if not first or read(Path(first["run"])/"status.json").get("state")!="completed":raise RuntimeError("First G6 not complete; do not overlap")
  coverage(Path(first["run"]))
 if os.environ.get("CUDA_VISIBLE_DEVICES")=="":raise RuntimeError("Cannot inherit CPU-only GPU visibility")
 free=int(subprocess.check_output(["nvidia-smi","--id=0","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
 if free<22000:raise RuntimeError(f"GPU free {free} MiB, require22000; no other task stopped")
 if shutil.disk_usage(ROOT).free<30*1024**3:raise RuntimeError("Need at least30GiB free disk")
 runtime=json.loads(subprocess.check_output([PY,"-c","import torch,json; ok=torch.cuda.is_available(); print(json.dumps(dict(python=__import__('sys').executable,torch=torch.__version__,cuda=torch.version.cuda,available=ok,gpu=torch.cuda.get_device_name(0) if ok else None)))"],text=True))
 assert runtime["available"] and runtime["torch"]=="2.13.0+cu132" and runtime["cuda"]=="13.2" and "5090" in runtime["gpu"]
 pretrained=Path("/Disk/HDD/Sk/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth")
 pretrained_sha=sha(pretrained)
 assert pretrained_sha=="36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51","Pretrained source drift"
 return {"runtime":runtime,"pretrained_sha256":pretrained_sha,"gpu_free_mib":free,"hostname":subprocess.check_output(["hostname"],text=True).strip(),"launcher_sha256":sha(__file__),"readiness_sha256":sha(READY)}
def compare_repeats():
 a,b=[read(Path(existing(n)["run"])/"summary.json") for n in (1,2)]
 starts=[read(Path(existing(n)["run"])/"startup_verification.json") for n in (1,2)]
 result={"best_epoch_equal":a["best_epoch"]==b["best_epoch"],"best_state_equal":a["best_state_hashes"]==b["best_state_hashes"],"last_state_equal":a["last_state_hashes"]==b["last_state_hashes"],"mean_dice_difference_pp":(b["mean_dice"]-a["mean_dice"])*100,"mean_hd95_mm_difference":b["mean_hd95_mm"]-a["mean_hd95_mm"],"organs":[{"label_id":x["label_id"],"organ":x["organ"],"dice_difference_pp":(y["dice"]-x["dice"])*100,"hd95_mm_difference":y["hd95_mm"]-x["hd95_mm"]} for x,y in zip(a["organs"],b["organs"])]}
 result["initial_state_equal"]=starts[0]["initial_state"]==starts[1]["initial_state"]
 result["initial_rng_equal"]=starts[0]["rng"]==starts[1]["rng"]
 result["initial_optimizer_group_count_equal"]=starts[0]["optimizer_groups"]==starts[1]["optimizer_groups"]
 result["all_epoch_case_metric_csv_equal"]={}
 for file in ("test_epoch_metrics.csv","test_case_metrics.csv","physical_case_organ_metrics.csv"):
  result["all_epoch_case_metric_csv_equal"][file]=sha(Path(a["selection_dir"])/file)==sha(Path(b["selection_dir"])/file)
 write(BASE/"comparison.json",result)
def supervise(repeat,run):
 run=Path(run).resolve();assert run.is_relative_to(BASE.resolve()) and run.parent.name==f"G6_{repeat}"
 with (BASE/"queue.lock").open("a+") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  try:
   evidence=preflight(repeat)
   env=os.environ.copy();env.pop("TORCHINDUCTOR_DETERMINISTIC",None);env.pop("WANDB_RUN_ID",None)
   env.update(HF_HUB_OFFLINE="1",TORCH_HOME="/Disk/HDD/Sk/.cache/torch",TIMM_USE_OLD_CACHE="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONHASHSEED="0",OMP_NUM_THREADS="1",WANDB_MODE="offline",WANDB_RUN_ID=uuid.uuid4().hex,TORCHINDUCTOR_CACHE_DIR=str(run/"cache/inductor"),TRITON_CACHE_DIR=str(run/"cache/triton"))
   cb=[{"class_path":"training.synapse_physical_selection.SynapsePhysicalHD95Selection","init_args":{"last_n_epochs":50,"expected_test_cases":12,"output_dir":str(run),"label_map":"priors/synapse_label_map.verified.json"}},{"class_path":"scripts.launch_synapse_g6.G6InitialState","init_args":{"output_dir":str(run)}}]
   cmd=[PY,"scripts/run_synapse_deterministic_v2.py","fit","--config",CONFIG,"--seed_everything=0","--tta_views=1","--trainer.default_root_dir="+str(run),"--trainer.logger.init_args.save_dir="+str(run),"--trainer.logger.init_args.name=baseline-repair-G6-"+str(repeat),"--trainer.callbacks="+json.dumps(cb),"--deterministic_output_dir="+str(run/"protocol_records")]
   with (run/"train.log").open("w") as log:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    time.sleep(.2);ident=identity(child.pid)
    write(run/"launch.json",{"training_identity":ident,"pid":child.pid,"command":cmd,"environment":{k:env[k] for k in ("HF_HUB_OFFLINE","TORCH_HOME","TIMM_USE_OLD_CACHE","CUBLAS_WORKSPACE_CONFIG","PYTHONHASHSEED","OMP_NUM_THREADS","WANDB_MODE","TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR")},"preflight":evidence})
    write(run/"status.json",{"state":"running","training_pid":child.pid,"repeat":repeat,"protocol":"ce2d_stableseg_execution_v2"})
    code=child.wait()
   write(run/"training_exit.json",{"exit_code":code,"finished_utc":datetime.now(timezone.utc).isoformat()})
   if code:raise RuntimeError(f"Training exited{code}; preserve and do not restart")
   summary=coverage(run)
   write(run/"status.json",{"state":"completed","repeat":repeat,"roi_enabled":True,"summary":str(run/"summary.json")})
   if repeat==2:compare_repeats()
  except BaseException as e:
   write(run/"status.json",{"state":"failed","repeat":repeat,"error":repr(e),"no_auto_restart":True});raise
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--repeat",type=int,choices=(1,2),required=True);parser.add_argument("--check",action="store_true");parser.add_argument("--supervise",action="store_true");parser.add_argument("--run")
 args=parser.parse_args();BASE.mkdir(parents=True,exist_ok=True)
 if args.supervise:return supervise(args.repeat,args.run)
 receipt=existing(args.repeat)
 if receipt:
  run=Path(receipt["run"]);state=read(run/"status.json");launch=read(run/"launch.json") if (run/"launch.json").exists() else {}
  print(json.dumps({"existing":True,"run":str(run),"status":state,"training_process_alive":same_process(launch.get("training_identity")),"no_duplicate_started":True}));return
 if args.check:print(json.dumps({"existing":False,"repeat":args.repeat}));return
 with (BASE/"queue.lock").open("a+") as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:print(json.dumps({"deferred":"Prior G6 supervisor still holds queue lock; no process stopped"}));return
  try:evidence=preflight(args.repeat)
  except Exception as e:
   write(BASE/f"deferred_{args.repeat}.json",{"reason":str(e),"utc":datetime.now(timezone.utc).isoformat()});print(json.dumps({"deferred":str(e)}));return
  run=BASE/f"G6_{args.repeat}"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");run.mkdir(parents=True,exist_ok=False)
  write(run/"status.json",{"state":"launching","repeat":args.repeat})
  with (run/"supervisor.log").open("w") as log:
   child=subprocess.Popen([PY,str(Path(__file__).resolve()),"--repeat",str(args.repeat),"--supervise","--run",str(run)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  receipt={"run":str(run),"repeat":args.repeat,"supervisor_pid":child.pid,"created_utc":datetime.now(timezone.utc).isoformat(),"preflight":evidence}
  write(BASE/f"repeat_{args.repeat}.json",receipt);print(json.dumps(receipt))
if __name__=="__main__":main()
