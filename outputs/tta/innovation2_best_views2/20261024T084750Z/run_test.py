from pathlib import Path
import csv,datetime as dt,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path('/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse');RUN=Path('/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/tta/innovation2_best_views2/20261024T084750Z')
ENV={"HF_HUB_OFFLINE":"1","TORCH_HOME":"/Disk/HDD/Sk/.cache/torch","TIMM_USE_OLD_CACHE":"1"}
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def write(p,data):
 temp=p.with_suffix(p.suffix+".tmp");temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(p)
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
meta=json.loads((RUN/"source_checkpoint.json").read_text())
status={"state":"starting","started_at_utc":now(),"supervisor_pid":os.getpid(),"tta_views":2,"checkpoint_epoch":187,"output_root":str(RUN)}
write(RUN/"status.json",status)
try:
 apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
 if apps:raise RuntimeError("GPU is busy: "+apps)
 checkpoint=Path(meta["checkpoint"])
 if sha(checkpoint)!=meta["checkpoint_sha256"]:raise RuntimeError("Checkpoint hash changed before launch")
 cb=[{"class_path":"training.synapse_tta.SynapseTTASelection","init_args":{"expected_test_cases":12,"output_dir":str(RUN)}}]
 cmd=[sys.executable,"-u","scripts/evaluate_synapse_tta.py","test","--config","configs/synapse/vit_query_mul_scale_fusion_asee_ccps_tta.yaml","--ckpt_path",str(checkpoint),"--compile_test","--tta_views=2","--trainer.accelerator=gpu","--trainer.devices=1","--trainer.logger=false","--trainer.enable_progress_bar=false","--trainer.default_root_dir="+str(RUN),"--trainer.callbacks="+json.dumps(cb)]
 write(RUN/"launch.json",{"command":cmd,"environment":ENV,"source":meta,"supervisor_sha256":sha(Path(__file__))})
 with (RUN/"test.log").open("w") as log:
  process=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,**ENV},stdout=log,stderr=subprocess.STDOUT)
  status.update(state="running",testing_pid=process.pid);write(RUN/"status.json",status)
  code=process.wait()
 status["test_exit_code"]=code
 if code:raise RuntimeError("2-view test exited "+str(code))
 output=next(RUN.rglob("physical_test_summary.json")).parent
 result=json.loads((output/"physical_test_summary.json").read_text())
 if result["num_cases"]!=12 or result["checkpoint_epoch"]!=187 or result["checkpoint_sha256"]!=meta["checkpoint_sha256"]:
  raise RuntimeError("Evaluation checkpoint/case count mismatch")
 if not (output/"run_manifest.json").is_file() or not (output/"readiness_snapshot.json").is_file():raise RuntimeError("Missing test provenance")
 before={int(x["label_id"]):x for x in rows(Path(meta["single_view_reference"])/"physical_epoch_organ_metrics.csv") if int(x["epoch"])==187}
 after=rows(output/"physical_epoch_organ_metrics.csv")
 if len(before)!=8 or len(after)!=8:raise RuntimeError("Expected eight organ means")
 named=[]
 for row in sorted(after,key=lambda x:int(x["label_id"])):
  c=int(row["label_id"]);ref=before[c]
  named.append({"label_id":c,"organ_zh":row["organ_zh"],"dice_percent":float(row["dice_percent"]),"hd95_mm":float(row["hd95_mm"]),"baseline_dice_percent":float(ref["dice_percent"]),"baseline_hd95_mm":float(ref["hd95_mm"]),"dice_delta_pp":float(row["dice_percent"])-float(ref["dice_percent"]),"hd95_delta_mm":float(row["hd95_mm"])-float(ref["hd95_mm"]),"legacy_empty_cases":int(row["legacy_empty_cases"])})
 base_cases={(x["case"],int(x["label_id"])):x for x in rows(Path(meta["single_view_reference"])/"physical_case_organ_metrics.csv") if int(x["epoch"])==187}
 current=rows(output/"physical_case_organ_metrics.csv")
 if len(current)!=96 or {(x["case"],int(x["label_id"])) for x in current}!=set(base_cases):raise RuntimeError("Case/label set mismatch")
 cross={x["case"]:x for x in rows(output/"kidney_confusion.csv")};kidneys=[]
 for row in current:
  c=int(row["label_id"])
  if c not in (3,4):continue
  ref=base_cases[(row["case"],c)];comparable=row["status"]==ref["status"]=="nonempty"
  dd=float(row["dice_percent"])-float(ref["dice_percent"]);dh=float(row["hd95_mm"])-float(ref["hd95_mm"])
  kidneys.append({"case":row["case"],"label_id":c,"organ_zh":row["organ_zh"],"dice_percent":float(row["dice_percent"]),"baseline_dice_percent":float(ref["dice_percent"]),"dice_delta_pp":dd,"dice_degraded":dd<0,"hd95_mm":float(row["hd95_mm"]),"baseline_hd95_mm":float(ref["hd95_mm"]),"hd95_delta_mm":dh,"hd95_comparable":comparable,"hd95_degraded":dh>0 if comparable else None,"status":row["status"],"baseline_status":ref["status"],"pred_on_opposite_kidney_voxels":int(cross[row["case"]]["pred_left_on_gt_right_voxels" if c==3 else "pred_right_on_gt_left_voxels"]),"baseline_cross_kidney_voxels":None})
 for name,data in [("organ_comparison.csv",named),("kidney_case_comparison.csv",kidneys)]:
  with (RUN/name).open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 if sha(checkpoint)!=meta["checkpoint_sha256"]:raise RuntimeError("Checkpoint changed during evaluation")
 summary={"checkpoint":meta,"mean_dice_percent":result["mean_dice"]*100,"mean_hd95_mm":result["mean_hd95_mm"],"mean_hd95_voxel":result["mean_hd95_voxel"],"baseline_dice_percent":sum(float(x["dice_percent"]) for x in before.values())/8,"baseline_hd95_mm":sum(float(x["hd95_mm"]) for x in before.values())/8,"organs":named,"kidneys":kidneys,"kidney_note":"Original no-TTA run did not record cross-kidney voxel counts; no baseline counts or count deltas fabricated."}
 write(RUN/"results_summary.json",summary)
 text="# Innovation2 最佳第187轮：2视角TTA\n\n"
 text+="原图+水平翻转，完整12例，原FP16/compile、真实mm HD95及legacy空掩码规则。来源训练用户停止于214轮/完整测试213轮，不声称完成220轮。\n\n"
 text+=f"平均Dice={summary['mean_dice_percent']:.6f}%，平均HD95={summary['mean_hd95_mm']:.6f}mm；无TTA参考Dice={summary['baseline_dice_percent']:.6f}%，HD95={summary['baseline_hd95_mm']:.6f}mm。\n\n"
 text+="|ID|器官|Dice (%)|HD95 (mm)|Dice变化(pp)|HD95变化(mm)|\n|---|---|---:|---:|---:|---:|\n"
 for x in named:text+=f"|{x['label_id']}|{x['organ_zh']}|{x['dice_percent']:.4f}|{x['hd95_mm']:.4f}|{x['dice_delta_pp']:+.4f}|{x['hd95_delta_mm']:+.4f}|\n"
 text+="\n左右肾逐病例变化见kidney_case_comparison.csv。原无TTA未采集对侧误分体素数，因此不伪造其基线/变化；本组计数保留在原始kidney_confusion.csv。空mask的HD95=0仅为原规则占位，标记不可比较。\n"
 (RUN/"results_summary.md").write_text(text)
 status.update(state="completed",exit_code=0,finished_at_utc=now(),result_folder=str(output))
except Exception as exc:
 status.update(state="failed",exit_code=1,finished_at_utc=now(),error=str(exc),traceback=traceback.format_exc())
write(RUN/"status.json",status)
