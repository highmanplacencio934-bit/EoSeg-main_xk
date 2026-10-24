"""Locked, independent 1/2/4/8-view comparison after successful innovation2 training."""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/tta/innovation2"
PYTHON="/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python"
ENV={"HF_HUB_OFFLINE":"1","TORCH_HOME":"/Disk/HDD/Sk/.cache/torch","TIMM_USE_OLD_CACHE":"1"}
VIEWS=(1,2,4,8)

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda:stream.read(8*1024**2),b""):h.update(data)
    return h.hexdigest()
def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)
def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def completed_training():
    source=ROOT/(ROOT/"outputs/innovation2/ccps_plan/latest_path.txt").read_text().strip()
    status=json.loads((source/"status.json").read_text())
    if status.get("state")!="completed" or status.get("exit_code")!=0:
        raise RuntimeError("Training not successfully completed: "+status.get("state","unknown"))
    selection=next(source.rglob("selection_summary.json")).parent
    info=json.loads((selection/"selection_summary.json").read_text())
    epochs=rows(selection/"test_epoch_metrics.csv")
    if [int(x["epoch"]) for x in epochs]!=list(range(171,221)) or info["last_evaluated_epoch"]!=220:
        raise RuntimeError("Full 220-epoch / final-50 evaluation verification failed")
    cases=rows(selection/"test_case_metrics.csv")
    if len(cases)!=600 or any(sum(int(x["epoch"])==e for x in cases)!=12 for e in range(171,221)):
        raise RuntimeError("Incomplete training case evaluations")
    return source,selection,info

def resources_and_readiness():
    if socket.gethostname()!="5090" or os.path.realpath(sys.executable)!=os.path.realpath(PYTHON):
        raise RuntimeError("Wrong host/interpreter")
    ready=json.loads((ROOT/"outputs/tta/tta_readiness.json").read_text())
    if ready.get("state")!="passed":raise RuntimeError("TTA CPU/CLI readiness not passed")
    for rel,h in ready["sha256"].items():
        if digest(ROOT/rel)!=h:raise RuntimeError("Source changed after TTA readiness: "+rel)
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
    if apps:raise RuntimeError("GPU busy, wait without starting: "+apps)
    return ready

def verify_identity(selection,output,best):
    reference={(x["case"],int(x["label_id"])):x for x in rows(selection/"physical_case_organ_metrics.csv") if int(x["epoch"])==best}
    current=rows(output/"physical_case_organ_metrics.csv")
    if len(reference)!=96 or len(current)!=96:raise RuntimeError("Identity replay needs 12*8 organ rows")
    differences={metric:0. for metric in ("dice","hd95_mm","hd95_voxel")}
    for row in current:
        ref=reference[(row["case"],int(row["label_id"]))]
        for metric in differences:differences[metric]=max(differences[metric],abs(float(row[metric])-float(ref[metric])))
    if max(differences.values())>1e-10:raise RuntimeError("Original one-view replay mismatch: "+str(differences))
    return differences

def kidney_case_changes(reports):
    by_view={}
    for views,folder,_ in reports:
        data=[x for x in rows(folder/"physical_case_organ_metrics.csv") if int(x["label_id"]) in (3,4)]
        index={(x["case"],int(x["label_id"])):x for x in data}
        cases={case for case,_ in index}
        cross_rows=rows(folder/"kidney_confusion.csv")
        cross={x["case"]:x for x in cross_rows}
        if len(data)!=24 or len(index)!=24 or len(cases)!=12 or any((c,k) not in index for c in cases for k in (3,4)):
            raise ValueError("Kidney comparison needs 12 distinct cases with both label3 and label4")
        if len(cross_rows)!=12 or len(cross)!=12 or set(cross)!=cases:
            raise ValueError("Kidney confusion case IDs must match organ metrics")
        by_view[views]=(index,cross)
    baseline=by_view[1][0]
    result=[]
    for views,(index,cross) in by_view.items():
        if set(index)!=set(baseline):raise ValueError("TTA kidney case/label IDs do not match one-view baseline")
        for key in sorted(index):
            case,label=key;row=index[key];ref=baseline[key]
            count_key="pred_left_on_gt_right_voxels" if label==3 else "pred_right_on_gt_left_voxels"
            count=int(cross[case][count_key]);ref_count=int(by_view[1][1][case][count_key])
            delta_dice=float(row["dice_percent"])-float(ref["dice_percent"])
            delta_hd=float(row["hd95_mm"])-float(ref["hd95_mm"])
            comparable=row["status"]==ref["status"]=="nonempty"
            result.append({"views":views,"case":case,"label_id":label,"organ_zh":row["organ_zh"],
                "dice_percent":float(row["dice_percent"]),"baseline_dice_percent":float(ref["dice_percent"]),
                "dice_delta_pp":delta_dice,"dice_degraded":delta_dice<0,
                "hd95_mm":float(row["hd95_mm"]),"baseline_hd95_mm":float(ref["hd95_mm"]),
                "hd95_delta_mm":delta_hd,"hd95_comparable":comparable,
                "hd95_degraded":delta_hd>0 if comparable else None,
                "status":row["status"],"baseline_status":ref["status"],
                "pred_on_opposite_kidney_voxels":count,"cross_kidney_voxels_delta":count-ref_count,
                "gt_left_voxels":int(cross[case]["gt_left_voxels"]),
                "gt_right_voxels":int(cross[case]["gt_right_voxels"])})
    return result


def make_report(run,reports):
    totals=[];organs=[];kidneys=[]
    case_changes=kidney_case_changes(reports)
    for views,folder,seconds in reports:
        report=json.loads((folder/"physical_test_summary.json").read_text())
        manifest=folder/"run_manifest.json"
        if not manifest.is_file():raise ValueError("Missing per-view TTA run_manifest")
        totals.append({"views":views,"dice_percent":report["mean_dice"]*100,
            "hd95_mm":report["mean_hd95_mm"],"hd95_voxel":report["mean_hd95_voxel"],
            "process_seconds":seconds,"test_seconds":report["test_seconds"],
            "run_manifest":str(manifest),"run_manifest_sha256":digest(manifest)})
        data=rows(folder/"physical_epoch_organ_metrics.csv")
        if len(data)!=8:raise RuntimeError("Expected eight named foreground organs")
        organs.extend({"views":views,**row} for row in data)
        for organ_id in (3,4):
            row=next(x for x in data if int(x["label_id"])==organ_id)
            per_case=[x for x in case_changes if x["views"]==views and x["label_id"]==organ_id]
            kidneys.append({"views":views,"label_id":organ_id,"organ_zh":row["organ_zh"],
                "dice_percent":float(row["dice_percent"]),"hd95_mm":float(row["hd95_mm"]),
                "pred_on_opposite_kidney_voxels":sum(x["pred_on_opposite_kidney_voxels"] for x in per_case),
                "cross_kidney_voxels_delta":sum(x["cross_kidney_voxels_delta"] for x in per_case),
                "dice_degraded_cases":sum(x["dice_degraded"] for x in per_case),
                "hd95_degraded_cases":sum(x["hd95_degraded"] is True for x in per_case),
                "hd95_noncomparable_cases":sum(not x["hd95_comparable"] for x in per_case)})
    identity=next(x for x in totals if x["views"]==1)
    for row in totals:
        row["dice_delta_pp"]=row["dice_percent"]-identity["dice_percent"]
        row["hd95_delta_mm"]=row["hd95_mm"]-identity["hd95_mm"]
    baseline={x["label_id"]:x for x in kidneys if x["views"]==1}
    for row in kidneys:
        ref=baseline[row["label_id"]]
        row["baseline_dice_percent"]=ref["dice_percent"]
        row["baseline_hd95_mm"]=ref["hd95_mm"]
        row["dice_delta_pp"]=row["dice_percent"]-ref["dice_percent"]
        row["hd95_delta_mm"]=row["hd95_mm"]-ref["hd95_mm"]
    degraded=[x for x in case_changes if x["views"]!=1 and (x["dice_degraded"] or x["hd95_degraded"] is True)]
    report={"checkpoint":json.loads((run/"source_checkpoint.json").read_text()),"views":totals,
        "best_mean_dice_views":max(totals,key=lambda x:x["dice_percent"])["views"],
        "best_mean_hd95_views":min(totals,key=lambda x:x["hd95_mm"])["views"],"kidneys":kidneys,
        "kidney_case_changes":case_changes,"kidney_degraded_cases":degraded,
        "interpretation":"Same fixed checkpoint; Dice/HD95 winners separate. Kidney deltas join case+label to one-view. Degradation flags mean strict numeric changes, not significance; HD95 legacy empty zeros are noncomparable."}
    write(run/"comparison_summary.json",report)
    for filename,data,fields in [
        ("comparison_organs.csv",organs,list(organs[0])),
        ("kidney_comparison.csv",kidneys,list(kidneys[0])),
        ("kidney_case_comparison.csv",case_changes,list(case_changes[0])),
        ("kidney_degraded_cases.csv",degraded,list(case_changes[0]))]:
        with (run/filename).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)
    text="# Innovation2 固定最佳checkpoint：TTA对比\n\n"
    text+="|视角|平均Dice (%)|HD95 (mm)|Dice变化(pp)|HD95变化(mm)|测试分钟|\n|---|---:|---:|---:|---:|---:|\n"
    for x in totals:text+=f"|{x['views']}|{x['dice_percent']:.6f}|{x['hd95_mm']:.6f}|{x['dice_delta_pp']:+.6f}|{x['hd95_delta_mm']:+.6f}|{x['test_seconds']/60:.2f}|\n"
    text+="\nDice与HD95分别比较，不保证同一视角同时最好。8视角包含未用于训练增强的90度旋转；左右肾分开核查，不交换通道。保留原空掩码规则，胆囊空GT的HD95零值为占位。\n\n"
    for views in VIEWS:
        text+=f"## {views}视角分器官\n\n|ID|器官|Dice (%)|HD95 (mm)|\n|---|---|---:|---:|\n"
        for x in sorted([x for x in organs if x['views']==views],key=lambda x:int(x['label_id'])):
            text+=f"|{x['label_id']}|{x['organ_zh']}|{float(x['dice_percent']):.4f}|{float(x['hd95_mm']):.4f}|\n"
        text+="\n"
    text+="## 左右肾相对单视角的变化\n\n|视角|器官|Dice (%)|变化(pp)|HD95 (mm)|变化(mm)|Dice退化病例|HD95退化病例|HD95不可比较病例|对侧误分体素|\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for x in kidneys:text+=f"|{x['views']}|{x['organ_zh']}|{x['dice_percent']:.4f}|{x['dice_delta_pp']:+.4f}|{x['hd95_mm']:.4f}|{x['hd95_delta_mm']:+.4f}|{x['dice_degraded_cases']}|{x['hd95_degraded_cases']}|{x['hd95_noncomparable_cases']}|{x['pred_on_opposite_kidney_voxels']}|\n"
    text+="\n## 左右肾逐病例退化\n\n严格数值变化，不作统计显著性判断。空mask的HD95占位零值仅保留原数值，不作改善/退化判定。完整逐病例对照见kidney_case_comparison.csv，退化子集见kidney_degraded_cases.csv。\n\n"
    text+="|视角|病例|器官|Dice变化(pp)|HD95变化(mm)|HD95可比较|\n|---|---|---|---:|---:|---|\n"
    for x in degraded:text+=f"|{x['views']}|{x['case']}|{x['organ_zh']}|{x['dice_delta_pp']:+.4f}|{x['hd95_delta_mm']:+.4f}|{'是' if x['hd95_comparable'] else '否'}|\n"
    if not degraded:text+="\n没有检出严格数值退化的肾脏病例。\n"
    (run/"comparison_summary.md").write_text(text)

def worker(run):
    state={"state":"running","supervisor_pid":os.getpid(),"started_at_utc":now(),"completed_views":[]}
    write(run/"status.json",state)
    try:
        resources_and_readiness();source,selection,info=completed_training()
        checkpoint=selection/"best_test.ckpt";checkpoint_hash=digest(checkpoint)
        write(run/"source_checkpoint.json",{"path":str(checkpoint),"sha256":checkpoint_hash,"epoch":info["best_epoch"],"source_training":str(source)})
        reports=[]
        for views in VIEWS:
            resources_and_readiness()
            out=run/f"views_{views}";out.mkdir()
            callbacks=[{"class_path":"training.synapse_tta.SynapseTTASelection","init_args":{"expected_test_cases":12,"output_dir":str(out)}}]
            cmd=[PYTHON,"-u","scripts/evaluate_synapse_tta.py","test","--config","configs/synapse/vit_query_mul_scale_fusion_asee_ccps_tta.yaml",
                "--ckpt_path",str(checkpoint),"--compile_test",f"--tta_views={views}","--trainer.accelerator=gpu","--trainer.devices=1",
                "--trainer.default_root_dir="+str(out),"--trainer.enable_progress_bar=false","--trainer.callbacks="+json.dumps(callbacks)]
            write(out/"launch.json",{"command":cmd,"environment":ENV,"checkpoint_sha256":checkpoint_hash})
            start=time.perf_counter()
            with (out/"test.log").open("w") as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,**ENV},stdout=log,stderr=subprocess.STDOUT)
                state.update(active_views=views,testing_pid=proc.pid);write(run/"status.json",state)
                code=proc.wait()
            if code:raise RuntimeError(f"{views}-view evaluation exited {code}")
            summary_file=next(out.rglob("physical_test_summary.json"));folder=summary_file.parent
            summary=json.loads(summary_file.read_text())
            if summary["num_cases"]!=12 or summary["checkpoint_sha256"]!=checkpoint_hash or summary["checkpoint_epoch"]!=info["best_epoch"]:raise RuntimeError("Checkpoint/case count mismatch")
            if views==1:write(run/"one_view_replay.json",{"passed":True,"max_absolute_differences":verify_identity(selection,folder,info["best_epoch"])})
            reports.append((views,folder,time.perf_counter()-start));state["completed_views"].append(views);write(run/"status.json",state)
            if digest(checkpoint)!=checkpoint_hash:raise RuntimeError("Source checkpoint changed")
        make_report(run,reports);state.update(state="completed",exit_code=0,finished_at_utc=now())
    except Exception as exc:state.update(state="failed",exit_code=1,error=str(exc),traceback=traceback.format_exc(),finished_at_utc=now())
    write(run/"status.json",state)

def launch():
    completed_training();resources_and_readiness();BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/"launch.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=BASE/"launch_receipt.json"
        if receipt.exists():print(receipt.read_text());return
        run=BASE/dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ");run.mkdir(exist_ok=False)
        write(receipt,{"output_root":str(run),"requested_at_utc":now()})
        (BASE/"latest_path.txt").write_text(str(run.relative_to(ROOT)))
        with (run/"supervisor.log").open("w") as log:
            p=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),"--worker",str(run)],cwd=ROOT,
                env={**os.environ,**ENV},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({"output_root":str(run),"supervisor_pid":p.pid}))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--worker",type=Path);parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.check_only:
        source,selection,info=completed_training();resources_and_readiness();print(json.dumps(info))
    else:launch()
