from pathlib import Path
from datetime import datetime,timezone
import csv,fcntl,hashlib,json,os,subprocess,traceback
ROOT=Path('/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse');RUN=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,x):
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
status={'state':'starting','run_name':'baseline-2','roi_enabled':True,'supervisor_pid':os.getpid(),'started_at_utc':now(),'output_root':str(RUN)};write(RUN/'status.json',status)
try:
 with (RUN/'run.lock').open('a+') as local,(ROOT/'outputs/ablations/journal_20261025/queue.lock').open('a+') as shared:
  fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
  launch=read(RUN/'launch.json')
  for name,digest in launch['source_sha256'].items():assert sha(ROOT/name)==digest,name
  ready=read(ROOT/launch['roi_readiness']);assert ready['status']=='passed' and sha(ROOT/launch['roi_readiness'])==launch['roi_readiness_sha256']
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=launch['required_free_mib'],'Insufficient free memory; do not stop other users'
  env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None);env.update(launch['environment'])
  with (RUN/'train.log').open('w') as log:
   child=subprocess.Popen(launch['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   process=Path('/proc')/str(child.pid);ticks=int((process/'stat').read_text().split(') ',1)[1].split()[19])
   status.update(state='training',training_pid=child.pid,training_started_at_utc=now(),training_identity={'pid':child.pid,'start_ticks':ticks,'cwd':str(ROOT),'command':launch['command']});write(RUN/'status.json',status)
   code=child.wait()
  write(RUN/'training_exit.json',{'exit_code':code,'finished_at_utc':now()});status['exit_code']=code
  if code!=0:
   status.update(state='exited_nonzero',reason='Inspect actual user stop or failure; no automatic restart',finished_at_utc=now());write(RUN/'status.json',status);raise SystemExit(code if code>0 else 1)
  folders=list((RUN/'synapse_test_selection').glob('*'));assert len(folders)==1;folder=folders[0]
  selection=read(folder/'selection_summary.json');assert selection['max_epochs']==220 and selection['last_evaluated_epoch']==220
  epochs=list(csv.DictReader((folder/'test_epoch_metrics.csv').open()));cases=list(csv.DictReader((folder/'test_case_metrics.csv').open()));organ_rows=list(csv.DictReader((folder/'physical_epoch_organ_metrics.csv').open()));case_organs=list(csv.DictReader((folder/'physical_case_organ_metrics.csv').open()))
  expected=(ROOT/'project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt').read_text().splitlines();window=set(range(171,221))
  assert len(epochs)==50 and {int(x['epoch']) for x in epochs}==window and len(cases)==600 and len(organ_rows)==400 and len(case_organs)==4800
  for epoch in window:assert {x['case'] for x in cases if int(x['epoch'])==epoch}==set(expected)
  best=selection['best_epoch'];organs=[x for x in organ_rows if int(x['epoch'])==best];assert len(organs)==8 and {int(x['label_id']) for x in organs}==set(range(1,9))
  names={x['label_id']:x for x in read(ROOT/'priors/synapse_label_map.verified.json')['classes']}
  for row in organs:
   label=int(row['label_id']);assert row['organ']==names[label]['name'] and row['organ_zh']==names[label]['name_zh']
  event=next(x for x in epochs if int(x['epoch'])==best)
  original=read(ROOT/launch['reference_run']/'synapse_test_selection/20261021T132645Z_81f3214a/selection_summary.json')
  report={'status':'completed','run_name':'baseline-2','roi_enabled':True,'seed':0,'max_epochs':220,'best_epoch':best,'mean_dice':selection['best_mean_dice'],'mean_dice_percent':100*selection['best_mean_dice'],'mean_hd95_mm':float(event['mean_hd95']),'mean_hd95_voxel':sum(float(r['hd95_voxel']) for r in organs)/8,'organs':organs,'reference_mean_dice_percent':100*original['best_mean_dice'],'dice_difference_pp':100*(selection['best_mean_dice']-original['best_mean_dice']),'checkpoint':str(folder/'best_test.ckpt'),'checkpoint_sha256':sha(folder/'best_test.ckpt'),'selection_folder':str(folder.relative_to(ROOT)),'test_epoch_records':50,'test_case_records':600,'case_organ_records':4800,'metric_policy':'unchanged legacy Dice; verified physical HD95 mm plus voxel; single-view selection','finished_at_utc':now()}
  write(RUN/'summary.json',report)
  lines=['# baseline-2：seed0纯EoSeg＋ROI指标加速','',f"完整220轮；best epoch={best}；平均Dice={report['mean_dice_percent']:.6f}%；相对首轮={report['dice_difference_pp']:+.6f}个百分点。",'', '| 标签ID | 器官 | Dice（%） | HD95（mm） | HD95（voxel） |','|---|---|---:|---:|---:|']
  for row in sorted(organs,key=lambda x:int(x['label_id'])):lines.append(f"| {row['label_id']} | {row['organ_zh']} | {float(row['dice_percent']):.6f} | {float(row['hd95_mm']):.6f} | {float(row['hd95_voxel']):.6f} |")
  lines.extend([f"| — | 8器官平均 | {report['mean_dice_percent']:.6f} | {report['mean_hd95_mm']:.6f} | {report['mean_hd95_voxel']:.6f} |",'', '同一best、完整12病例、原legacy空mask规则；原训练参数/数据/增强保持，ROI仅加速评估。没有按TTA重新选权重。'])
  (RUN/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');status.update(state='completed',finished_at_utc=now(),summary=str(RUN/'summary.json'));write(RUN/'status.json',status)
except SystemExit:raise
except BaseException as error:
 status.update(state='failed',error=type(error).__name__+': '+str(error),finished_at_utc=now());write(RUN/'status.json',status);traceback.print_exc();raise
