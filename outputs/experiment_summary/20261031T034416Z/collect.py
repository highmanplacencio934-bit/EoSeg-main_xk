"""Read-only catalog of formal Synapse training and independent tests; writes only this report directory."""
from pathlib import Path
import csv,json,os,sys,time,zipfile,math,hashlib
from datetime import datetime,timezone
from collections import Counter
from statistics import fmean
import torch
ROOT=Path('/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse');OUT=ROOT/'outputs'
sys.path.insert(0,str(ROOT))
from scripts.launch_synapse_g6 import identity,same_process
DEST=Path(__file__).resolve().parent
LABELS=sorted(json.loads((ROOT/'priors/synapse_label_map.verified.json').read_text())['classes'],key=lambda x:x['label_id'])
CASES=set((ROOT/'project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt').read_text().splitlines())
def read(p):
 try:return json.loads(Path(p).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp(p):
 s=Path(p).stat();return s.st_size,s.st_mtime_ns
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def num(x):
 if x is None or x=='':return None
 value=float(x);assert math.isfinite(value);return value
def avg(values):
 values=list(values);return fmean(values) if values and all(v is not None for v in values) else None
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def writecsv(p,data,fields):
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)
def load_stable(p):
 for _ in range(5):
  try:
   a=stamp(p)
   if not zipfile.is_zipfile(p):time.sleep(.3);continue
   d=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
   if a==stamp(p):return d
  except (FileNotFoundError,RuntimeError,EOFError,OSError):pass
  time.sleep(.3)
 raise RuntimeError('Unstable checkpoint '+str(p))
def rep(p):return str(Path(p).relative_to(ROOT))
def mdlink(p):return '['+rep(p)+'](<'+'//sshfs.r/sk@anhuiuniversity.icu!21103'+str(p)+'>)'
def fmt(x,pct=False):return '—' if x is None else format(x*(100 if pct else 1),'.6f')
aliases={
 'outputs/baseline/20261021T132635Z':'原始baseline',
 'outputs/baseline-2/20261026T130228Z':'baseline-2（ROI版）',
 'outputs/baseline-lock/20261030T031914Z':'baseline-lock 第1次',
 'outputs/baseline-lock/repeat_2/20261030T080537Z':'baseline-lock 第2次',
 'outputs/baseline-repair/G6_1/20261028T072539Z':'baseline-repair 第1次',
 'outputs/baseline-repair_2/G6_2/20261029T022332Z':'baseline-repair_2 第2次',
 'outputs/innovation1/asee_plan/20261022T044551Z':'ASEE双先验主实验（原CUDA）',
 'outputs/asee-repair/repeat_1/20261029T075801Z':'ASEE双先验主实验（V2修复）',
 'outputs/innovation2/ccps_plan/20261023T121754Z':'ASEE+CCPS seed0（原CUDA）',
 'outputs/innovation2/ccps_seed42/20261024T131734Z':'ASEE+CCPS seed42（原CUDA）',
 'outputs/innovation2/ccps_low_lr_2e-5_100ep/20261028T124208Z':'ASEE+CCPS 低LR补训100轮（V2）',
 'outputs/ablations/journal_20261025/S1/20261025T230957Z/training':'S1仅文本（原CUDA）',
 'outputs/ablations/journal_20261025/S2/20261026T231139Z/training':'S2仅共现（原CUDA，早停）',
 'outputs/ablations/asee_text_only_v2/20261030T125953Z':'S1仅文本（V2）',
 'outputs/ablations/asee_probability_only_v2/20261030T185908Z':'S2仅共现（V2）',
 'outputs/experiments/asee_ccps_full_v2/20261030T233233Z':'ASEE+CCPS完整主模型（V2）'}
catalog=[];excluded=[];orgtable=[];evidence=[]
for p in sorted(OUT.rglob('selection_summary.json')):
 if DEST in p.parents:continue
 s=read(p)
 if not s or s.get('max_epochs',0)<50:
  excluded.append(dict(path=rep(p),reason='冒烟/短测，无正式训练预算'));continue
 sel=p.parent;run=sel.parent.parent;runrel=rep(run)
 if runrel not in aliases:raise RuntimeError('Unclassified formal run '+runrel)
 docs=[]
 for parent in [run,run.parent]:
  for n in ['status.json','training_exit.json','launch.json','startup_verification.json','early_stop_report.json']:
   q=parent/n;d=read(q)
   if d is not None:docs.append((q,d))
 state=next((d for q,d in docs if q.name=='status.json'),{})
 exitdoc=next((d for q,d in docs if q.name=='training_exit.json'),{})
 startup=next((d for q,d in docs if q.name=='startup_verification.json'),{})
 mp=sel/'run_manifest.json'
 if not mp.exists():mp=run/'run_manifest.json'
 manifest=read(mp) or {}
 cfg=manifest.get('resolved_cli_config',{})
 if isinstance(cfg,str):
  import yaml;cfg=yaml.safe_load(cfg)
 if 'fit' in cfg:cfg=cfg['fit']
 model=cfg.get('model',{});args=model.get('init_args',{});network=args.get('network',{})
 netargs=network.get('init_args',{});seed=cfg.get('seed_everything',startup.get('seed',0))
 protocols=[(q,read(q)) for q in run.rglob('protocol_manifest.json') if not any(x in q.parts for x in ['cache','provenance','wandb'])]
 valid=[(q,d) for q,d in protocols if d and d.get('protocol')=='ce2d_stableseg_execution_v2']
 repair=bool(valid)
 argv=manifest.get('argv',[])
 commands=list(argv) if isinstance(argv,list) else [str(argv)]
 for q,d in docs:
  for key in ['command','cmd','training_command']:
   v=d.get(key)
   if v:commands.extend(v if isinstance(v,list) else [str(v)])
  if isinstance(d.get('child_identity'),dict):commands.append(d['child_identity'].get('command',''))
 commandtext=' '.join(str(v) for v in commands)
 explicit_false=s.get('deterministic_repair_enabled') is False or startup.get('deterministic_repair_enabled') is False
 if not repair:assert explicit_false or any(x in commandtext for x in ['main.py','run_synapse_roi.py','baseline-lock/preparation/run.py']),runrel
 if repair:assert all(d['backend']['deterministic'] is True and d['backend']['warn_only'] is False for q,d in valid)
 roi=repair or startup.get('roi_enabled') is True or s.get('roi_enabled') is True or 'run_synapse_roi.py' in commandtext
 protocol=valid[0][1]['protocol'] if repair else 'original_cuda'
 bestrow=None
 for _ in range(6):
  before=[stamp(sel/n) for n in ['selection_summary.json','test_epoch_metrics.csv','test_case_metrics.csv']]
  s=read(p);er=rows(sel/'test_epoch_metrics.csv');cr=rows(sel/'test_case_metrics.csv')
  if before!=[stamp(sel/n) for n in ['selection_summary.json','test_epoch_metrics.csv','test_case_metrics.csv']]:time.sleep(.3);continue
  rr=[x for x in er if int(x['epoch'])==s['best_epoch']]
  assert len(rr)==1;bestrow=rr[0];break
 assert bestrow is not None
 assert abs(float(bestrow['mean_dice'])-s['best_mean_dice'])<1e-12
 first=min(int(x['epoch']) for x in er);last=max(int(x['epoch']) for x in er)
 assert last==s['last_evaluated_epoch'] and {int(x['epoch']) for x in er}==set(range(first,last+1))
 assert len(cr)==len(er)*12
 br=[x for x in cr if int(x['epoch'])==s['best_epoch']]
 assert len(br)==12 and {x['case'] for x in br}==CASES
 phys=sel/'physical_case_organ_metrics.csv';pr=rows(phys) if phys.exists() else []
 if pr:
  pr=[x for x in pr if int(x['epoch'])==s['best_epoch']]
  assert len(pr)==96 and {(x['case'],int(x['label_id'])) for x in pr}=={(c,k) for c in CASES for k in range(1,9)}
 distdisabled=s.get('hd95_mm_enabled') is False and s.get('hd95_voxel_enabled') is False
 organs=[]
 for label in LABELS:
  k=label['label_id'];dice=float(bestrow['dice_class_'+str(k)])
  assert abs(dice-fmean(float(x['dice_class_'+str(k)]) for x in br))<1e-12
  mm=vx=None
  if pr:
   a=[x for x in pr if int(x['label_id'])==k]
   assert abs(dice-fmean(float(x['dice']) for x in a))<1e-12
   mm=avg(num(x['hd95_mm']) for x in a);vx=avg(num(x['hd95_voxel']) for x in a)
  elif not distdisabled:
   vx=num(bestrow.get('hd95_class_'+str(k)))
  if distdisabled:assert mm is None and vx is None
  organs.append(dict(label_id=k,organ=label['name_zh'],dice=dice,hd95_mm=mm,hd95_voxel=vx))
 checkpoint=load_stable(sel/'best_test.ckpt');assert int(checkpoint['epoch'])+1==s['best_epoch']
 lastckpt=load_stable(sel/'last.ckpt');completed=int(lastckpt['epoch'])+1;step=int(lastckpt['global_step'])
 statelabel=state.get('state','unknown');code=exitdoc.get('exit_code',state.get('exit_code',state.get('training_exit_code')))
 if statelabel=='completed':
  assert code==0 and completed==s['max_epochs']
  assert step==s['max_epochs']*138
 if statelabel=='running':
  launch=next((d for q,d in docs if q.name=='launch.json'),{})
  live=identity(launch['pid']);assert live and same_process(launch['training_identity'])
 else:live=None
 lr=startup.get('base_lr',args.get('lr'))
 warm=bool(s.get('weights_only_warm_start'))
 sha_best=(read(run/'summary.json') or {}).get('best_checkpoint_sha256')
 x=dict(id='E'+str(len(catalog)+1).zfill(2),name=aliases[runrel],run=runrel,state=statelabel,exit_code=code,seed=seed,lr=lr,max_epochs=s['max_epochs'],completed_epochs=completed,global_step=step,best_epoch=s['best_epoch'],last_evaluated_epoch=last,evaluation_window=str(first)+'-'+str(last),evaluated_epochs=len(er),case_rows=len(cr),mean_dice=s['best_mean_dice'],mean_hd95_mm=avg(y['hd95_mm'] for y in organs),mean_hd95_voxel=avg(y['hd95_voxel'] for y in organs),cuda_repair_enabled=repair,cuda_protocol=protocol,roi_enabled=roi,network_class=network.get('class_path',startup.get('network')),model_class=model.get('class_path'),use_text=netargs.get('use_text'),use_probability=netargs.get('use_probability'),enhance_patch=netargs.get('enhance_patch'),enhance_query=netargs.get('enhance_query'),ccps_config={k:v for k,v in netargs.items() if k.startswith('selector_') or k in ['patch_gate_effect','detach_prior_for_selector']},warm_start=warm,source_checkpoint=s.get('source_checkpoint'),optimizer_and_scheduler_reset=s.get('optimizer_and_scheduler_reset'),tta_views=1,checkpoint=str(sel/'best_test.ckpt'),checkpoint_sha256=sha_best,organs=organs,selection=str(p),training_identity=live)
 catalog.append(x)
 for y in organs:orgtable.append(dict(experiment_id=x['id'],experiment=x['name'],run=runrel,best_epoch=x['best_epoch'],cuda_repair_enabled=repair,**y,dice_percent=y['dice']*100))
 sources=[dict(path=rep(q),sha256=sha(q)) for q,d in docs]+[dict(path=rep(mp),sha256=sha(mp))] if mp.exists() else []
 sources+=[dict(path=rep(q),protocol=d['protocol'],profile=d.get('execution_profile'),backend=d.get('backend')) for q,d in valid]
 evidence.append(dict(experiment_id=x['id'],run=runrel,selection_snapshot=s,source_config=dict(seed_everything=seed,model_class=model.get('class_path'),network=network,lr=lr,data=cfg.get('data'),precision=cfg.get('trainer',{}).get('precision'),trainer_deterministic=cfg.get('trainer',{}).get('deterministic')),cuda_repair_evidence=sources,argv=argv))
 del checkpoint,lastckpt
# Only successful standalone physical summaries. Parent summary wrappers are duplicates of these leaves.
tests=[];testorgs=[]
for p in sorted(OUT.rglob('physical_test_summary.json')):
 if DEST in p.parents or any(t in p.parts for t in ['audit','smoke','optimization','preparation']):continue
 d=read(p)
 if not d or d.get('num_cases')!=12:continue
 folder=p.parent;ep=folder/'physical_epoch_organ_metrics.csv';case=folder/'physical_case_organ_metrics.csv'
 assert ep.exists() and case.exists()
 ar=rows(ep);cases=rows(case);assert len(cases)==96 and {(x['case'],int(x['label_id'])) for x in cases}=={(c,k) for c in CASES for k in range(1,9)}
 contract=read(folder/'tta_contract.json') or {};views=d.get('tta_views',contract.get('tta_views',1))
 scheme=d.get('tta_scheme',contract.get('tta_scheme'))
 if views==1:scheme='1-view'
 elif not scheme:scheme='legacy_hflip_2' if views==2 else 'retired_flip_4_with_vertical'
 ckpt=Path(d['ckpt_path'])
 source=next((x for x in catalog if Path(x['checkpoint'])==ckpt),None);assert source,str(ckpt)
 result=[]
 for label in LABELS:
  k=label['label_id'];rr=[x for x in cases if int(x['label_id'])==k];assert len(rr)==12
  result.append(dict(label_id=k,organ=label['name_zh'],dice=fmean(float(x['dice']) for x in rr),hd95_mm=avg(num(x['hd95_mm']) for x in rr),hd95_voxel=avg(num(x['hd95_voxel']) for x in rr)))
 assert abs(fmean(x['dice'] for x in result)-d['mean_dice'])<1e-12
 assert abs(fmean(x['hd95_mm'] for x in result)-d['mean_hd95_mm'])<1e-10
 m=read(folder/'run_manifest.json') or {}
 t=dict(id='T'+str(len(tests)+1).zfill(2),summary=rep(p),evaluation_folder=rep(folder),source_training_id=source['id'],source_training=source['name'],source_checkpoint=d['ckpt_path'],source_checkpoint_epoch=d['checkpoint_epoch'],checkpoint_sha256=d.get('checkpoint_sha256'),source_training_cuda_repair_enabled=source['cuda_repair_enabled'],source_training_protocol=source['cuda_protocol'],tta_views=views,tta_scheme=scheme,scales=d.get('scales',contract.get('scales')),mean_dice=d['mean_dice'],mean_hd95_mm=d['mean_hd95_mm'],mean_hd95_voxel=d.get('mean_hd95_voxel'),organs=result,training_performed=False,teacher_adapter='teacher_hd95' in p.parts,inference_argv=m.get('argv'),duplicate_of=None)
 for other in tests:
  if t['checkpoint_sha256']==other['checkpoint_sha256'] and t['tta_views']==other['tta_views'] and t['tta_scheme']==other['tta_scheme'] and t['teacher_adapter']==other['teacher_adapter'] and abs(t['mean_dice']-other['mean_dice'])<1e-12:t['duplicate_of']=other['id'];break
 tests.append(t)
 for y in result:testorgs.append(dict(test_id=t['id'],source_training=t['source_training'],tta_views=views,tta_scheme=scheme,**y,dice_percent=y['dice']*100))
# Historical seed1/2 formal-budget plans: one started then stopped before first test, three cancelled.
unscored=[]
seedroot=OUT/'innovation1/asee_seed_audit/20261023T023553Z'
if seedroot.exists():
 for p in sorted(seedroot.iterdir()):
  if not p.is_dir() or not (p/'status.json').exists():continue
  st=read(p/'status.json');la=read(p/'launch.json') or {}
  assert not list(p.glob('synapse_test_selection/*/selection_summary.json'))
  began=st.get('training_pid') is not None or st.get('state')=='stopped_by_user'
  saved=list(p.glob('synapse_test_selection/*/last.ckpt'));epoch=step=None
  if saved:
   cp=load_stable(saved[0]);epoch=int(cp['epoch'])+1;step=int(cp['global_step'])
  command=la.get('command',[]);assert 'main.py' in command
  unscored.append(dict(name=p.name,method=st.get('method'),seed=st.get('seed'),state=st['state'],started=began,max_epochs=la.get('max_epochs'),completed_epochs=epoch,global_step=step,mean_dice=None,best_epoch=None,cuda_repair_enabled=False,cuda_evidence_scope='actual original main.py command' if began else 'planned original main.py command; not started',run=rep(p),reason='用户停止/取消，未形成完整测试结果'))

# Retain identities and source epochs; never infer missing metrics from current run or mix checkpoints.
checked=datetime.now(timezone.utc).isoformat()
data=dict(checked_utc=checked,formal_training_count=len(catalog),status_counts=dict(Counter(x['state'] for x in catalog)),unscored_training_records=unscored,standalone_test_count=len(tests),unique_standalone_test_count=sum(x['duplicate_of'] is None for x in tests),training=catalog,tests=tests,excluded_selection_summaries=excluded,label_map_path='priors/synapse_label_map.verified.json',metric_policy='legacy Dice; no background; each column uses one best checkpoint; HD95_mm and voxel separate',cuda_flag_definition='cuda_repair_enabled=true means this project V2 stable CE/point sampling and strict guard were enabled; does not assert every full CUDA training is bitwise reproducible.')
save(DEST/'experiments.json',data);save(DEST/'evidence.json',evidence)
writecsv(DEST/'unscored_seed_records.csv',unscored,['name','method','seed','state','started','max_epochs','completed_epochs','global_step','mean_dice','cuda_repair_enabled','cuda_evidence_scope','run','reason'])
fields=['id','name','run','state','exit_code','seed','lr','max_epochs','completed_epochs','global_step','evaluation_window','best_epoch','mean_dice','mean_hd95_mm','mean_hd95_voxel','cuda_repair_enabled','cuda_protocol','roi_enabled','warm_start','source_checkpoint']
writecsv(DEST/'training_summary.csv',catalog,fields)
writecsv(DEST/'training_organ_metrics.csv',orgtable,['experiment_id','experiment','run','best_epoch','cuda_repair_enabled','label_id','organ','dice','dice_percent','hd95_mm','hd95_voxel'])
writecsv(DEST/'independent_test_summary.csv',tests,['id','source_training_id','source_training','source_training_cuda_repair_enabled','source_training_protocol','tta_views','tta_scheme','mean_dice','mean_hd95_mm','mean_hd95_voxel','source_checkpoint_epoch','source_checkpoint','duplicate_of','summary'])
writecsv(DEST/'independent_test_organs.csv',testorgs,['test_id','source_training','tta_views','tta_scheme','label_id','organ','dice','dice_percent','hd95_mm','hd95_voxel'])
md='# outputs主实验、重复训练与消融全量汇总\n\n核验UTC：'+checked+'。本报告只读原始产物，不停止/重启训练，不修改模型、训练配置、原CSV或checkpoint。\n\n'
md+='## CUDA标记与统计口径\n\n“CUDA修复=开”表示实际启用ce2d_stableseg_execution_v2：独立稳定CE/点采样、严格deterministic guard；CCPS另有局部BN边界/AOT backward off。不是“开启CUDA非确定性”。“关”表示原CUDA入口，未接入这套修复，不能据此断言每个算子都非确定。开也不能保证两次完整训练最终逐位相同。ROI只属指标基础设施，不等于CUDA修复。\n\n所有训练主表为各自最佳checkpoint的1-view legacy Dice；8-view不混入主表。单位Dice原值0..1、表内百分数；HD95_mm/HD95_voxel分列，未计算为“—”，不是0。背景0不整理；器官按verified label_id映射，每组8器官来自同一best，绝不逐器官跨epoch挑最大。\n\n'
md+='发现'+str(len(catalog))+'组已形成评分的正式训练，状态计数'+str(data['status_counts'])+'；另有'+str(len(tests))+'份完整12例独立推理结果（含明确标注的重复复测）。冒烟/审计/预备目录不作为论文主实验，未执行计划不伪造结果。\n\n'
md+='## 全部正式训练（主实验、组件消融、seed与收敛性检查）\n\n|ID|实验|CUDA修复|ROI|seed|训练轮数/预算|状态|best epoch|1-view Dice%|HD95_mm|HD95_voxel|\n|---|---|---|---|---:|---|---|---:|---:|---:|---:|\n'
for x in catalog:
 md+='|'+('|'.join([x['id'],x['name'],'开(V2)' if x['cuda_repair_enabled'] else '关','开' if x['roi_enabled'] else '关',str(x['seed']),str(x['completed_epochs'])+'/'+str(x['max_epochs']),x['state'],str(x['best_epoch']),fmt(x['mean_dice'],True),fmt(x['mean_hd95_mm']),fmt(x['mean_hd95_voxel'])]))+'|\n'
md+='\n低LR组是历史CCPS best187的weights-only补训，LR2e-5、额外100轮，优化器/调度器重建且使用V2；不冒充从原预训练的220轮对照。原CUDA CCPS seed0/42及旧S2均用户早停，候选窗口不等于完整50轮。运行中的完整主模型为本报告快照，后续可能改变。\n\n'
md+='## 正式训练逐器官Dice（%）\n\n|ID|实验|'+('|'.join(x['name_zh'] for x in LABELS))+'|平均|\n|---|---|'+('|'.join(['---:']*9))+'|\n'
for x in catalog:md+='|'+x['id']+'|'+x['name']+'|'+('|'.join(fmt(y['dice'],True) for y in x['organs']))+'|'+fmt(x['mean_dice'],True)+'|\n'
md+='\n## 独立推理：TTA、物理HD95及重复复测\n\n表中CUDA标记为“源权重训练时”的修复开关；这些任务没有反向训练，不能用推理入口是否接入确定性loss来判断源训练协议。\n\n|ID|源训练|源训练CUDA修复|视角/协议|Dice%|HD95_mm|HD95_voxel|重复说明|\n|---|---|---|---|---:|---:|---:|---|\n'
for t in tests:
 md+='|'+('|'.join([t['id'],t['source_training'],'开(V2)' if t['source_training_cuda_repair_enabled'] else '关',str(t['tta_views'])+' / '+t['tta_scheme']+(' / 教师适配' if t['teacher_adapter'] else ''),fmt(t['mean_dice'],True),fmt(t['mean_hd95_mm']),fmt(t['mean_hd95_voxel']),'复测同'+t['duplicate_of'] if t['duplicate_of'] else '独立测试']))+'|\n'
md+='\n当前8-view均为四尺度1/.875/1.125/.75×原方向/水平翻转；历史4-view含垂直翻转的结果已明确标为retired_flip_4_with_vertical，不当成当前多尺度4-view，也不恢复退役代码。M0与指定baseline best复测是同一权重/协议/结果，不能当两组独立训练。教师HD95与物理HD95同权重同指标但接口不同，保留各自来源。\n\n'
md+='## 独立推理逐器官Dice（%）\n\n|ID|源训练/视角|'+('|'.join(x['name_zh'] for x in LABELS))+'|平均|\n|---|---|'+('|'.join(['---:']*9))+'|\n'
for t in tests:md+='|'+t['id']+'|'+t['source_training']+'/'+str(t['tta_views'])+'|'+('|'.join(fmt(y['dice'],True) for y in t['organs']))+'|'+fmt(t['mean_dice'],True)+'|\n'
md+='\n## 逐器官HD95与完整来源\n\ntraining_organ_metrics.csv、independent_test_organs.csv逐行保存器官名/label_id/Dice原值与百分数/HD95_mm/HD95_voxel；experiments.json保存每组所有8器官和真实路径，evidence.json记录配置/入口/确定性guard的证据与哈希。\n\n'
for x in catalog:md+='- '+x['id']+' '+x['name']+'：'+mdlink(ROOT/x['run'])+'；选择窗口'+x['evaluation_window']+'；来源'+mdlink(Path(x['selection']))+'。\n'
for t in tests:md+='- '+t['id']+'：'+mdlink(ROOT/t['summary'])+'，源训练'+t['source_training_id']+'，best'+str(t['source_checkpoint_epoch'])+'。\n'
md+='\n## 未形成测试成绩的历史seed1/2条目\n\n|条目|seed|状态|实际启动|已保存轮数|CUDA修复|平均Dice|\n|---|---:|---|---|---|---|---|\n'
for u in unscored:md+='|'+u['name']+'|'+str(u['seed'])+'|'+u['state']+'|'+('是' if u['started'] else '否')+'|'+(str(u['completed_epochs']) if u['completed_epochs'] is not None else '—')+'|'+('关（原入口）' if u['started'] else '计划为关，未启动')+'|未形成正式评估|\n'
md+='\n这4项不混入16组已有评分的总表，其中1项已启动后早停且未到评估阶段，3项在排队阶段取消。状态和路径另见unscored_seed_records.csv。\n'
md+='\n## 边界\n\nS3/S4、A1/A2/A10/A11没有发现正式selection结果，不因为存在计划或配置就记为完成；ASEE修复第二次未发现正式run。已按用户删除的旧非ROI baseline-2和旋转增强输出不在当前outputs中，不从文档估造最终成绩。所有旧run文档标题只作提示，实际模型开关以归档配置/协议/状态核验为准。完整主模型相对单先验同时增加另一先验和CCPS，三组不能单独隔离CCPS贡献。不同训练协议、seed、预算、TTA和距离单位不可直接混成一个排行榜。\n'
(DEST/'summary.md').write_text(md,encoding='utf-8')
save(DEST/'file_changes.json',dict(created=[str(DEST/x) for x in ['collect.py','experiments.json','evidence.json','training_summary.csv','training_organ_metrics.csv','independent_test_summary.csv','independent_test_organs.csv','summary.md','unscored_seed_records.csv','file_changes.json','collect.log','validation.json']],modified_existing=[],deleted=[],production_or_training_changed=False))
print(json.dumps(dict(report=str(DEST/'summary.md'),formal_training_count=len(catalog),status_counts=data['status_counts'],test_count=len(tests),training=[{k:x[k] for k in ['id','name','state','completed_epochs','best_epoch','mean_dice','cuda_repair_enabled','roi_enabled']} for x in catalog],tests=[{k:t[k] for k in ['id','source_training','tta_views','tta_scheme','mean_dice','mean_hd95_mm','duplicate_of']} for t in tests]),ensure_ascii=False),flush=True)
