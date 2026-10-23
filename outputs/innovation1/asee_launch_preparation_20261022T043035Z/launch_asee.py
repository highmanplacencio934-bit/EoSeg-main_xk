"""Launch isolated ASEE smoke/formal jobs; algorithm settings remain in YAML."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
SUPERVISOR = r'''"""Detached supervisor for one reviewed ASEE run."""
import datetime,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
meta=json.loads((root/'launch.json').read_text())
env=os.environ.copy();env.update(meta['environment'])
status={'state':'launching','supervisor_pid':os.getpid(),'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'output_root':str(root)}
def write_status():
    temp=root/'status.json.tmp'
    temp.write_text(json.dumps(status,indent=2))
    temp.replace(root/'status.json')
try:
    with (root/'train.log').open('a',buffering=1) as log:
        child=subprocess.Popen(meta['command'],cwd=meta['project_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status.update(state='running',training_pid=child.pid);write_status()
        code=child.wait()
    status.update(state='completed' if code==0 else 'failed',exit_code=code,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write_status()
except BaseException as error:
    status.update(state='supervisor_failed',error=repr(error),finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write_status();raise
'''
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','formal']);args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:raise RuntimeError('Start from verified project root')
    prior=ROOT/'priors/synapse_asee_v1.pt'
    if not prior.is_file():raise FileNotFoundError(prior)
    if args.mode=='formal':
        verification=json.loads(Path(__file__).with_name('smoke_checkpoint_verification.json').read_text())
        if verification.get('passed') is not True or verification.get('prior_sha256')!=hashlib.sha256(prior.read_bytes()).hexdigest():raise RuntimeError('Real smoke/checkpoint verification missing or prior changed')
    baseline=json.loads((ROOT/'outputs/baseline/20261021T132635Z/status.json').read_text())
    if baseline.get('state')!='completed' or baseline.get('exit_code')!=0:raise RuntimeError('Baseline not confirmed complete')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    free_mib=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free_mib<24576:raise RuntimeError('Insufficient free GPU headroom: '+str(free_mib)+' MiB; other processes: '+processes)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'outputs/innovation1'/('asee_plan_smoke' if args.mode=='smoke' else 'asee_plan')/stamp;out.mkdir(parents=True,exist_ok=False)
    callbacks=[{'class_path':'training.synapse_test_selection.SynapseTestSelection','init_args':{'last_n_epochs':1 if args.mode=='smoke' else 50,'expected_test_cases':12,'output_dir':str(out)}}]
    if args.mode=='smoke':callbacks.append({'class_path':'outputs.innovation1.asee_launch_preparation_20261022T043035Z.asee_runtime_probe.ASEERuntimeProbe','init_args':{'output_dir':str(out/'runtime_audit')}})
    cmd=[sys.executable,'-u','main.py','fit','--config','configs/synapse/vit_query_mul_scale_fusion_asee_plan.yaml','--trainer.default_root_dir='+str(out),'--trainer.logger.init_args.save_dir='+str(out),'--trainer.callbacks='+json.dumps(callbacks)]
    if args.mode=='smoke':cmd+=['--trainer.max_epochs=1','--trainer.limit_train_batches=8']
    environment={'HF_HUB_OFFLINE':'1','TORCH_HOME':'/Disk/HDD/Sk/.cache/torch','TIMM_USE_OLD_CACHE':'1','WANDB_MODE':'offline','PYTHONUNBUFFERED':'1'}
    sources=[ROOT/'main.py',ROOT/'priors/synapse_label_map.verified.json',ROOT/'scripts/build_synapse_asee_prior.py',Path(__file__)]
    for folder in ['models','datasets','training','configs/synapse']:sources += [f for f in (ROOT/folder).rglob('*') if f.suffix in {'.py','.yaml'}]
    meta={'mode':args.mode,'project_root':str(ROOT),'output_root':str(out),'command':cmd,'environment':environment,'scientific_config':'configs/synapse/vit_query_mul_scale_fusion_asee_plan.yaml','max_epochs':1 if args.mode=='smoke' else 220,'batch_size':16,'precision':'16-mixed','compile_disabled':False,'prior_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},'dice_policy':'legacy pred_nonempty/gt_empty=1; both_empty=0','hd95_unit':'voxel','initialization':'Original timm pretrained offline cache; not final baseline checkpoint','existing_gpu_processes':processes,'minimum_free_gpu_mib':24576,'gpu_query':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free','--format=csv'],text=True)}
    (out/'organ_label_map.json').write_bytes((ROOT/'priors/synapse_label_map.verified.json').read_bytes())
    (out/'launch.json').write_text(json.dumps(meta,indent=2))
    script=out/'run_training.py';script.write_text(SUPERVISOR)
    env=os.environ.copy();env.update(environment)
    with (out/'supervisor.log').open('ab') as log:
        process=subprocess.Popen([sys.executable,'-u',str(script)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    Path(__file__).with_name(args.mode+'_run_path.txt').write_text(str(out.relative_to(ROOT)))
    print(json.dumps({'run':str(out),'supervisor_pid':process.pid,'mode':args.mode}))
if __name__=='__main__':main()
