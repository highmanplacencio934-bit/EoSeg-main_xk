"""Output-path adapter only: reuse all validated G6 training and comparison functions."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys
from datetime import datetime,timezone
ROOT=Path('/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse')
sys.path.insert(0,str(ROOT))
from scripts import launch_synapse_g6 as g
OLD=ROOT/'outputs/baseline-repair'
BASE=ROOT/'outputs/baseline-repair_2'
g.BASE=BASE

def verify():
 assert g.sha(__file__)==g.read(BASE/'preparation/adapter_validation.json')['adapter_sha256']
 assert g.read(ROOT/'outputs/innovation2/ccps_low_lr_2e-5_100ep/20261028T124208Z/status.json')['state']=='completed'
 assert g.read(ROOT/'outputs/innovation2/ccps_low_lr_2e-5_100ep/20261028T124208Z/training_exit.json')['exit_code']==0
 assert g.read(BASE/'repeat_1.json')['run']==g.read(OLD/'repeat_1.json')['run']
 return g.preflight(2)

def main():
 a=argparse.ArgumentParser();a.add_argument('--supervise');a.add_argument('--check',action='store_true');args=a.parse_args()
 if args.supervise:
  with (OLD/'queue.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   verify()
   g.supervise(2,args.supervise)
  return
 with (OLD/'queue.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:print(json.dumps({'deferred':'Previous run holds shared queue lock'}));return
  with (BASE/'launch.lock').open('a+') as guard:
   fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
   for path in [OLD/'repeat_2.json',BASE/'repeat_2.json']:
    if path.exists():
     receipt=g.read(path);run=Path(receipt['run']);print(json.dumps({'existing':True,'receipt':receipt,'status':g.read(run/'status.json'),'no_duplicate_started':True}));return
   evidence=verify()
   if args.check:print(json.dumps({'passed':True,'preflight':evidence,'training_started':False}));return
   run=BASE/'G6_2'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');run.mkdir(parents=True,exist_ok=False)
   g.write(run/'status.json',{'state':'launching','repeat':2})
   with (run/'supervisor.log').open('w') as log:
    child=subprocess.Popen([g.PY,str(Path(__file__).resolve()),'--supervise',str(run)],cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
   receipt={'run':str(run),'repeat':2,'supervisor_pid':child.pid,'created_utc':datetime.now(timezone.utc).isoformat(),'preflight':evidence,'output_path_adapter':str(Path(__file__).resolve())}
   g.write(BASE/'repeat_2.json',receipt);g.write(OLD/'repeat_2.json',receipt)
   (BASE/'latest_path.txt').write_text(str(run)+'\n')
   print(json.dumps(receipt))
if __name__=='__main__':main()
