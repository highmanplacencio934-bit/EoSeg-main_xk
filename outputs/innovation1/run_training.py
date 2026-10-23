"""Detached supervisor for one reviewed ASEE run."""
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
