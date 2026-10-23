"""Detached supervisor: records status; never changes training hyperparameters."""
import datetime,json,os,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parent
meta=json.loads((root/"launch.json").read_text())
env=os.environ.copy();env.update(meta["environment"])
start=datetime.datetime.now(datetime.timezone.utc)
with (root/"train.log").open("a",buffering=1) as log:
    child=subprocess.Popen(meta["command"],cwd=meta["project_root"],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    status=dict(state="running",supervisor_pid=os.getpid(),training_pid=child.pid,started_at_utc=start.isoformat(),output_root=str(root))
    (root/"status.json").write_text(json.dumps(status,indent=2))
    code=child.wait()
    status.update(state="completed" if code==0 else "failed",exit_code=code,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (root/"status.json").write_text(json.dumps(status,indent=2))
