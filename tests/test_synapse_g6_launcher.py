import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts import launch_synapse_g6 as g
class G6LauncherTests(unittest.TestCase):
 def test_lightning_unwrapped_compile_is_detected(self):
  from types import SimpleNamespace
  self.assertTrue(g.compiled_model_active(SimpleNamespace(_compiler_ctx={"compiler":"dynamo"})))
  self.assertTrue(g.compiled_model_active(SimpleNamespace(_orig_mod=object())))
  self.assertFalse(g.compiled_model_active(SimpleNamespace()))
  self.assertFalse(g.compiled_model_active(SimpleNamespace(_compiler_ctx=None)))
 def invoke(self,*args):
  with patch("sys.argv",["launch_synapse_g6.py",*args]),contextlib.redirect_stdout(io.StringIO()):
   g.main()
 def test_existing_failed_run_not_restarted(self):
  with tempfile.TemporaryDirectory() as d,patch.object(g,"BASE",Path(d)),patch.object(g.subprocess,"Popen") as spawn:
   run=Path(d)/"run";run.mkdir();g.write(run/"status.json",{"state":"failed"});g.write(Path(d)/"repeat_1.json",{"run":str(run)})
   self.invoke("--repeat","1");spawn.assert_not_called()
 def test_check_without_run_does_not_launch(self):
  with tempfile.TemporaryDirectory() as d,patch.object(g,"BASE",Path(d)),patch.object(g.subprocess,"Popen") as spawn:
   self.invoke("--repeat","1","--check");spawn.assert_not_called()
 def test_failed_preflight_creates_only_deferral(self):
  with tempfile.TemporaryDirectory() as d,patch.object(g,"BASE",Path(d)),patch.object(g,"preflight",side_effect=RuntimeError("readiness pending")),patch.object(g.subprocess,"Popen") as spawn:
   self.invoke("--repeat","1");spawn.assert_not_called()
   self.assertTrue((Path(d)/"deferred_1.json").exists());self.assertFalse((Path(d)/"repeat_1.json").exists())
 def test_queue_lock_blocks_duplicate(self):
  import fcntl
  with tempfile.TemporaryDirectory() as d,patch.object(g,"BASE",Path(d)),patch.object(g.subprocess,"Popen") as spawn:
   lock=(Path(d)/"queue.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   self.invoke("--repeat","1");spawn.assert_not_called();lock.close()
 def test_second_run_waits_for_first(self):
  with tempfile.TemporaryDirectory() as d,patch.object(g,"BASE",Path(d)),patch("training.deterministic_execution_v2.require_readiness"),patch.object(g.subprocess,"check_output") as query:
   g.write(Path(d)/"preparation.json",{"passed":True,"launcher_sha256":g.sha(g.__file__)})
   with self.assertRaisesRegex(RuntimeError,"First G6 not complete"):g.preflight(2)
   query.assert_not_called()
 def test_stale_pid_or_changed_command_rejected(self):
  record={"pid":1,"start_ticks":"1","cwd":str(g.ROOT),"cmdline":"expected"}
  for changed in (dict(record,start_ticks="2",state="R"),dict(record,cmdline="different",state="R"),dict(record,state="Z")):
   with patch.object(g,"identity",return_value=changed):self.assertFalse(g.same_process(record))
 def test_incomplete_training_cannot_complete(self):
  with tempfile.TemporaryDirectory() as d:
   run=Path(d);folder=run/"synapse_test_selection"/"one";folder.mkdir(parents=True)
   g.write(folder/"selection_summary.json",{"max_epochs":220,"last_evaluated_epoch":219})
   with self.assertRaises(AssertionError):g.coverage(run)
if __name__=="__main__":unittest.main()
