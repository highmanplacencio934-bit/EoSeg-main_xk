"""Independent launcher isolation, readiness and result integrity checks."""
from pathlib import Path
import sys,json,tempfile,unittest,hashlib,os,csv
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import launch_synapse_joint_augmentation as launch

class JointLauncherTests(unittest.TestCase):
    def test_idempotent_launch_preserves_original_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'new'
            proc=Mock(pid=123)
            with patch.object(launch,'BASE',base),patch.object(launch,'ROOT',Path(tmp)),patch.object(launch,'preflight',return_value={}),patch.object(launch.subprocess,'Popen',return_value=proc) as popen:
                launch.launch()
                receipt=(base/'launch_receipt.json').read_bytes()
                launch.launch()
                self.assertEqual((base/'launch_receipt.json').read_bytes(),receipt)
                popen.assert_called_once()

    def test_changed_new_source_rejected_without_touching_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.py';source.write_text('original')
            reference=root/'reference.json';reference.write_text('{}')
            ready=root/'ready.json';ready.write_text(json.dumps({'state':'passed','config':launch.CONFIG,
                'cpu':{'passed':True},'gpu':{'passed':True},'sha256':{'source.py':hashlib.sha256(source.read_bytes()).hexdigest()},
                'reference_result_sha256':hashlib.sha256(reference.read_bytes()).hexdigest()}))
            cwd=Path.cwd()
            try:
                with patch.object(launch,'ROOT',root),patch.object(launch,'READY',ready),patch.object(launch,'REFERENCE',reference),patch.object(launch,'original_preflight',return_value={}),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                    self.assertIn('joint_readiness_sha256',launch.preflight())
                    source.write_text('changed')
                    with self.assertRaisesRegex(RuntimeError,'changed'):launch.preflight()
            finally:os.chdir(cwd)

    def test_empty_gpu_visibility_rejected(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaisesRegex(RuntimeError,'hidden'):launch.preflight()

    def test_fit_command_has_original_budget_and_no_checkpoint_resume(self):
        cmd=launch.fit_command(Path('/example/new'))
        self.assertIn('--seed_everything=0',cmd)
        self.assertIn(launch.CONFIG,cmd)
        self.assertFalse(any(x.startswith('--ckpt_path') or x.startswith('--trainer.max_epochs') or x=='--compile_disabled' for x in cmd))
        callbacks=json.loads(next(x.split('=',1)[1] for x in cmd if x.startswith('--trainer.callbacks=')))
        self.assertEqual(callbacks[0]['init_args']['last_n_epochs'],50)
        self.assertEqual(callbacks[0]['init_args']['matched_window_end'],213)

if __name__=='__main__':unittest.main()
