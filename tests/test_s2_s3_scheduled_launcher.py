"""CPU-only scheduling safety tests; never launch training or query a real GPU."""
from pathlib import Path
from unittest import mock
import contextlib,csv,hashlib,io,json,tempfile,unittest
from scripts import launch_synapse_s2_s3_scheduled as launch


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))


def csvfile(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def completed_fixture(root):
    run=root/'baseline';folder=run/'synapse_test_selection/run';folder.mkdir(parents=True)
    cases=[f'fixture{i:02d}' for i in range(12)]
    lists=root/'project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt';lists.parent.mkdir(parents=True);lists.write_text('\n'.join(cases)+'\n')
    write(run/'status.json',{'state':'completed','roi_enabled':True})
    write(run/'training_exit.json',{'exit_code':0})
    best=folder/'best_test.ckpt';best.write_bytes(b'fixture only')
    write(run/'summary.json',{'max_epochs':220,'test_epoch_records':50,'test_case_records':600,'case_organ_records':4800,'checkpoint':str(best),'checkpoint_sha256':hashlib.sha256(best.read_bytes()).hexdigest()})
    write(folder/'selection_summary.json',{'max_epochs':220,'last_evaluated_epoch':220})
    csvfile(folder/'test_epoch_metrics.csv',[{'epoch':e} for e in range(171,221)])
    csvfile(folder/'test_case_metrics.csv',[{'epoch':e,'case':c} for e in range(171,221) for c in cases])
    csvfile(folder/'physical_case_organ_metrics.csv',[{'epoch':e,'case':c,'label_id':i} for e in range(171,221) for c in cases for i in range(1,9)])
    return run,folder


class ScheduledLauncherTests(unittest.TestCase):
    def test_other_users_are_not_a_rejection_if_memory_is_enough(self):
        with mock.patch.object(launch.subprocess,'check_output',return_value='26000\n') as query:
            self.assertTrue(launch.gpu_free(launch.TRAIN_FREE_MIB))
            self.assertIn('--query-gpu=memory.free',query.call_args.args[0])
            self.assertNotIn('--query-compute-apps=pid',query.call_args.args[0])

    def test_insufficient_memory_rejects_without_process_control(self):
        with mock.patch.object(launch.subprocess,'check_output',return_value='5000\n'),mock.patch.object(launch.os,'kill') as kill:
            self.assertFalse(launch.gpu_free(launch.TEST_FREE_MIB));kill.assert_not_called()

    def test_train_and_test_use_separate_memory_budgets(self):
        with mock.patch.object(launch.subprocess,'check_output',return_value='10000\n'):
            self.assertTrue(launch.gpu_free());self.assertFalse(launch.gpu_free(launch.TRAIN_FREE_MIB))

    def test_running_baseline_cannot_be_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write(root/'status.json',{'state':'training','roi_enabled':True})
            with self.assertRaisesRegex(RuntimeError,'not fully completed'):launch.require_baseline_2_complete(root)

    def test_baseline_failed_exit_cannot_unlock_successor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write(root/'status.json',{'state':'completed','roi_enabled':True});write(root/'training_exit.json',{'exit_code':1})
            with self.assertRaisesRegex(RuntimeError,'did not exit0'):launch.require_baseline_2_complete(root)

    def test_complete_baseline_requires_all_cases_and_training_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,folder=completed_fixture(root)
            with mock.patch.object(launch,'ROOT',root),mock.patch('torch.load',return_value={'epoch':219,'global_step':30360}):
                proof=launch.require_baseline_2_complete(run);self.assertEqual(proof['max_epochs'],220)

    def test_missing_case_rejected_even_with_completed_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,folder=completed_fixture(root)
            p=folder/'test_case_metrics.csv';rows=read_csv(p);csvfile(p,rows[:-1])
            with mock.patch.object(launch,'ROOT',root),self.assertRaisesRegex(RuntimeError,'coverage incomplete'):launch.require_baseline_2_complete(run)

    def test_duplicate_case_organ_cannot_fake_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,folder=completed_fixture(root)
            p=folder/'physical_case_organ_metrics.csv';rows=read_csv(p);rows[0]=rows[1].copy();csvfile(p,rows)
            with mock.patch.object(launch,'ROOT',root),self.assertRaisesRegex(RuntimeError,'missing/duplicate organ'):launch.require_baseline_2_complete(run)

    def test_stale_last_checkpoint_cannot_fake_full_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,_=completed_fixture(root)
            with mock.patch.object(launch,'ROOT',root),mock.patch('torch.load',return_value={'epoch':171,'global_step':23736}),self.assertRaisesRegex(RuntimeError,'all training steps'):launch.require_baseline_2_complete(run)

    def test_s3_request_does_not_skip_planned_s2(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            registry={'queue':[{'id':key,'status':'completed' if key in ['M0','M1','S1'] else 'planned'} for key in ['M0','M1','S1','S2','S3']]}
            write(root/'experiment_registry.json',registry);write(root/'orchestration_state.json',{'active_experiment':None})
            with mock.patch.object(launch,'BASE',root),mock.patch.object(launch,'preflight') as preflight,mock.patch.object(launch.subprocess,'Popen') as spawn,contextlib.redirect_stdout(io.StringIO()) as output:
                launch.advance(expected_experiment='S3');preflight.assert_not_called();spawn.assert_not_called()
            self.assertIn('waiting_for_ordered_predecessor',output.getvalue())

    def test_unrequested_experiments_rejected_before_predecessor_io(self):
        with mock.patch.object(launch,'require_baseline_2_complete') as predecessor:
            with self.assertRaisesRegex(RuntimeError,'only authorizes S2/S3'):launch.preflight('S4',require_gpu=False)
            predecessor.assert_not_called()


if __name__=='__main__':unittest.main()
