"""Low-observer G5: native loader and original training path; hashes only at boundaries."""
from __future__ import annotations
import argparse,functools,hashlib,json,os,pickle,random,runpy,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def write(path,value):
    path.write_text(json.dumps(value,indent=2,default=str)+"\n")

def compare_records(left,right):
    keys=("profile","compiled","initial_state","initial_rng","optimizer_groups","estimated_steps","precision", "batches", "actual_optimizer_updates", "losses", "lrs", "scaler", "final_state", "final_optimizer", "final_scheduler", "final_rng", "strict")
    for value in (left,right):
        if value.get("state")!="captured_native_batch_target" or not 20<=value.get("batches",0)<=100 or value.get("actual_optimizer_updates",0)<2:
            raise ValueError("Incomplete low-observer trajectory")
        if any(key not in value for key in keys):raise ValueError("Missing trajectory evidence")
        if value['strict'] is not True or value['compiled'] is not True or value['precision']!='16-mixed' or value['estimated_steps']!=30360:
            raise ValueError("Wrong strict/compile/precision/schedule protocol")
        if len(value['losses'])!=value['batches'] or len(value['scaler'])!=value['batches'] or len(value['lrs'])!=value['actual_optimizer_updates']:
            raise ValueError("Incomplete per-batch/update observations")
        if not value['initial_state'] or not value['final_state'] or not value['final_optimizer']:
            raise ValueError("Empty model or optimizer state")
    differences=[key for key in keys if left[key]!=right[key]]
    return {"exact_equality":not differences,"differences":differences,"checked_fields":list(keys)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('deterministic','compare','self-test'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--profile',choices=('baseline','asee','ccps'),default='baseline')
    parser.add_argument('--tag');parser.add_argument('--left');parser.add_argument('--right')
    parser.add_argument('--batch-target',type=int,default=20)
    parser.add_argument('--required-free-mib',type=int,default=22000)
    parser.add_argument('--target-updates',type=int,default=2)
    parser.add_argument('--max-attempts',type=int,default=16)
    args=parser.parse_args();base=args.output_dir.resolve();base.mkdir(parents=True,exist_ok=True)
    if args.mode=='compare':
        left,right=[json.loads((base/('trajectory_'+tag)/'low_trace.json').read_text()) for tag in (args.left,args.right)]
        result=compare_records(left,right);write(base/('low_comparison_'+args.left+'_'+args.right+'.json'),result)
        print(json.dumps(result));raise SystemExit(0 if result['exact_equality'] else 1)
    if args.mode=='deterministic':
        if not 20<=args.batch_target<=100:raise SystemExit('Need 20-100 native batches')
        if args.required_free_mib<21606:raise SystemExit('Do not lower memory safety floor')
        free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<args.required_free_mib:raise SystemExit('Insufficient memory reserve')
        if os.environ.get('CUDA_VISIBLE_DEVICES')=='':raise SystemExit('Cannot inherit CPU-only visibility')
    import numpy as np
    import torch
    from lightning.pytorch import Callback,Trainer
    torch.set_num_threads(1)
    def pack(value):
        if isinstance(value,torch.Tensor):
            t=value.detach().cpu().contiguous()
            header=json.dumps({'dtype':str(t.dtype),'shape':list(t.shape)},sort_keys=True).encode()
            return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(header+t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        if isinstance(value,dict):return {str(k):pack(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [pack(v) for v in value]
        return value
    def rng():
        result={'cpu':pack(torch.get_rng_state()),'python':hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),'numpy':hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()}
        if torch.cuda.is_initialized():result['cuda']=[pack(x) for x in torch.cuda.get_rng_state_all()]
        return result
    def state(module):return pack(module.state_dict())
    if args.mode=='self-test':
        t=torch.arange(5.)
        before=torch.get_rng_state().clone();assert pack(t)==pack(t.clone());assert torch.equal(before,torch.get_rng_state())
        fixture={k:{} for k in ('initial_state','initial_rng','optimizer_groups','estimated_steps','precision','losses','lrs','scaler','final_state','final_optimizer','final_scheduler','final_rng','strict')}
        fixture.update(state='captured_native_batch_target',batches=20,actual_optimizer_updates=2,
            profile='baseline',compiled=True,strict=True,precision='16-mixed',estimated_steps=30360,
            initial_state={'fixture':1},final_state={'fixture':1},final_optimizer={'fixture':1},
            losses=[1]*20,scaler=[1]*20,lrs=[[0],[4e-7]])
        assert compare_records(fixture,fixture)['exact_equality']
        other=dict(fixture,final_optimizer={'fixture_difference':1});assert not compare_records(fixture,other)['exact_equality']
        incomplete=dict(fixture,batches=1)
        try:compare_records(fixture,incomplete)
        except ValueError:pass
        else:raise AssertionError('Accepted incomplete fixture')
        write(base/'low_observer_cpu_self_test.json',{'passed':True,'snapshot_rng_unchanged':True,'difference_rejected':True,'incomplete_rejected':True,'cuda_initialized':torch.cuda.is_initialized()})
        return
    os.chdir(ROOT)
    run=base/('trajectory_'+args.tag);run.mkdir(exist_ok=False)
    trace={'profile':args.profile,'observer':'low; no criterion, sample or gradient instrumentation','state':'initializing','batches':0,'actual_optimizer_updates':0,'lrs':[],'scaler':[]}
    losses=[];starts=[];ends=[];wall=[]
    class LowCapture(Callback):
        def on_train_start(self,trainer,module):
            assert trainer.estimated_stepping_batches==30360 and trainer.max_epochs==220 and trainer.precision=='16-mixed'
            trace.update(estimated_steps=int(trainer.estimated_stepping_batches),precision=trainer.precision,
                         strict=torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled(),
                         optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in trainer.optimizers[0].param_groups])
            assert trace['strict']
            native=trainer.strategy.pre_backward
            @functools.wraps(native)
            def observe(loss,*a,**kw):
                losses.append(loss.detach().clone())
                return native(loss,*a,**kw)
            trainer.strategy.pre_backward=observe
            def before(optimizer,*a):
                trace['lrs'].append([g['lr'] for g in optimizer.param_groups])
            def after(*a):trace['actual_optimizer_updates']+=1
            self.handles=[trainer.optimizers[0].register_step_pre_hook(before),trainer.optimizers[0].register_step_post_hook(after)]
        def on_train_batch_start(self,trainer,module,batch,batch_idx):
            assert batch[0].shape==(16,3,384,384)
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            starts.append(begin);ends.append(end);begin.record();self.wall_start=time.perf_counter()
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            ends[-1].record();wall.append(time.perf_counter()-self.wall_start)
            trace['batches']+=1
            trace['scaler'].append(trainer.precision_plugin.scaler.state_dict())
            if trace['batches']>=args.batch_target:trainer.should_stop=True
        def on_train_end(self,trainer,module):
            raw=getattr(module,'_orig_mod',module);optimizer=trainer.optimizers[0]
            torch.cuda.synchronize()
            trace['batch_cuda_ms']=[start.elapsed_time(end) for start,end in zip(starts,ends)]
            trace['batch_wall_seconds']=wall
            trace['losses']=[pack(loss) for loss in losses]
            trace['loss_values']=[float(loss.cpu()) for loss in losses]
            trace['final_state']=state(raw)
            names={id(p):name for name,p in raw.named_parameters()}
            trace['final_optimizer']={names[id(p)]:pack(value) for p,value in optimizer.state.items()}
            trace['final_scheduler']=pack(trainer.lr_scheduler_configs[0].scheduler.state_dict())
            trace['final_rng']=rng()
            trace['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
            trace['peak_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2
            trace['state']='captured_native_batch_target' if trace['batches']==args.batch_target and trace['actual_optimizer_updates']>=2 else 'incomplete'
            for handle in self.handles:handle.remove()
            write(run/'low_trace.json',trace)
        def on_exception(self,trainer,module,error):
            trace.update(state='failed',error=repr(error));write(run/'low_trace.json',trace)
    original_fit=Trainer.fit
    @functools.wraps(original_fit)
    def intercepted(trainer,model,*a,**kw):
        raw=getattr(model,'_orig_mod',model)
        trace.update(initial_state=state(raw),initial_rng=rng(),compiled=hasattr(model,'_orig_mod'))
        assert trace['compiled']
        trainer.callbacks.append(LowCapture())
        return original_fit(trainer,model,*a,**kw)
    Trainer.fit=intercepted
    entry=ROOT/'scripts/run_synapse_deterministic.py'
    cb='training.synapse_physical_selection.SynapsePhysicalHD95Selection'
    sys.argv=[str(entry),'fit','--config','configs/synapse/deterministic/'+args.profile+'.yaml',
        '--seed_everything=0','--trainer.default_root_dir='+str(run),'--trainer.logger=false',
        '--trainer.enable_progress_bar=false','--trainer.callbacks='+json.dumps([{'class_path':cb,'init_args':{'last_n_epochs':50,'expected_test_cases':12,'output_dir':str(run)}}]),
        '--deterministic_probe_batches=100','--deterministic_output_dir='+str(run/'protocol_records')]
    runpy.run_path(str(entry),run_name='__main__')
    if trace['state']!='captured_native_batch_target':raise SystemExit('Low observer did not capture the requested trajectory')

if __name__=='__main__':main()
