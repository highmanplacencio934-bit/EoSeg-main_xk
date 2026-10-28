"""Exact trace comparison that preserves native AMP overflow skips."""
from pathlib import Path
import argparse,json,hashlib

ATTEMPT_FIELDS=("batch","rng_before_batch","forward","random_samples","sample_point_calls","matcher",
    "loss_components","loss_total","loss_total_scaled","scaled_gradients","unscaled_gradients",
    "clipped_gradients","weights_after_step","optimizer_step_lrs","optimizer_state_after",
    "state_keys_changed_from_initial","parameter_keys_changed_from_initial","scaler_before","scaler_after","optimizer_called")

def validate_trace(trace, *, batch_target=0,target_updates=2):
    expected='captured_native_batch_target' if batch_target else 'captured_target_optimizer_updates'
    if trace.get('state')!=expected:raise ValueError('Trajectory did not complete its declared target')
    attempts=trace.get('attempts',[])
    if batch_target and (not 20<=batch_target<=100 or len(attempts)!=batch_target):raise ValueError('Incomplete native batch window')
    calls=[(i,x) for i,x in enumerate(attempts) if x.get('optimizer_called')]
    if len(calls)!=trace.get('actual_optimizer_updates') or len(calls)<2 or (not batch_target and len(calls)!=target_updates):
        raise ValueError('Actual AdamW calls do not match recorded target')
    groups=trace.get('optimizer_groups',[])
    if not groups or any(g.get('lr')!=0 for g in groups):raise ValueError('Initial original LR0 boundary is missing')
    required=('batch','rng_before_batch','forward','random_samples','sample_point_calls','matcher','loss_components',
              'loss_total','loss_total_scaled','scaled_gradients','unscaled_gradients','scaler_before','scaler_after','optimizer_called')
    skipped=[]
    for i,attempt in enumerate(attempts):
        if any(key not in attempt for key in required):raise ValueError('Missing observed training boundary')
        if any(not attempt[key] for key in ('forward','random_samples','sample_point_calls','matcher','loss_components','scaled_gradients','unscaled_gradients')):
            raise ValueError('Empty training observation')
        if attempt['optimizer_called']:
            if any(not attempt.get(key) for key in ('clipped_gradients','weights_after_step','optimizer_state_after','optimizer_step_lrs')):
                raise ValueError('Missing actual AdamW observations')
        else:
            before=attempt['scaler_before'];after=attempt['scaler_after']
            if not before or not after or not after['scale']<before['scale']:
                raise ValueError('Skipped AdamW call has no native GradScaler overflow evidence')
            skipped.append(i+1)
    first=attempts[0]
    if first['optimizer_called']:
        if any(lr!=0 for lr in first['optimizer_step_lrs']) or first.get('parameter_keys_changed_from_initial'):
            raise ValueError('Parameters moved during first LR0 AdamW call')
    positive=[(i,x) for i,x in calls if any(lr>0 for lr in x['optimizer_step_lrs']) and x.get('parameter_keys_changed_from_initial')]
    if not positive:raise ValueError('No positive-LR parameter-changing call captured')
    return {'batches':len(attempts),'adamw_calls':len(calls),'amp_skip_batches':skipped,
        'first_lr0_attempt':'adamw_called' if first['optimizer_called'] else 'overflow_skip',
        'first_positive_lr_call_batch':positive[0][0]+1}

def compare_pair(left,right,initial_left,initial_right,*,batch_target=0,target_updates=2):
    semantics=[validate_trace(x,batch_target=batch_target,target_updates=target_updates) for x in (left,right)]
    stages=[('initial_state',initial_left['state'],initial_right['state']),('initial_rng',initial_left['rng'],initial_right['rng'])]
    for key in ('optimizer_groups','estimated_steps','precision','native_batches','target_updates','native_rng_hashes','effective_rng_hashes'):
        stages.append((key,left[key],right[key]))
    stages.append(('attempt_count',len(left['attempts']),len(right['attempts'])))
    for i,(a,b) in enumerate(zip(left['attempts'],right['attempts'])):
        for field in ATTEMPT_FIELDS:stages.append((f'attempt{i+1}.{field}',a.get(field),b.get(field)))
    differences=[name for name,a,b in stages if a!=b]
    return {'exact_equality':not differences,'checked_stages':len(stages),'different_stages':differences,
        'first_difference':differences[0] if differences else None,'semantics':semantics,
        'criterion':'all observed values/hashes exact; no tolerance; native AMP skips retained'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',nargs='?',default='compare',choices=['compare'])
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--left',required=True);p.add_argument('--right',required=True)
    p.add_argument('--batch-target',type=int,default=0);p.add_argument('--target-updates',type=int,default=2)
    a=p.parse_args();base=a.output_dir.resolve();dirs=[base/('trajectory_'+tag) for tag in (a.left,a.right)]
    files=[directory/name for directory in dirs for name in ('trajectory_trace.json','trajectory_initialization.json')]
    trace=[json.loads((d/'trajectory_trace.json').read_text()) for d in dirs]
    init=[json.loads((d/'trajectory_initialization.json').read_text()) for d in dirs]
    result=compare_pair(*trace,*init,batch_target=a.batch_target,target_updates=a.target_updates)
    result.update(left=str(dirs[0]),right=str(dirs[1]),validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(f.relative_to(base)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    dest=base/('amp_comparison_'+a.left+'_'+a.right+'.json')
    if dest.exists():raise RuntimeError('Refuse to overwrite prior comparison evidence')
    dest.write_text(json.dumps(result,indent=2));print(json.dumps(result))
    raise SystemExit(0 if result['exact_equality'] else 1)
if __name__=='__main__':main()
