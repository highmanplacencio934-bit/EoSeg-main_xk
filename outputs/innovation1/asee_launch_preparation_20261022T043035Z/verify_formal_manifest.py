"""Assert the real formal run changes only ASEE and output names/locations."""
from pathlib import Path
import hashlib,json,sys
import yaml
ROOT=Path(__file__).resolve().parents[3]
def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        output=[]
        for key in sorted(a.keys()|b.keys()):output+=diff(a.get(key,'MISSING'),b.get(key,'MISSING'),path+'.'+key if path else key)
        return output
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        output=[]
        for i,(av,bv) in enumerate(zip(a,b)):output+=diff(av,bv,f'{path}.{i}')
        return output
    return [] if a==b else [{'path':path,'baseline':a,'innovation':b}]
def config(manifest):
    c=manifest['resolved_cli_config'];return yaml.safe_load(c) if isinstance(c,str) else c
def main():
    folder=Path(__file__).parent
    output=ROOT/(folder/'formal_run_path.txt').read_text().strip()
    manifest_path=next((output/'synapse_test_selection').glob('*/run_manifest.json'))
    manifest=json.loads(manifest_path.read_text())
    baseline_path=next((ROOT/'outputs/baseline/20261021T132635Z/synapse_test_selection').glob('*/run_manifest.json'))
    baseline=json.loads(baseline_path.read_text())
    a,b=config(baseline),config(manifest)
    changes=diff(a,b)
    expected={'trainer.default_root_dir','trainer.logger.init_args.name','trainer.logger.init_args.save_dir','trainer.callbacks.0.init_args.output_dir','model.init_args.network.class_path'}
    new_options={'prior_path','queries_per_class','patch_update','freeze_text_projection','enhance_patch','enhance_query','use_text','use_probability'}
    expected|={'model.init_args.network.init_args.'+key for key in new_options}
    unexpected=[row for row in changes if row['path'] not in expected]
    assert not unexpected,unexpected
    assert b['trainer']['max_epochs']==220 and b['trainer']['limit_train_batches'] is None
    assert b['trainer']['callbacks'][0]['init_args']['last_n_epochs']==50
    assert len(b['trainer']['callbacks'])==1 and not b['compile_disabled']
    assert manifest['device']=='cuda:0' and manifest['estimated_steps']==30360
    assert b['model']['init_args']['network']['init_args']['patch_update']=='plan'
    before=json.loads((folder/'source_before_launch.json').read_text())
    drift=[name for name,digest in before.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    assert not drift,drift
    result={'passed':True,'manifest':str(manifest_path),'baseline_manifest':str(baseline_path),'changes':changes,'unexpected_config_changes':unexpected,'protected_source_drift':drift,'protected_files':len(before),'estimated_steps':manifest['estimated_steps'],'device':manifest['device'],'prior_sha256':hashlib.sha256((ROOT/'priors/synapse_asee_v1.pt').read_bytes()).hexdigest()}
    (folder/'formal_manifest_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
