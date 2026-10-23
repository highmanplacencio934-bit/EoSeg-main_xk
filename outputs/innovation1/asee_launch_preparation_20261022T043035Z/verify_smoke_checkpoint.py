"""CPU restore/provenance check for the real ASEE smoke checkpoint."""
from pathlib import Path
import hashlib,json,sys
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from models.vit import ViT
from models.eoseg_asee import EoSegMultiQSegFusionASEE
from models.asee import load_prior

def main():
    torch.set_num_threads(2)
    root=ROOT/Path(__file__).with_name('smoke_run_path.txt').read_text().strip()
    status=json.loads((root/'status.json').read_text())
    assert status['state']=='completed' and status['exit_code']==0,status
    folders=list((root/'synapse_test_selection').iterdir());assert len(folders)==1
    folder=folders[0]
    saved=torch.load(folder/'best_test.ckpt',map_location='cpu',weights_only=True,mmap=True)
    network_state={key.removeprefix('network.'):value for key,value in saved['state_dict'].items() if key.startswith('network.')}
    prior,digest=load_prior('priors/synapse_asee_v1.pt')
    assert network_state['_extra_state']['artifact_sha256']==digest
    # Restore the entire encoder from checkpoint; suppress a redundant pretrained load.
    encoder=ViT(img_size=(384,384),ckpt_path='CPU_RESTORE_FROM_FULL_SMOKE_STATE')
    network=EoSegMultiQSegFusionASEE(encoder,9,'priors/synapse_asee_v1.pt')
    incompatible=network.load_state_dict(network_state,strict=True)
    actual=network.state_dict()
    tensors=[key for key in actual if isinstance(actual[key],torch.Tensor)]
    assert all(torch.equal(actual[key],network_state[key]) for key in tensors)
    assert actual['_extra_state']==network_state['_extra_state']
    assert torch.equal(actual['asee.text_features'],prior['text_features'])
    assert torch.equal(actual['asee.conditional_matrix'],prior['conditional_matrix'])
    assert not any('_orig_mod' in key for key in saved['state_dict'])
    for name in ['text_proj','prob_proj','ffn_patch_in','norm_patch_in','ffn_patch_out','norm_patch_out','query_prior_proj']:
        assert all(param.requires_grad for param in getattr(network.asee,name).parameters())
    runtime=json.loads((root/'runtime_audit/runtime_fit.json').read_text())
    assert len(runtime['steps'])==8 and all(x['finite_unscaled_gradients'] for x in runtime['steps'][-2:])
    assert runtime['steps'][-1]['actual_optimizer_updates']>=2
    import csv
    rows=list(csv.DictReader((folder/'test_case_metrics.csv').open()));assert len(rows)==12 and len({x['case'] for x in rows})==12
    result={'passed':True,'checkpoint':str(folder/'best_test.ckpt'),'checkpoint_epoch_zero_based':saved['epoch'],'tensor_keys_restored_exactly':len(tensors),'metadata_matches':True,'prior_sha256':digest,'asee_parameters':sum(x.numel() for x in network.asee.parameters()),'missing_keys':incompatible.missing_keys,'unexpected_keys':incompatible.unexpected_keys,'train_steps':runtime['steps'],'test_cases':len(rows),'peak_allocated_bytes':max(x.get('gpu_peak_allocated_bytes',0) for x in runtime['resources']),'peak_reserved_bytes':max(x.get('gpu_peak_reserved_bytes',0) for x in runtime['resources']),'dynamo_counters':runtime['dynamo_counters']}
    Path(__file__).with_name('smoke_checkpoint_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
