from pathlib import Path
import json,struct,sys
from wandb.proto.wandb_internal_pb2 import Record
ROOT=Path('/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse')
BASE=ROOT/'outputs/baseline-repair_2'
A=Path(json.loads((BASE/'repeat_1.json').read_text())['run'])
B=Path(json.loads((BASE/'repeat_2.json').read_text())['run'])
def histories(folder):
 files=list(folder.glob('wandb/offline-run-*/*.wandb'))
 if len(files)!=1:return []
 raw=files[0].read_bytes();pos=7;pending=b'';rows=[]
 while pos+7<=len(raw):
  remain=32768-pos%32768
  if remain<7:pos+=remain;continue
  crc,n,t=struct.unpack_from('<IHB',raw,pos);pos+=7
  if n==0 and t==0:pos+=32768-pos%32768;continue
  if pos+n>len(raw):break
  data=raw[pos:pos+n];pos+=n
  if t==1:payload=data
  elif t==2:pending=data;continue
  elif t==3:pending+=data;continue
  elif t==4:payload=pending+data;pending=b''
  else:raise ValueError((pos,t,n))
  rec=Record();rec.ParseFromString(payload)
  if rec.HasField('history'):
   rows.append({item.key or '.'.join(item.nested_key):json.loads(item.value_json) for item in rec.history.item})
 return rows
ha,hb=histories(A),histories(B)
la={int(x['epoch']):x for x in ha if 'losses/train_loss_total' in x}
lb={int(x['epoch']):x for x in hb if 'losses/train_loss_total' in x}
report={'reference':str(A),'current':str(B),'completed_loss_epochs_available':len(lb),'epochs':[]}
for e in sorted(set(la)&set(lb))[:20]:
 row={'epoch':e+1,'reference_global_step':la[e]['trainer/global_step'],'current_global_step':lb[e]['trainer/global_step'],'losses':{}}
 for k in sorted(k for k in la[e] if k.startswith('losses/')):
  x,y=la[e][k],lb[e][k];row['losses'][k]={'reference':x,'current':y,'difference':y-x,'exact_equal':x==y}
 report['epochs'].append(row)
pa,pb=A/'startup_verification.json',B/'startup_verification.json'
if pa.exists() and pb.exists():
 a,b=json.loads(pa.read_text()),json.loads(pb.read_text())
 report['initial_state_equal']=a['initial_state']==b['initial_state'];report['initial_state_entries']=len(b['initial_state']);report['initial_rng_equal']=a['rng']==b['rng']
 report['context_equal']={k:a.get(k)==b.get(k) for k in ['network','profile','seed','max_epochs','estimated_steps','optimizer_groups','compute_hd95_voxel','compiled','device']}
fa,fb=list(A.glob('synapse_test_selection/*/run_manifest.json')),list(B.glob('synapse_test_selection/*/run_manifest.json'))
if len(fa)==len(fb)==1:
 a,b=json.loads(fa[0].read_text()),json.loads(fb[0].read_text())
 report['optimizer_groups_exact_equal']=a['optimizer_groups']==b['optimizer_groups'];report['optimizer_defaults_exact_equal']=a['optimizer_defaults']==b['optimizer_defaults'];report['source_hashes_exact_equal']=a['source_sha256']==b['source_sha256'];report['data_list_hashes_equal']=a['manifest_sha256']==b['manifest_sha256']
 def lr(rows):
  return {x['trainer/global_step']:{k:v for k,v in x.items() if k.startswith('lr-')} for x in rows if 'trainer/global_step' in x and any(k.startswith('lr-') for k in x)}
 x,y=lr(ha),lr(hb);steps=sorted(set(x)&set(y))[:10]
 report['lr_log_shared_steps']=steps;report['lr_log_groups_equal']={str(s):x[s]==y[s] for s in steps}
(B/'early_epoch_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
