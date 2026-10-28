"""Mathematical equivalence and local dependency isolation; CUDA gates are separate."""
import unittest
import torch
from torch.nn import functional as F
from training.mask_classification_loss import MaskClassificationLoss
from training.deterministic_mask_loss import DeterministicMaskClassificationLoss

class DeterministicLossTests(unittest.TestCase):
    def make(self,cls=MaskClassificationLoss):
        return cls(64,3.,.75,5.,5.,2.,9,.1).double()

    def test_from_existing_preserves_state_rng_mode(self):
        original=self.make().eval()
        rng=torch.get_rng_state().clone()
        candidate=DeterministicMaskClassificationLoss.from_existing(original)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(set(original.state_dict()),set(candidate.state_dict()))
        self.assertEqual(list(original.parameters()),list(candidate.parameters()))
        self.assertFalse(candidate.training)
        for key,value in original.state_dict().items():
            self.assertTrue(torch.equal(value,candidate.state_dict()[key]))

    def test_weighted_mean_and_noncontiguous_background_noobject(self):
        c=self.make(DeterministicMaskClassificationLoss)
        x=torch.randn(2,10,5,dtype=torch.double).transpose(1,2).requires_grad_(True)
        labels=[torch.tensor([0,3,8]),torch.tensor([1])]
        indices=[(torch.tensor([0,2,4]),torch.arange(3)),(torch.tensor([1]),torch.tensor([0]))]
        target=torch.full((2,5),9,dtype=torch.long)
        target[0,[0,2,4]]=labels[0]; target[1,1]=1
        output=c.loss_labels(x,labels,indices)['loss_cross_entropy']
        logp=x.log_softmax(-1).reshape(-1,10)
        weights=c.empty_weight[target.reshape(-1)]
        manual=(-logp[torch.arange(10),target.reshape(-1)]*weights).sum()/weights.sum()
        torch.testing.assert_close(output,manual,rtol=1e-12,atol=1e-12)
        self.assertIsNotNone(torch.autograd.grad(output,x)[0])

    def test_all_unmatched_ce_matches_original(self):
        original=self.make(); candidate=DeterministicMaskClassificationLoss.from_existing(original)
        logits=torch.randn(2,5,10,dtype=torch.double,requires_grad=True)
        labels=[torch.tensor([],dtype=torch.long)]*2
        indices=[(torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long))]*2
        a=original.loss_labels(logits,labels,indices)['loss_cross_entropy']
        b=candidate.loss_labels(logits,labels,indices)['loss_cross_entropy']
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(a,logits)[0],torch.autograd.grad(b,logits)[0],rtol=1e-12,atol=1e-12)

    def test_full_criterion_preserves_rng_samples_matching_and_gradients(self):
        import transformers.models.mask2former.modeling_mask2former as hf
        native_sample=hf.sample_point
        original=self.make().float(); candidate=DeterministicMaskClassificationLoss.from_existing(original)
        torch.manual_seed(71)
        masks=torch.randn(2,5,8,8,dtype=torch.float32)
        logits=torch.randn(2,5,10,dtype=torch.float32)
        targets=[{'masks':(torch.rand(3,8,8)>.5).float(),'labels':torch.tensor([0,3,8])},
                 {'masks':(torch.rand(2,8,8)>.5).float(),'labels':torch.tensor([1,6])}]
        state=torch.get_rng_state().clone()
        outputs=[]
        for criterion in (original,candidate):
            torch.set_rng_state(state)
            m=masks.clone().requires_grad_(True); c=logits.clone().requires_grad_(True)
            losses=criterion(m,targets,c)
            total=criterion.loss_total(losses,lambda *a,**kw:None)
            total.backward()
            outputs.append((losses,m.grad,c.grad,torch.get_rng_state().clone()))
        self.assertIs(hf.sample_point,native_sample)
        for key in outputs[0][0]:
            torch.testing.assert_close(outputs[0][0][key],outputs[1][0][key],rtol=1e-4,atol=1e-5)
        for index in (1,2):
            torch.testing.assert_close(outputs[0][index],outputs[1][index],rtol=1e-4,atol=1e-5)
        self.assertTrue(torch.equal(outputs[0][3],outputs[1][3]))

if __name__=='__main__': unittest.main()
