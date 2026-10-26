import unittest
from unittest.mock import patch
from main import LightningCLI
from lightning.pytorch.cli import LightningCLI as BaseCLI
class PhysicalCheckpointCLI(unittest.TestCase):
 def test_physical_test_preserves_explicit_wrapper(self):
  cli=object.__new__(LightningCLI);cli.config={'subcommand':'test','test':{'model':{'class_path':'training.synapse_physical_semantic.SynapsePhysicalHD95Semantic'}}}
  with patch.object(BaseCLI,'_parse_ckpt_path') as original:
   cli._parse_ckpt_path();original.assert_not_called()
 def test_original_fit_and_test_keep_checkpoint_behavior(self):
  for sub,cls in [('fit','training.synapse_physical_semantic.SynapsePhysicalHD95Semantic'),('test','training.mask_classification_semantic.MaskClassificationSemantic')]:
   cli=object.__new__(LightningCLI);cli.config={'subcommand':sub,sub:{'model':{'class_path':cls}}}
   with patch.object(BaseCLI,'_parse_ckpt_path') as original:
    cli._parse_ckpt_path();original.assert_called_once_with()
