"""Startup regressions without allocating the real baseline backbone."""
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.run_artifacts import log_project_code


class ArchivalTests(unittest.TestCase):
    def test_disabled_logger(self):
        log_project_code(None)

    def test_missing_ignore_and_reference_exclusion(self):
        experiment = SimpleNamespace(log_code=Mock())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_project_code(SimpleNamespace(experiment=experiment), root)
            args, kwargs = experiment.log_code.call_args
            self.assertEqual(args[0], str(root.resolve()))
            self.assertTrue(kwargs["exclude_fn"](root / "project_TransUNet/a.py"))
            self.assertTrue(kwargs["exclude_fn"](root / "outputs/a.yaml"))
            self.assertFalse(kwargs["exclude_fn"](root / "models/a.py"))
            self.assertFalse(kwargs["include_fn"](root / "data.npz"))

    def test_archival_failure_does_not_abort_fit(self):
        experiment = SimpleNamespace(log_code=Mock(side_effect=OSError("offline")))
        with tempfile.TemporaryDirectory() as tmp, self.assertLogs(level="WARNING"):
            log_project_code(SimpleNamespace(experiment=experiment), Path(tmp))


class CLIRegressionTests(unittest.TestCase):
    def test_dino_import_does_not_import_rvsa(self):
        sys.modules.pop("models.vit", None)
        from models.vit import ViT
        self.assertEqual(ViT.__init__.__defaults__[0], 16)
        self.assertNotIn("models.vit_rvsa_mtp", sys.modules)

    def test_custom_fit_with_no_logger(self):
        from main import LightningCLI
        loop = SimpleNamespace()
        trainer = SimpleNamespace(logger=None, fit=Mock(), fit_loop=SimpleNamespace(epoch_loop=loop))
        cli = SimpleNamespace(trainer=trainer, config={"subcommand": "fit", "fit": {"compile_disabled": True}})
        model = object()
        LightningCLI.fit(cli, model)
        trainer.fit.assert_called_once_with(model)

    def test_independent_test_default_stays_eager(self):
        import torch
        from main import LightningCLI
        model = torch.nn.Linear(2, 2)
        original = model.forward
        trainer = SimpleNamespace(test=Mock(return_value=[{"test/mean_dice": 0.5}]))
        cli = SimpleNamespace(trainer=trainer, config={"subcommand": "test", "test": {
            "compile_test": False, "compile_disabled": False}})
        with patch("main.torch.compile") as compiler:
            result = LightningCLI.test(cli, model, ckpt_path="model.ckpt")
        compiler.assert_not_called()
        trainer.test.assert_called_once_with(model, ckpt_path="model.ckpt")
        self.assertEqual(result, [{"test/mean_dice": 0.5}])
        self.assertEqual(model.forward, original)

    def test_independent_test_compiles_only_forward_and_restores_it(self):
        import torch
        from main import LightningCLI
        for fail in (False, True):
            with self.subTest(trainer_fails=fail):
                model = torch.nn.Linear(2, 2)
                original = model.forward
                compiled = Mock(wraps=original)
                inputs = torch.ones(1, 2)
                expected = model(inputs)

                def run(actual_model, **kwargs):
                    self.assertIs(actual_model, model)
                    self.assertIs(actual_model.forward, compiled)
                    self.assertEqual(kwargs, {"ckpt_path": "model.ckpt"})
                    self.assertTrue(torch.equal(actual_model(inputs), expected))
                    if fail:
                        raise RuntimeError("test failed")
                    return [0.5]

                trainer = SimpleNamespace(test=Mock(side_effect=run))
                cli = SimpleNamespace(trainer=trainer, config={"subcommand": "test", "test": {
                    "compile_test": True, "compile_disabled": False}})
                with patch("main.torch.compile", return_value=compiled) as compiler:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "test failed"):
                            LightningCLI.test(cli, model, ckpt_path="model.ckpt")
                    else:
                        self.assertEqual(LightningCLI.test(cli, model, ckpt_path="model.ckpt"), [0.5])
                compiler.assert_called_once_with(original)
                self.assertEqual(model.forward, original)

    def test_independent_test_rejects_conflicting_compile_flags(self):
        import torch
        from main import LightningCLI
        trainer = SimpleNamespace(test=Mock())
        cli = SimpleNamespace(trainer=trainer, config={"subcommand": "test", "test": {
            "compile_test": True, "compile_disabled": True}})
        with patch("main.torch.compile") as compiler, self.assertRaisesRegex(ValueError, "cannot be combined"):
            LightningCLI.test(cli, torch.nn.Linear(2, 2))
        compiler.assert_not_called()
        trainer.test.assert_not_called()

    def test_real_cli_checkpoint_schema_roundtrip(self):
        import torch
        from timm.models.vision_transformer import VisionTransformer
        from main import LightningCLI, LightningModule, LightningDataModule

        def tiny_factory(name, **kwargs):
            return VisionTransformer(img_size=kwargs["img_size"], patch_size=kwargs["patch_size"],
                                     embed_dim=32, depth=4, num_heads=4, reg_tokens=4,
                                     no_embed_class=True, init_values=1e-5, num_classes=0)

        class ProbeComplete(Exception):
            pass

        class ProbeCLI(LightningCLI):
            def before_instantiate_classes(self):
                raise ProbeComplete()

        from lightning.pytorch.callbacks import LearningRateMonitor
        options = dict(subclass_mode_model=True, subclass_mode_data=True, save_config_callback=None,
                       seed_everything_default=0, trainer_defaults={"callbacks": [LearningRateMonitor()]})
        args = ["--config", str(ROOT / "configs/synapse/vit_query_mul_scale_fusion.yaml"),
                "--trainer.accelerator=cpu", "--trainer.precision=32-true", "--trainer.logger=false",
                "--trainer.enable_model_summary=false"]
        with patch("timm.create_model", side_effect=tiny_factory):
            cli = LightningCLI(LightningModule, LightningDataModule, run=False, args=args, **options)
            self.assertEqual(cli.model.hparams["_class_path"],
                             "training.mask_classification_semantic.MaskClassificationSemantic")
            self.assertFalse(cli.config["compile_test"])
            self.assertFalse(cli.trainer.loggers)
            self.assertFalse(any(isinstance(cb, LearningRateMonitor) for cb in cli.trainer.callbacks))
            payload = {"state_dict": cli.model.state_dict(), "hyper_parameters": dict(cli.model.hparams)}
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.ckpt"
                torch.save(payload, path)
                torch.load(path, weights_only=True)
                with self.assertRaises(ProbeComplete):
                    ProbeCLI(LightningModule, LightningDataModule,
                             args=["test"] + args + ["--ckpt_path", str(path), "--compile_test"], **options)


if __name__ == "__main__":
    unittest.main()
