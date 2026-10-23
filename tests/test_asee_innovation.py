"""CPU-only ASEE contracts, train-prior isolation, and baseline-equivalence regression."""
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
import yaml
from timm.models.vision_transformer import VisionTransformer

from models.asee import AnatomicalSemanticEncoding, load_prior, validate_prior, TRAIN_LIST_SHA256, TEST_LIST_SHA256
from models.eoseg import EoSegMultiQSegFusion
from models.eoseg_asee import EoSegMultiQSegFusionASEE
from scripts.build_synapse_asee_prior import count_training_slices, conditional_from_counts, load_label_mapping, build_prior

ROOT = Path(__file__).resolve().parents[1]


def fixture_prior():
    # Synthetic tensors for unit tests ONLY; never published as training priors.
    return {"schema_version": 1, "dataset": "synapse", "class_ids": list(range(1, 9)),
            "class_names": [f"test_class_{i}" for i in range(1, 9)],
            "label_mapping_verified": True, "label_mapping_source": "synthetic test fixture",
            "statistics": {"unit": "train_npz_slice", "num_samples": 2211,
                           "train_list_sha256": TRAIN_LIST_SHA256, "test_list_sha256": TEST_LIST_SHA256},
            "text_features": torch.randn(8, 12), "conditional_matrix": torch.randn(8, 8)}


def tiny_encoder():
    encoder = nn.Module()
    encoder.backbone = VisionTransformer(img_size=32, patch_size=16, embed_dim=32,
                                        depth=4, num_heads=4, reg_tokens=4, num_classes=0)
    encoder.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))
    encoder.register_buffer("pixel_std", torch.ones(1, 3, 1, 1))
    return encoder


class ASEETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test_prior.pt"
        self.prior = fixture_prior()
        torch.save(self.prior, self.path)

    def module(self, **kwargs):
        return AnatomicalSemanticEncoding(32, self.prior["text_features"], self.prior["conditional_matrix"], **kwargs)

    def network(self, **kwargs):
        return EoSegMultiQSegFusionASEE(tiny_encoder(), 9, str(self.path), **kwargs)

    def test_plan_formula_attention_and_gradients(self):
        module = self.module()
        p = torch.randn(2, 4, 32, requires_grad=True)
        q = torch.randn(2, 16, 32, requires_grad=True)
        out, queries, attn = module(p, q)
        e = module.text_proj(module.text_features) + module.prob_proj(module.conditional_matrix)
        l = module.norm_patch_in(module.ffn_patch_in(p) + p)
        a = torch.softmax((l @ e.T) / (32 ** .5), dim=-1)
        ctx = a @ e
        torch.testing.assert_close(out, module.norm_patch_out(module.ffn_patch_out(ctx) + ctx + p), rtol=0, atol=0)
        torch.testing.assert_close(queries, q + module.query_prior_proj(e.repeat_interleave(2, 0)).unsqueeze(0), rtol=0, atol=0)
        torch.testing.assert_close(attn.sum(-1), torch.ones(2, 4))
        (out.square().mean() + queries.square().mean()).backward()
        self.assertFalse(module.text_features.requires_grad)
        self.assertFalse(module.conditional_matrix.requires_grad)
        for name, param in module.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        self.assertGreater(module.text_proj[0].weight.grad.abs().sum(), 0)
        self.assertGreater(module.prob_proj[0].weight.grad.abs().sum(), 0)

    def test_reference_variant_matches_supplied_reference(self):
        spec = importlib.util.spec_from_file_location("asee_reference_readonly", ROOT / "asee_module.py")
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        text = Path(self.temp.name) / "text.pt"
        matrix = Path(self.temp.name) / "matrix.pt"
        torch.save(self.prior["text_features"], text)
        torch.save(self.prior["conditional_matrix"], matrix)
        original = ref.ASEEModule(embed_dim=32, text_dim=12, text_emb_path=text, cond_prob_path=matrix)
        module = self.module(patch_update="reference_gated")
        state = {k.replace("text_emb", "text_features").replace("cond_prob_matrix", "conditional_matrix"): v
                 for k, v in original.state_dict().items()}
        module.load_state_dict(state, strict=True)
        args = (torch.randn(2, 4, 32), torch.randn(2, 16, 32))
        for actual, expected in zip(module(*args), original(*args)):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    def test_disabled_asee_is_exact_baseline_in_eval_and_train(self):
        baseline = EoSegMultiQSegFusion(tiny_encoder(), 9, 16)
        model = self.network(enhance_patch=False, enhance_query=False)
        incompatible = model.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(k.startswith("asee.") or k == "_extra_state" for k in incompatible.missing_keys))
        image = torch.rand(2, 3, 32, 32)
        for training in (False, True):
            baseline.train(training); model.train(training)
            for actual, expected in zip(model(image), baseline(image)):
                self.assertEqual(len(actual), 1)
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)

    def test_prefix_preserved_and_injection_occurs_once_before_tail(self):
        model = self.network().eval()
        seen = {}
        def capture_previous(_module, _inputs, output):
            seen["before"] = output.detach().clone()
        def capture_tail(_module, inputs):
            seen["tail"] = inputs[0].detach().clone()
        def capture_asee(_module, _inputs, _output):
            seen["calls"] = seen.get("calls", 0) + 1
        handles = [model.encoder.backbone.blocks[0].register_forward_hook(capture_previous),
                   model.encoder.backbone.blocks[1].register_forward_pre_hook(capture_tail),
                   model.asee.register_forward_hook(capture_asee)]
        try:
            masks, classes = model(torch.rand(2, 3, 32, 32))
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(seen["calls"], 1)
        self.assertEqual(seen["tail"].shape, (2, 25, 32))  # 16 query + CLS + 4 registers + 4 patches
        torch.testing.assert_close(seen["tail"][:, 16:21], seen["before"][:, :5], rtol=0, atol=0)
        self.assertEqual(masks[0].shape, (2, 16, 32, 32))
        self.assertEqual(classes[0].shape, (2, 16, 10))
        self.assertTrue(torch.isfinite(masks[0]).all())

    def test_freeze_and_ablation_paths(self):
        module = self.module(freeze_text_projection=True, enhance_patch=False, use_probability=False)
        p, q = torch.randn(2, 4, 32), torch.randn(2, 16, 32)
        patches, queries, attention = module(p, q)
        self.assertIs(patches, p)
        self.assertIsNone(attention)
        queries.square().mean().backward()
        self.assertTrue(all(t.grad is None for t in module.text_proj.parameters()))
        self.assertTrue(all(not t.requires_grad for t in module.prob_proj.parameters()))
        patch_only = self.module(enhance_query=False, use_text=False)
        _, output_queries, attention = patch_only(p, q)
        self.assertIs(output_queries, q)
        self.assertIsNotNone(attention)

    def test_bad_prior_mapping_shape_split_and_nonfinite_rejected(self):
        for mutate in (
                lambda p: p.update(label_mapping_verified=False),
                lambda p: p.update(class_ids=list(range(8))),
                lambda p: p.update(conditional_matrix=torch.zeros(8, 7)),
                lambda p: p["statistics"].update(train_list_sha256="different"),
                lambda p: p.update(text_features=torch.full((8, 12), float("nan")))):
            bad = copy.deepcopy(self.prior); mutate(bad)
            with self.assertRaises(ValueError): validate_prior(bad)
        with self.assertRaises(ValueError): self.network(num_q=9)
        with self.assertRaises(ValueError): self.module()(torch.zeros(1, 4, 32), torch.zeros(1, 15, 32))

    def test_checkpoint_provenance_roundtrip_and_wrong_prior_rejected(self):
        model = self.network().eval()
        output = model(torch.rand(1, 3, 32, 32))
        ckpt = Path(self.temp.name) / "model.ckpt"
        torch.save(model.state_dict(), ckpt)
        state = torch.load(ckpt, weights_only=True, map_location="cpu")
        clone = self.network().eval()
        clone.load_state_dict(state, strict=True)
        self.assertEqual(clone.prior_metadata, model.prior_metadata)
        for key, value in model.state_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(clone.state_dict()[key], value, rtol=0, atol=0)
        bad = copy.deepcopy(state); bad["_extra_state"]["artifact_sha256"] = "wrong"
        with self.assertRaises(ValueError): clone.load_state_dict(bad, strict=True)
        self.assertTrue(torch.isfinite(output[0][0]).all())

    def test_cooccurrence_matches_reference_and_uses_only_requested_slices(self):
        directory = Path(self.temp.name)
        np.savez(directory / "case0005_slice000.npz", label=np.array([[0, 1, 2], [3, 4, 0]], dtype=np.int16))
        np.savez(directory / "case0005_slice001.npz", label=np.array([[0, 5, 6], [7, 8, 1]], dtype=np.float32))
        # This malformed extra file is deliberately absent from the training manifest.
        (directory / "case0008_slice000.npz").write_bytes(b"not a training sample")
        counts, pairs, hashes = count_training_slices(directory, ["case0005_slice000", "case0005_slice001"])
        np.testing.assert_array_equal(counts, [2, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(pairs[1, 2], 1)
        self.assertEqual(pairs[1, 5], 0)
        eps = 1e-5
        expected = np.log(((pairs + eps) / (counts[:, None] + eps) + eps) /
                          ((counts[None, :] + eps) / (2 + eps) + eps))
        np.testing.assert_allclose(conditional_from_counts(counts, pairs, 2).numpy(), expected, rtol=1e-6)
        self.assertEqual(len(hashes), 2)
        with self.assertRaises(ValueError): conditional_from_counts(np.zeros(8), pairs, 2)
        with self.assertRaises(ValueError): count_training_slices(directory, ["../test"])
        with self.assertRaises(ValueError): count_training_slices(directory, ["case0005_slice000"] * 2)

    def test_fractional_and_nonfinite_raw_labels_rejected(self):
        directory = Path(self.temp.name)
        for value in (.5, float("nan"), float("inf"), 9.):
            np.savez(directory / "case0005_slice000.npz", label=np.array([[value]], dtype=np.float32))
            with self.assertRaises(ValueError): count_training_slices(directory, ["case0005_slice000"])

    def test_unverified_mapping_and_changed_manifest_fail_before_encoding(self):
        with self.assertRaises(ValueError):
            load_label_mapping(ROOT / "configs/synapse/asee_label_map.example.json")
        mapping = json.loads((ROOT / "configs/synapse/asee_label_map.example.json").read_text())
        mapping.update(verified=True, source="synthetic fixture")
        path = Path(self.temp.name) / "mapping.json"
        path.write_text(json.dumps(mapping))
        (Path(self.temp.name) / "train.txt").write_text("case0005_slice000\n")
        (Path(self.temp.name) / "test_vol.txt").write_text("case0008\n")
        args = SimpleNamespace(output=str(Path(self.temp.name) / "prior.pt"), list_dir=self.temp.name,
                               train_dir=self.temp.name, label_map_json=str(path), encoder_path="absent")
        with self.assertRaisesRegex(ValueError, "unchanged fixed split"): build_prior(args)
        self.assertFalse(Path(args.output).exists())

    def test_innovation_configs_preserve_all_unstudied_settings(self):
        baseline = yaml.safe_load((ROOT / "configs/synapse/vit_query_mul_scale_fusion.yaml").read_text())
        keys = {"prior_path", "queries_per_class", "patch_update", "freeze_text_projection",
                "enhance_patch", "enhance_query", "use_text", "use_probability"}
        for name in ("asee_plan", "asee_reference", "asee_plan_frozen_text"):
            config = yaml.safe_load((ROOT / f"configs/synapse/vit_query_mul_scale_fusion_{name}.yaml").read_text())
            self.assertTrue(config["trainer"].pop("default_root_dir").startswith("outputs/innovation1/"))
            self.assertTrue(config["trainer"]["callbacks"][0]["init_args"].pop("output_dir").startswith("outputs/innovation1/"))
            self.assertEqual(config["trainer"]["logger"]["init_args"].pop("save_dir"), "outputs/innovation1")
            config["trainer"]["logger"]["init_args"]["name"] = baseline["trainer"]["logger"]["init_args"]["name"]
            network = config["model"]["init_args"]["network"]
            network["class_path"] = baseline["model"]["init_args"]["network"]["class_path"]
            for key in keys: network["init_args"].pop(key)
            self.assertEqual(config, baseline)

    def test_original_hungarian_loss_backpropagates_through_integrated_model(self):
        from training.mask_classification_semantic import MaskClassificationSemantic
        network = self.network()
        module = MaskClassificationSemantic(network=network, img_size=(32, 32), num_classes=9,
                                            attn_mask_annealing_enabled=False, num_points=32)
        masks, classes = network(torch.rand(2, 3, 32, 32))
        foreground = torch.zeros(32, 32, dtype=torch.bool)
        foreground[8:24, 8:24] = True
        targets = [{"labels": torch.tensor([0, 6]), "masks": torch.stack((~foreground, foreground))} for _ in range(2)]
        losses = module.criterion(masks_queries_logits=masks[0], class_queries_logits=classes[0], targets=targets)
        loss = module.criterion.loss_total(losses, lambda *a, **k: None)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for param in (network.asee.text_proj[0].weight, network.asee.prob_proj[0].weight,
                      network.query_embed, network.encoder.backbone.blocks[-1].attn.qkv.weight):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum(), 0)

    def test_checkpoint_rejects_changed_asee_configuration(self):
        state = self.network().state_dict()
        for kwargs in ({"patch_update": "reference_gated"}, {"enhance_patch": False},
                       {"enhance_query": False}, {"use_text": False},
                       {"use_probability": False}, {"freeze_text_projection": True}):
            with self.subTest(kwargs=kwargs):
                clone = self.network(**kwargs)
                with self.assertRaisesRegex(ValueError, "ASEE configuration"):
                    clone.load_state_dict(copy.deepcopy(state), strict=False)

    def test_asee_weights_without_provenance_rejected_even_non_strict(self):
        state = self.network().state_dict()
        del state["_extra_state"]
        with self.assertRaisesRegex(ValueError, "ASEE.*provenance"):
            self.network().load_state_dict(state, strict=False)

    def test_prior_relative_path_resolves_from_project(self):
        with mock.patch("models.asee.PROJECT_ROOT", Path(self.temp.name), create=True):
            prior, digest = load_prior(self.path.name)
        torch.testing.assert_close(prior["text_features"], self.prior["text_features"])
        self.assertEqual(digest, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_prior_digest_matches_loaded_bytes_when_file_replaced(self):
        expected = hashlib.sha256(self.path.read_bytes()).hexdigest()
        real_load = torch.load
        def replace_after_load(*args, **kwargs):
            value = real_load(*args, **kwargs)
            self.path.write_bytes(b"replaced after deserialization")
            return value
        with mock.patch("models.asee.torch.load", side_effect=replace_after_load):
            prior, digest = load_prior(self.path)
        self.assertEqual(digest, expected)
        torch.testing.assert_close(prior["text_features"], self.prior["text_features"])

    def test_impossible_or_nonfinite_cooccurrence_rejected(self):
        counts = np.array([2., 1.]); pairs = np.array([[2., 1.], [1., 1.]])
        bad_tables = [(counts, pairs, 2, float("nan")),
                      (np.array([float("nan"), 1]), pairs, 2, 1e-5),
                      (np.array([3., 1]), pairs, 2, 1e-5),
                      (counts, np.array([[2., 0.], [1., 1.]]), 2, 1e-5),
                      (counts, np.array([[1., 1.], [1., 1.]]), 2, 1e-5),
                      (counts, pairs, 2.5, 1e-5)]
        for args in bad_tables:
            with self.subTest(args=args), self.assertRaises(ValueError):
                conditional_from_counts(*args)

    def test_text_encoding_bounds_tokens_and_detects_source_changes(self):
        from scripts.build_synapse_asee_prior import encode_text
        directory = Path(self.temp.name)/"encoder"; directory.mkdir()
        config_file = directory/"config.json"; config_file.write_text("fixture")
        tokenizer = mock.Mock(return_value={"input_ids": torch.ones(8, 20, dtype=torch.long)})
        encoder = mock.Mock()
        encoder.cpu.return_value = encoder; encoder.eval.return_value = encoder
        encoder.config = SimpleNamespace(max_position_embeddings=512)
        encoder.return_value = SimpleNamespace(last_hidden_state=torch.ones(8, 20, 768))
        transformer = SimpleNamespace(AutoModel=mock.Mock(), AutoTokenizer=mock.Mock())
        transformer.AutoModel.from_pretrained.return_value = encoder
        transformer.AutoTokenizer.from_pretrained.return_value = tokenizer
        with mock.patch.dict("sys.modules", {"transformers": transformer}):
            features, _ = encode_text(["word "*700]*8, directory)
            self.assertEqual(tokenizer.call_args.kwargs.get("max_length"), 512)
            self.assertEqual(tuple(features.shape), (8, 768))
            def mutate(**kwargs):
                config_file.write_text("changed during encode")
                return SimpleNamespace(last_hidden_state=torch.ones(8, 20, 768))
            encoder.side_effect = mutate
            with self.assertRaisesRegex(RuntimeError, "encoder.*changed"):
                encode_text(["organ"]*8, directory)

    def test_cpu_mixed_precision_compiled_forward_and_backward(self):
        for variant in ("plan", "reference_gated"):
            with self.subTest(variant=variant):
                network = self.network(patch_update=variant).train()
                compiled = torch.compile(network, backend="eager", fullgraph=True)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    masks, classes = compiled(torch.rand(2, 3, 32, 32))
                    loss = masks[0].float().square().mean() + classes[0].float().square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for param in (network.asee.text_proj[0].weight, network.asee.prob_proj[0].weight):
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(torch.isfinite(param.grad).all())

    def test_innovation_keeps_original_empty_gt_dice_rule(self):
        from training.mask_classification_semantic import MaskClassificationSemantic
        pred = np.ones((2, 2, 2), dtype=np.uint8)
        gt = np.zeros_like(pred)
        self.assertEqual(MaskClassificationSemantic.calculate_metric_percase(pred, gt)[0], 1.)
        self.assertEqual(MaskClassificationSemantic.calculate_metric_percase(gt, gt)[0], 0.)

    def test_common_optimizer_assigns_asee_to_original_head_schedule(self):
        from training.mask_classification_semantic import MaskClassificationSemantic
        network = self.network()
        module = MaskClassificationSemantic(network=network, img_size=(32, 32), num_classes=9,
                                            attn_mask_annealing_enabled=False, lr=2e-4)
        module._trainer = SimpleNamespace(estimated_stepping_batches=30360)
        configured = module.configure_optimizers()
        optimizer = configured["optimizer"]
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        groups = [g for g in optimizer.param_groups if g["name"].startswith("network.asee.")]
        self.assertTrue(groups)
        self.assertTrue(all(g["initial_lr"] == 2e-4 and g["weight_decay"] == .05 for g in groups))
        self.assertEqual(configured["lr_scheduler"]["interval"], "step")


if __name__ == "__main__":
    unittest.main()
