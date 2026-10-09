"""Small CPU correctness checks; no benchmark training or quality reproduction.

Run: python -m unittest discover -s tests -v
"""
import contextlib
import copy
import io
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from argparse import Namespace
from core.trainer import choose_model, model_train
from utils.data import Data_Train, Data_Val, Data_Test, ValDataset, TestDataset

BASELINES = ["sasrec", "gru4rec", "bert4rec", "lightsans", "core", "fearec",
             "eulerformer", "svae", "diffurec", "dreamrec", "sdifrec"]


def args_for(name, **overrides):
    with (SRC / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    config.update(model=name, device="cpu", hidden_size=32, max_len=12,
                  item_num=32, batch_size=2, diffusion_steps=4,
                  pretrained=False, freeze_emb=False, epochs=1,
                  description="runtime_check", dataset="tiny_runtime_check")
    config.update(overrides)
    return Namespace(**config)


def loaders(args):
    train = [[1, 2, 3], [2, 4, 6, 8, 10], list(range(1, 13))]
    val = [[4], [12], [13]]
    test = [[5], [14], [15]]
    return (Data_Train(train, args).get_pytorch_dataloaders(),
            Data_Val(train, val, args).get_pytorch_dataloaders(),
            Data_Test(train, val, test, args).get_pytorch_dataloaders())


class BaselineRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def check_model(self, name, **overrides):
        torch.manual_seed(7)
        args = args_for(name, **overrides)
        model = choose_model(args)
        train, val, test = loaders(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for batch in (next(iter(train)), train.dataset[len(train.dataset) - 1]):
            hist, target = batch
            if hist.dim() == 1:
                hist, target = hist.unsqueeze(0), target.unsqueeze(0)
            model.train()
            optimizer.zero_grad()
            out, last, aux = model(hist, target, train_flag=True)
            ce = model.calculate_loss(out, target)
            loss = aux if name == "dreamrec" else ce
            if name == "svae":
                loss = ce + 0.2 * aux
            elif name == "eulerformer":
                loss = ce + aux
            self.assertTrue(torch.isfinite(loss).all(), (name, "loss"))
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), (name, "gradients"))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for loader in (val, test):
                for hist, target in loader:
                    _, last, _ = model(hist, target, train_flag=False)
                    self.assertEqual(last.shape, (hist.shape[0], args.hidden_size))
                    scores = model.calculate_score(last)
                    self.assertEqual(scores.shape, (hist.shape[0], args.item_num + 1))
                    self.assertTrue(torch.isfinite(scores).all(), (name, "scores"))

    def test_all_baselines_train_backward_and_infer(self):
        for name in BASELINES:
            with self.subTest(model=name):
                self.check_model(name)

    def test_default_model_dimensions(self):
        for name in BASELINES:
            with self.subTest(model=name):
                self.check_model(name, hidden_size=128, max_len=50, diffusion_steps=32)

    def test_classifier_free_guidance_branches(self):
        for name in ("dreamrec", "diffurec"):
            with self.subTest(model=name):
                self.check_model(name, cfg_scale=2.0)

    def test_sdifrec_shared_timestep_branch(self):
        self.check_model("sdifrec", independent=False)

    def test_fearec_short_sequences(self):
        self.check_model("fearec", max_len=4)

    def test_eulerformer_auxiliary_loss_large_logits(self):
        model = choose_model(args_for("eulerformer", hidden_size=128))
        for layer in model.trm_encoder.layer:
            with torch.no_grad():
                layer.multi_head_attention.aux_w.fill_(20)
        hist, target = next(iter(loaders(model.args)[0]))
        out, _, aux = model(hist, target)
        loss = model.calculate_loss(out, target) + aux
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None))

    def test_eulerformer_loss_preserves_finite_objective(self):
        model = choose_model(args_for("eulerformer"))
        attention = model.trm_encoder.layer[0].multi_head_attention
        attention.eval()
        q, k = torch.randn(2, 4, 32), torch.randn(2, 4, 32)
        from models.eulerformer import _euler_trans
        expected = 0.0
        for tensor in (q, k):
            _, angle = _euler_trans(tensor)
            logits = ((torch.cos(angle) * attention.aux_w) @ torch.cos(angle).transpose(-2, -1)
                      + (torch.sin(angle) * attention.aux_w) @ torch.sin(angle).transpose(-2, -1)) / attention.tep
            numerator = torch.exp(logits).diagonal(dim1=-1, dim2=-2)
            denominator = torch.exp(logits).sum(dim=-1) + 1e-5
            expected = expected + (-torch.log(numerator / denominator)).mean() * attention.lamb
        torch.testing.assert_close(attention._angle_consistency_loss(q, k), expected,
                                   atol=1e-10, rtol=1e-4)

    def test_shared_trainer_still_supports_bbdrec(self):
        with tempfile.TemporaryDirectory() as folder:
            previous = os.getcwd()
            try:
                os.chdir(folder)
                args = args_for("bbdrec")
                checkpoint = Path("saved/pretrain") / args.dataset / "pretrain.pth"
                checkpoint.parent.mkdir(parents=True)
                torch.save({"item_embedding.weight": torch.randn(args.item_num + 1, args.hidden_size)}, checkpoint)
                model = choose_model(args)
                with contextlib.redirect_stdout(io.StringIO()):
                    best, results = model_train(model, *loaders(args), args,
                                               logging.getLogger("test"), "test")
                self.assertIsNotNone(best)
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in results.values()))
            finally:
                os.chdir(previous)

    def test_single_position_evaluation_data(self):
        for seq in ([1], [1, 2, 3]):
            for dataset in (ValDataset([seq], [[4]], 1),
                            TestDataset([seq], [[4]], [[5]], 1)):
                hist, target = dataset[0]
                self.assertEqual(hist.shape, (1,))
                self.assertEqual(target.shape, (1,))
                self.assertEqual(target.item(), 4 if isinstance(dataset, ValDataset) else 5)

    def test_short_training_and_zero_validation_can_save(self):
        # One tiny epoch must still select a checkpoint even when every metric is zero.
        for name in BASELINES:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as folder:
                args = args_for(name)
                model = choose_model(args)
                data = loaders(args)
                zero_metrics = {f"{metric}@{k}": 0.0 for k in args.metric_ks
                                for metric in ("HR", "NDCG")}
                previous = os.getcwd()
                try:
                    os.chdir(folder)
                    with patch("core.trainer.hrs_and_ndcgs_k", return_value=zero_metrics), \
                         contextlib.redirect_stdout(io.StringIO()):
                        best, results = model_train(model, *data, args, logging.getLogger("test"), "test")
                    self.assertIsNotNone(best)
                    checkpoint = Path("saved") / name / args.dataset / (args.description + ".pth")
                    restored = choose_model(copy.deepcopy(args))
                    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                    self.assertEqual(results, zero_metrics)
                finally:
                    os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
