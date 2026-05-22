import copy
import datetime
import os
import pickle

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from core.metrics import hrs_and_ndcgs_k
from models.bert4rec import BERT4Rec
from models.core import CORE
from models.eulerformer import EulerFormer
from models.fearec import FEARec
from models.gru4rec import GRU4Rec
from models.lightsans import LightSANs
from models.model import DiffusionRecommender
from models.sasrec import SASRec
from models.svae import SVAE
from utils.data import Data_Test, Data_Train, Data_Val


def item_num_create(args):
    length = {
        "ml-100k": 1008,
        "yelp": 64669,
        "sports": 12301,
        "baby": 4731,
        "toys": 7309,
        "beauty": 6086,
    }
    args.item_num = length[args.dataset]
    return args


def optimizers(model, args):
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return opt


def choose_model(args):
    device = args.device
    if args.model in [
        "diffurec",
        "dreamrec",
        "bbdrec",
        "bbdrec-0",
        "bbdrec-1",
        "sdifrec",
    ]:
        if args.model in ["bbdrec"]:
            args.pretrained = True
            args.freeze_emb = True
        if args.model in ["bbdrec-1"]:
            args.pretrained = True
            args.freeze_emb = False
        if args.model == "diffurec":
            args.split_onebyone = True
            args.parallel_ag = False
            args.is_causal = False
        if args.model == "dreamrec":
            args.split_onebyone = True
            args.parallel_ag = False
        model = DiffusionRecommender(args)
    elif args.model in ["sasrec", "pretrain"]:
        model = SASRec(args)
    elif args.model == "gru4rec":
        model = GRU4Rec(args)
    elif args.model == "bert4rec":
        # BERT4Rec is bidirectional and uses a [MASK] token at the end of the
        # input to predict the next item; it does not produce a parallel target
        # at every position.
        args.parallel_ag = False
        args.split_onebyone = True
        args.is_causal = False
        model = BERT4Rec(args)
    elif args.model == "core":
        # CORE encodes a sequence into a single normalised vector; the loss
        # only applies at the last position.
        args.parallel_ag = False
        args.split_onebyone = True
        model = CORE(args)
    elif args.model == "lightsans":
        # LightSANs pools into a single representation via its low-rank
        # attention; train with single-target sequences.
        args.parallel_ag = False
        args.split_onebyone = True
        model = LightSANs(args)
    elif args.model == "fearec":
        # FEARec uses a causal hybrid attention (time + frequency); the SSL
        # auxiliary losses from the original paper are dropped here.
        args.is_causal = True
        model = FEARec(args)
    elif args.model == "svae":
        # SVAE produces a per-position latent + decoded hidden; train with the
        # parallel-AG paradigm so every step contributes a target.
        args.parallel_ag = True
        args.split_onebyone = False
        model = SVAE(args)
    elif args.model == "eulerformer":
        # EulerFormer is autoregressive (SASRec-style) with a learnable Euler
        # rotary embedding and an angle-consistency aux loss.
        args.is_causal = True
        model = EulerFormer(args)
    else:
        raise NotImplementedError("Model not implemented.")

    return model.to(device)


def load_data(args):
    path_data = "../datasets/data/" + args.dataset + "/dataset.pkl"
    with open(path_data, "rb") as f:
        data_raw = pickle.load(f)
    tra_data = Data_Train(data_raw["train"], args)
    val_data = Data_Val(data_raw["train"], data_raw["val"], args)
    test_data = Data_Test(data_raw["train"], data_raw["val"], data_raw["test"], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()

    return tra_data_loader, val_data_loader, test_data_loader


def model_train(
    model_joint,
    tra_data_loader,
    val_data_loader,
    test_data_loader,
    args,
    logger,
    train_time,
):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    torch.set_float32_matmul_precision("high")
    optimizer = optimizers(model_joint, args)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    best_metrics_dict = {}
    best_epoch = {}
    for k in args.metric_ks:
        best_metrics_dict[f"Best_HR@{k}"] = 0
        best_metrics_dict[f"Best_NDCG@{k}"] = 0
        best_epoch[f"Best_epoch_HR@{k}"] = 0
        best_epoch[f"Best_epoch_NDCG@{k}"] = 0
    bad_count = 0
    best_model = None
    for epoch_temp in range(epochs):
        model_joint.train()
        if epoch_temp == 5 and args.model in ["bbdrec"]:
            print("=" * 60)
            print(
                f"[Warm-up] Finished at epoch {epoch_temp}, unfreezing item embedding"
            )
            logger.info(
                f"[Warm-up] Finished at epoch {epoch_temp}, unfreezing item embedding"
            )
            print("=" * 60)
            model_joint.item_embedding.weight.requires_grad = True
        ce_losses = []
        diff_losses = []
        flag_update = 0
        pbr_train = tqdm(
            enumerate(tra_data_loader),
            desc="Epoch: {}".format(epoch_temp),
            leave=False,
            total=len(tra_data_loader),
            disable=True,
        )
        for index_temp, train_batch in pbr_train:
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()
            out_seq, last_item, diff_loss = model_joint(
                train_batch[0], train_batch[1], train_flag=True
            )

            ce_loss = model_joint.calculate_loss(out_seq, train_batch[1])
            if args.model == "bbdrec":
                losses = [ce_loss, args.loss_scale * diff_loss]
            elif args.model == "dreamrec":
                losses = [diff_loss]
            elif args.model == "svae":
                # SVAE: CE on the decoded representation + (annealed) KL term.
                losses = [ce_loss, getattr(args, "svae_anneal", 0.2) * diff_loss]
            elif args.model == "eulerformer":
                # EulerFormer: CE + angle-consistency loss (already scaled by
                # ``euler_lamb`` inside the model).
                losses = [ce_loss, diff_loss]
            else:
                losses = [ce_loss]

            total_loss = sum(losses)
            total_loss.backward()
            ce_losses.append(ce_loss.item())
            diff_losses.append(diff_loss.item())
            optimizer.step()
            pbr_train.set_postfix_str(f"loss={ce_losses[-1]:.3f}")
        avg_ce_loss = sum(ce_losses) / len(ce_losses)
        avg_diff_loss = sum(diff_losses) / len(diff_losses)
        print(
            f"[Train] Epoch {epoch_temp:3d} | CE Loss: {avg_ce_loss:.4f} | Diff Loss: {avg_diff_loss:.4f}"
        )
        logger.info(
            f"[Train] Epoch {epoch_temp:3d} | CE Loss: {avg_ce_loss:.4f} | Diff Loss: {avg_diff_loss:.4f}"
        )
        lr_scheduler.step()
        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            metrics_dict = {}
            for k in args.metric_ks:
                metrics_dict[f"HR@{k}"] = []
                metrics_dict[f"NDCG@{k}"] = []
            model_joint.eval()
            with torch.no_grad():
                for val_batch in tqdm(
                    val_data_loader,
                    leave=False,
                    desc="Denoising..., Epoch: {}".format(epoch_temp),
                    disable=True,
                ):
                    val_batch = [x.to(device) for x in val_batch]
                    out_seq, last_item, _ = model_joint(
                        val_batch[0], val_batch[1], train_flag=False
                    )

                    scores_rec_diffu = model_joint.calculate_score(last_item)
                    metrics = hrs_and_ndcgs_k(
                        scores_rec_diffu, val_batch[1][:, -1:], metric_ks
                    )
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)

            for key_temp, values_temp in metrics_dict.items():
                values_mean = float(round(np.mean(values_temp) * 100, 4))
                if values_mean > best_metrics_dict["Best_" + key_temp]:
                    flag_update = 1
                    bad_count = 0
                    best_metrics_dict["Best_" + key_temp] = values_mean
                    best_epoch["Best_epoch_" + key_temp] = epoch_temp
                    best_epoch_temp = epoch_temp

            if flag_update == 0:
                bad_count += 1
                print(f"[Val] No improvement | Patience: {bad_count}/{args.patience}")
            else:
                print(f"[Val] *** New best model found at epoch {epoch_temp} ***")
                print(f"[Val] Best metrics: {best_metrics_dict}")
                print(f"[Val] Best epochs: {best_epoch}")
                logger.info(f"[Val] *** New best model found at epoch {epoch_temp} ***")
                logger.info(f"[Val] Best metrics: {best_metrics_dict}")
                logger.info(f"[Val] Best epochs: {best_epoch}")
                best_model = copy.deepcopy(model_joint)

            # Test set evaluation
            print(f"[Test] Evaluating at epoch {epoch_temp}...")
            logger.info(f"[Test] Evaluating at epoch {epoch_temp}...")
            test_metrics_dict = {}
            for k in args.metric_ks:
                test_metrics_dict[f"HR@{k}"] = []
                test_metrics_dict[f"NDCG@{k}"] = []

            with torch.no_grad():
                for test_batch in tqdm(
                    test_data_loader, leave=False, desc="Testing", disable=True
                ):
                    test_batch = [x.to(device) for x in test_batch]
                    out_seq, last_item, _ = model_joint(
                        test_batch[0], test_batch[1], train_flag=False
                    )
                    scores_rec_diffu = model_joint.calculate_score(last_item)
                    metrics = hrs_and_ndcgs_k(
                        scores_rec_diffu, test_batch[1][:, -1:], metric_ks
                    )
                    for k, v in metrics.items():
                        test_metrics_dict[k].append(v)

            test_metrics_dict_mean = {}
            for key_temp, values_temp in test_metrics_dict.items():
                values_mean = float(round(np.mean(values_temp) * 100, 4))
                test_metrics_dict_mean[key_temp] = values_mean

            print(f"[Test] Epoch {epoch_temp:3d} | Results: {test_metrics_dict_mean}")
            logger.info(
                f"[Test] Epoch {epoch_temp:3d} | Results: {test_metrics_dict_mean}"
            )
            print("=" * 60)
            if bad_count >= args.patience:
                break
    # Save best model
    saved_dir = os.path.join("saved", args.model, args.dataset)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    if args.model == "pretrain":
        output_path = os.path.join(saved_dir, "pretrain.pth")
    else:
        # output_path = os.path.join(
        #     saved_dir, str(train_time) + args.description + ".pth"
        # )
        output_path = os.path.join(saved_dir, args.description + ".pth")
    torch.save(best_model.state_dict(), str(output_path))
    print(f"[Save] Model saved to {output_path}")
    logger.info(f"[Save] Model saved to {output_path}")

    # Final test evaluation
    print("\n" + "=" * 60)
    print("[Configuration]")
    print("=" * 60)
    print(args)
    logger.info(f"[Configuration] {args}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("[Best Validation Results]")
    print("=" * 60)
    print(f"Best metrics: {best_metrics_dict}")
    print(f"Best epochs: {best_epoch}")
    logger.info(f"[Best Validation Results] {best_metrics_dict}")
    logger.info(f"[Best Validation Epochs] {best_epoch}")

    print("\n" + "=" * 60)
    print("[Final Test] Starting final evaluation on test set")
    print(f"[Final Test] Time: {datetime.datetime.now()}")
    logger.info("=" * 60)
    logger.info("[Final Test] Starting final evaluation on test set")
    logger.info(f"[Final Test] Time: {datetime.datetime.now()}")
    model_joint.eval()
    with torch.no_grad():
        test_metrics_dict = {}
        for k in args.metric_ks:
            test_metrics_dict[f"HR@{k}"] = []
            test_metrics_dict[f"NDCG@{k}"] = []
        test_metrics_dict_mean = {}
        for test_batch in tqdm(test_data_loader, leave=False, disable=True):
            test_batch = [x.to(device) for x in test_batch]
            out_seq, last_item, _ = best_model(
                test_batch[0], test_batch[1], train_flag=False
            )
            scores_rec_diffu = best_model.calculate_score(last_item)

            metrics = hrs_and_ndcgs_k(
                scores_rec_diffu, test_batch[1][:, -1:], metric_ks
            )
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)

    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = float(round(np.mean(values_temp) * 100, 4))
        test_metrics_dict_mean[key_temp] = values_mean

    print("\n" + "=" * 60)
    print("[Final Test Results]")
    print("=" * 60)
    print(f"Test: {test_metrics_dict_mean}")
    logger.info(f"[Final Test Results] {test_metrics_dict_mean}")

    return best_model, test_metrics_dict_mean
