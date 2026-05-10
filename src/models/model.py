import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bbdrec import BBDRec
from models.diffurec import DiffuRec
from models.dreamrec import DreamRec
from models.sdifrec import SdifRec


class DiffusionRecommender(nn.Module):
    """Diffusion-based sequential recommendation model."""

    def __init__(self, args):
        super(DiffusionRecommender, self).__init__()
        self.emb_dim = args.hidden_size
        self.args = args
        self.item_num = args.item_num
        self.item_embedding = self.embed_item(pretrained=args.pretrained)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = create_model_diffu(args)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=0)
        self.geodesic = args.geodesic

    def load_pretrained_emb_weight(self):
        path = os.path.join("saved", "pretrain", self.args.dataset, "pretrain.pth")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        pretrained_emb_weight = saved["item_embedding.weight"]
        return pretrained_emb_weight

    def embed_item(self, pretrained=False):
        if pretrained:
            embedding = nn.Embedding.from_pretrained(
                self.load_pretrained_emb_weight(),
                padding_idx=0,
                freeze=self.args.freeze_emb,
            )
        else:
            embedding = nn.Embedding(self.item_num + 1, self.emb_dim, padding_idx=0)
        return embedding

    def calculate_loss(self, out_seq, labels):
        index = labels > 0
        out_seq = out_seq[index]
        labels = labels[index]
        scores = torch.matmul(out_seq, self.item_embedding.weight.t())
        loss = self.loss_ce(scores.reshape(-1, scores.shape[-1]), labels.reshape(-1))
        return loss

    def calculate_score(self, item):
        scores = torch.matmul(
            item.reshape(-1, item.shape[-1]), self.item_embedding.weight.t()
        )
        return scores

    def forward(self, sequence, tag, train_flag=True):
        item_embeddings = self.item_embedding(sequence)
        tag_embeddings = self.item_embedding(tag)
        if self.geodesic:
            tag_embeddings = F.normalize(tag_embeddings, p=2, dim=-1)
        item_embeddings = self.embed_dropout(item_embeddings)

        mask_seq = (sequence > 0).float()
        mask_tag = (tag > 0).float().view(tag.shape[0], -1)

        if train_flag:
            out_seq, diff_loss = self.diffu(
                item_embeddings, tag_embeddings, mask_seq, mask_tag
            )
            last_item = out_seq[:, -1, :]
        else:
            out_seq = self.diffu.denoise_sample(
                item_embeddings, tag_embeddings, mask_seq, mask_tag
            )
            last_item = out_seq[:, -1, :]
            diff_loss = None
        return out_seq, last_item, diff_loss


def create_model_diffu(args):
    if args.model == "diffurec":
        return DiffuRec(args)
    elif args.model == "dreamrec":
        return DreamRec(args)
    elif args.model == "sdifrec":
        return SdifRec(args)
    elif args.model in ["bbdrec", "bbdrec-0", "bbdrec-1"]:
        return BBDRec(args)
    else:
        raise ValueError(f"Unknown diffusion model: {args.model}")
