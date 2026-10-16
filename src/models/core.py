import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import TransformerEncoder


class _CoreTransNet(nn.Module):
    """Causal transformer that outputs per-position attention weights."""

    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.position_embedding = nn.Embedding(args.max_len, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            args,
            num_blocks=getattr(args, "core_num_blocks", 2),
            is_causal=True,
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.emb_dropout)
        self.fn = nn.Linear(self.hidden_size, 1)

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        padding_mask = (item_seq > 0).float()
        output = self.trm_encoder(input_emb, padding_mask)

        alpha = self.fn(output)
        # Renormalise so only real tokens contribute.
        alpha = alpha.masked_fill(~mask.unsqueeze(-1), -9e15)
        alpha = torch.softmax(alpha, dim=1)
        return alpha


class CORE(nn.Module):
    def __init__(self, args):
        super(CORE, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.item_num = args.item_num
        self.max_len = args.max_len

        self.dnn_type = getattr(args, "core_dnn_type", "trm")
        self.temperature = getattr(args, "core_temperature", 0.07)
        self.sess_dropout = nn.Dropout(
            getattr(args, "core_sess_dropout", args.emb_dropout)
        )
        self.item_dropout = nn.Dropout(
            getattr(args, "core_item_dropout", args.dropout)
        )

        self.item_embedding = nn.Embedding(
            self.item_num + 1, self.hidden_size, padding_idx=0
        )

        if self.dnn_type == "trm":
            self.net = _CoreTransNet(args)
        elif self.dnn_type == "ave":
            self.net = None
        else:
            raise ValueError(
                f"core_dnn_type should be either 'trm' or 'ave', got '{self.dnn_type}'."
            )

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @staticmethod
    def _ave_net(item_seq, item_emb):
        mask = item_seq.gt(0)
        cnt = mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
        alpha = mask.to(torch.float) / cnt
        return alpha.unsqueeze(-1)

    def _encode(self, item_seq):
        x = self.item_embedding(item_seq)
        x = self.sess_dropout(x)
        if self.dnn_type == "trm":
            alpha = self.net(item_seq, x)
        else:
            alpha = self._ave_net(item_seq, x)
        seq_output = torch.sum(alpha * x, dim=1)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output

    def forward(self, item_seq, tgt_seq, train_flag=True):
        seq_output = self._encode(item_seq)  # [B, H]
        last_item = seq_output
        # Build a dense [B, L, H] out_seq with the pooled vector at the last
        # position so that ``calculate_loss`` can index ``tgt_seq > 0``.
        batch_size, seq_len = item_seq.shape
        out_seq = torch.zeros(
            batch_size,
            seq_len,
            self.hidden_size,
            device=item_seq.device,
            dtype=last_item.dtype,
        )
        out_seq[:, -1, :] = last_item
        return out_seq, last_item, torch.zeros(1, device=item_seq.device)

    def calculate_loss(self, seq_output, tgt_seq):
        index = tgt_seq > 0
        seq_repr = seq_output[index]
        tgt = tgt_seq[index]
        all_item_emb = self.item_embedding.weight
        all_item_emb = self.item_dropout(all_item_emb)
        all_item_emb = F.normalize(all_item_emb, dim=-1)
        logits = torch.matmul(seq_repr, all_item_emb.t()) / self.temperature
        loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        return loss

    def calculate_score(self, item):
        # ``item`` (i.e. ``last_item``) is already L2-normalised inside ``_encode``.
        all_item_emb = F.normalize(self.item_embedding.weight, dim=-1)
        scores = (
            torch.matmul(item.reshape(-1, item.shape[-1]), all_item_emb.t())
            / self.temperature
        )
        return scores
