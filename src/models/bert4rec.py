import torch
import torch.nn as nn

from models.components import TransformerEncoder


class BERT4Rec(nn.Module):
    def __init__(self, args):
        super(BERT4Rec, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.item_num = args.item_num
        self.max_len = args.max_len

        # Reserve item_num + 1 as the [MASK] token id.
        # Embedding indices: 0 -> padding, 1..item_num -> items,
        #                    item_num + 1 -> [MASK].
        self.mask_token = self.item_num + 1
        self.mask_ratio = getattr(args, "bert_mask_ratio", 0.2)
        self.num_blocks = getattr(args, "bert_num_blocks", 2)

        self.item_embedding = nn.Embedding(
            self.item_num + 2, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            args, num_blocks=self.num_blocks, is_causal=False
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.emb_dropout)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _reconstruct_input(self, item_seq):
        """Shift the sequence one step to the left and append [MASK] at end."""
        batch_size = item_seq.size(0)
        mask_col = torch.full(
            (batch_size, 1),
            self.mask_token,
            dtype=torch.long,
            device=item_seq.device,
        )
        return torch.cat([item_seq[:, 1:], mask_col], dim=1)

    def _random_mlm_masking(self, item_seq):
        """Replace a random subset of *real* tokens with [MASK]."""
        if self.mask_ratio <= 0.0:
            return item_seq
        prob = torch.full(item_seq.shape, self.mask_ratio, device=item_seq.device)
        # Never mask the final [MASK] position (left intact by the caller) and
        # never mask padding tokens.
        prob = prob.masked_fill(item_seq == 0, 0.0)
        prob[:, -1] = 0.0
        masked_indices = torch.bernoulli(prob).bool()
        out = item_seq.clone()
        out[masked_indices] = self.mask_token
        return out

    def _encode(self, item_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_emb = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # Real tokens (incl. the [MASK] token) attend bidirectionally; padding
        # positions are masked out.
        padding_mask = (item_seq > 0).float()
        out_seq = self.trm_encoder(input_emb, padding_mask)
        return out_seq

    def forward(self, item_seq, tgt_seq, train_flag=True):
        recon_seq = self._reconstruct_input(item_seq)
        if train_flag:
            recon_seq = self._random_mlm_masking(recon_seq)
        out_seq = self._encode(recon_seq)
        last_item = out_seq[:, -1, :]
        return out_seq, last_item, torch.zeros(1, device=item_seq.device)

    def _real_item_embedding(self):
        # Exclude the [MASK] token row when scoring real items.
        return self.item_embedding.weight[: self.item_num + 1]

    def calculate_loss(self, seq_output, tgt_seq):
        index = tgt_seq > 0
        seq_output = seq_output[index]
        tgt_seq = tgt_seq[index]
        logits = torch.matmul(seq_output, self._real_item_embedding().t())
        loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), tgt_seq.reshape(-1))
        return loss

    def calculate_score(self, item):
        scores = torch.matmul(
            item.reshape(-1, item.shape[-1]), self._real_item_embedding().t()
        )
        return scores
