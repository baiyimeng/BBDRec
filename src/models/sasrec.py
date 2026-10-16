import torch
from torch import nn

from models.components import TransformerEncoder


class SASRec(nn.Module):
    def __init__(self, args):
        super(SASRec, self).__init__()
        self.args = args

        self.hidden_size = args.hidden_size
        self.item_embedding = nn.Embedding(
            args.item_num + 1, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(args.max_len, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            args, num_blocks=2
        )

        self.dropout = nn.Dropout(args.emb_dropout)
        self.apply(self._init_weights)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.1)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def embedding_layer(self, item_seq):
        position_ids = (
            torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
            .unsqueeze(0)
            .expand_as(item_seq)
        )
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        return item_emb, position_embedding

    def forward(self, item_seq, tgt_seq, train_flag=True):
        item_emb, position_emb = self.embedding_layer(item_seq)
        if self.args.model == "pretrain":
            input_emb = item_emb
        else:
            input_emb = item_emb + position_emb
        input_emb = self.dropout(input_emb)
        mask_seq = (item_seq > 0).float()
        output_seq = self.trm_encoder(input_emb, mask_seq)
        last_item = output_seq[:, -1, :]
        return output_seq, last_item, torch.zeros(1, device=item_emb.device)

    def calculate_loss(self, seq_output, tgt_seq):
        index = tgt_seq > 0
        seq_output = seq_output[index]
        tgt_seq = tgt_seq[index]
        logits = torch.matmul(seq_output, self.item_embedding.weight.t())
        loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), tgt_seq.reshape(-1))
        return loss

    def calculate_score(self, item):
        scores = torch.matmul(
            item.reshape(-1, item.shape[-1]), self.item_embedding.weight.t()
        )
        return scores
