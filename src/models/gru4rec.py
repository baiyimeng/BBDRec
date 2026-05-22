import torch
from torch import nn


class GRU4Rec(nn.Module):
    def __init__(self, args):
        super(GRU4Rec, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        # Configurable defaults; these can be added to config.yaml if desired.
        self.num_layers = getattr(args, "gru_num_layers", 1)

        self.item_embedding = nn.Embedding(
            args.item_num + 1, self.hidden_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(args.emb_dropout)
        self.gru_layers = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, item_seq, tgt_seq, train_flag=True):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.emb_dropout(item_emb)
        gru_output, _ = self.gru_layers(item_emb)
        out_seq = self.dense(gru_output)
        last_item = out_seq[:, -1, :]
        return out_seq, last_item, torch.zeros(1, device=item_seq.device)

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
