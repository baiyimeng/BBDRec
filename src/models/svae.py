import torch
import torch.nn as nn


class SVAE(nn.Module):
    def __init__(self, args):
        super(SVAE, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.item_num = args.item_num
        self.max_len = args.max_len

        self.rnn_size = getattr(args, "svae_rnn_size", self.hidden_size)
        self.svae_hidden = getattr(args, "svae_hidden_size", self.hidden_size)
        self.latent_size = getattr(args, "svae_latent_size", self.hidden_size)

        self.item_embedding = nn.Embedding(
            self.item_num + 1, self.hidden_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(args.emb_dropout)

        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.rnn_size,
            batch_first=True,
            num_layers=1,
        )

        self.enc_linear = nn.Linear(self.rnn_size, self.svae_hidden)
        self.to_latent = nn.Linear(self.svae_hidden, 2 * self.latent_size)
        self.dec_linear1 = nn.Linear(self.latent_size, self.svae_hidden)
        self.dec_linear2 = nn.Linear(self.svae_hidden, self.hidden_size)

        self.tanh = nn.Tanh()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.zero_()

    def _encode(self, item_seq):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.emb_dropout(item_emb)
        rnn_out, _ = self.gru(item_emb)
        enc_hidden = self.tanh(self.enc_linear(rnn_out))
        return enc_hidden

    def _sample_latent(self, enc_hidden, train_flag):
        latent_params = self.to_latent(enc_hidden)
        mu = latent_params[..., : self.latent_size]
        log_sigma = latent_params[..., self.latent_size :]
        if train_flag:
            # ``log_sigma`` here is `log(sigma)`, matching the original SVAE
            # paper notation; the KL therefore uses `exp(log_sigma)` for the
            # variance term.
            sigma = torch.exp(log_sigma)
            noise = torch.randn_like(sigma)
            z = mu + sigma * noise
            kl = 0.5 * (-log_sigma + torch.exp(log_sigma) + mu**2 - 1.0)
            # average per-element across (B, L, latent) so the magnitude is
            # comparable to other models' scalar losses.
            kl_loss = kl.mean()
        else:
            z = mu
            kl_loss = torch.zeros(1, device=enc_hidden.device)
        return z, kl_loss

    def _decode(self, z):
        dec_hidden = self.tanh(self.dec_linear1(z))
        out_seq = self.dec_linear2(dec_hidden)
        return out_seq

    def forward(self, item_seq, tgt_seq, train_flag=True):
        enc_hidden = self._encode(item_seq)
        z, kl_loss = self._sample_latent(enc_hidden, train_flag)
        out_seq = self._decode(z)
        last_item = out_seq[:, -1, :]
        return out_seq, last_item, kl_loss

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
