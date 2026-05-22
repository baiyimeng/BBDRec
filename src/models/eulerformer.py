import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as fn


def _euler_trans(tensor):
    """Pairwise Euler (polar) decomposition: (lambda, theta)."""
    r = tensor[..., ::2]
    p = tensor[..., 1::2]
    lam = torch.sqrt(r * r + p * p + 1e-12)
    theta = torch.atan2(p, r)
    return lam, theta


class _EulerRotary(nn.Module):
    """Apply a learnable Euler-style rotary embedding to a hidden vector."""

    def __init__(self, max_len, hidden_size, init_factor=1.0, use_bias=False):
        super().__init__()
        if use_bias:
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("b", torch.zeros(1))
        self.delta = nn.Parameter(torch.ones(1) * init_factor)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
        ids = torch.arange(0, hidden_size // 2, dtype=torch.float)
        # Standard sinusoidal frequencies, shared by sin/cos pairs.
        freqs = torch.pow(10000.0, -2.0 * ids / hidden_size)
        alpha = position * freqs  # [max_len, hidden_size // 2]
        # Make ``alpha`` learnable to match the reference implementation.
        self.alpha = nn.Parameter(alpha)

    def forward(self, v, is_query=False):
        # v: [B, L, H] or [B, h, L, d]; we operate on the last dim.
        lam, theta = _euler_trans(v)  # both shaped (..., L, H/2)
        L = v.size(-2)
        alpha = self.alpha[:L].to(theta)
        # broadcast alpha across batch / heads
        while alpha.dim() < theta.dim():
            alpha = alpha.unsqueeze(0)
        theta = theta * self.delta + alpha
        if is_query:
            theta = theta + self.b
        r_new = lam * torch.cos(theta)
        p_new = lam * torch.sin(theta)
        # interleave back so original even/odd channel layout is preserved
        out = torch.stack([r_new, p_new], dim=-1).reshape(*v.shape)
        return out


class _EulerMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        max_len,
        is_causal,
        euler_init,
        euler_bias,
        tep,
        lamb,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})."
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        self.is_causal = is_causal

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        self.euler = _EulerRotary(
            max_len, hidden_size, init_factor=euler_init, use_bias=euler_bias
        )

        # Auxiliary "angle consistency" loss components.
        self.aux_dropout = nn.Dropout(p=0.5)
        self.aux_w = nn.Parameter(torch.ones(hidden_size // 2))
        self.tep = tep
        self.lamb = lamb
        self.aux_loss = torch.zeros(1)

    def _split_heads(self, x):
        new_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def _angle_consistency_loss(self, q, k):
        _, pq = _euler_trans(q)  # [B, L, H/2]
        _, pk = _euler_trans(k)

        def _cal(a, b):
            cos_a = torch.cos(a) * self.aux_w
            sin_a = torch.sin(a) * self.aux_w
            cos_b = torch.cos(b)
            sin_b = torch.sin(b)
            denom = cos_a @ cos_b.transpose(-2, -1) + sin_a @ sin_b.transpose(-2, -1)
            denom = denom / self.tep
            numerator = torch.diagonal(torch.exp(denom), dim1=-1, dim2=-2)
            denominator = torch.sum(torch.exp(denom), dim=-1) + 1e-5
            return torch.mean(-torch.log(numerator / denominator)) * self.lamb

        return _cal(pq, self.aux_dropout(pq)) + _cal(pk, self.aux_dropout(pk))

    def forward(self, input_tensor, additive_mask):
        q = self.query(input_tensor)
        k = self.key(input_tensor)
        v = self.value(input_tensor)

        # Apply Euler rotary on the flat hidden representation.
        q = self.euler(q, is_query=True)
        k = self.euler(k, is_query=False)

        self.aux_loss = self._angle_consistency_loss(q, k)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_attention_head_size
        scores = scores + additive_mask
        probs = self.attn_dropout(self.softmax(scores))

        context = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:-2], self.all_head_size)
        out = self.dense(context)
        out = self.out_dropout(out)
        out = self.LayerNorm(out + input_tensor)
        return out


def _gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _swish(x):
    return x * torch.sigmoid(x)


_ACT2FN = {
    "gelu": _gelu,
    "relu": fn.relu,
    "swish": _swish,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


class _FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = _ACT2FN[hidden_act]
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        h = self.dense_1(x)
        h = self.intermediate_act_fn(h)
        h = self.dense_2(h)
        h = self.dropout(h)
        h = self.LayerNorm(h + x)
        return h


class _EulerTransformerLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        max_len,
        is_causal,
        euler_init,
        euler_bias,
        tep,
        lamb,
    ):
        super().__init__()
        self.multi_head_attention = _EulerMultiHeadAttention(
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
            max_len,
            is_causal,
            euler_init,
            euler_bias,
            tep,
            lamb,
        )
        self.feed_forward = _FeedForward(
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, additive_mask):
        attention_output = self.multi_head_attention(hidden_states, additive_mask)
        return self.feed_forward(attention_output)

    @property
    def aux_loss(self):
        return self.multi_head_attention.aux_loss


class _EulerTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers,
        n_heads,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        max_len,
        is_causal,
        euler_init,
        euler_bias,
        tep,
        lamb,
    ):
        super().__init__()
        layer = _EulerTransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
            max_len,
            is_causal,
            euler_init,
            euler_bias,
            tep,
            lamb,
        )
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(n_layers)]
        )

    def forward(self, hidden_states, additive_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, additive_mask)
        return hidden_states

    @property
    def aux_loss(self):
        return sum(layer_module.aux_loss for layer_module in self.layer)


class EulerFormer(nn.Module):
    def __init__(self, args):
        super(EulerFormer, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.item_num = args.item_num
        self.max_len = args.max_len

        self.n_layers = getattr(args, "euler_n_layers", 2)
        self.n_heads = getattr(args, "euler_n_heads", 2)
        self.inner_size = getattr(args, "euler_inner_size", self.hidden_size * 4)
        self.euler_init = getattr(args, "euler_init_factor", 1.0)
        self.euler_bias = getattr(args, "euler_bias", False)
        self.tep = getattr(args, "euler_tep", 1.0)
        self.lamb = getattr(args, "euler_lamb", 1e-5)
        # EulerFormer's reference implementation is autoregressive (SASRec-style).
        self.is_causal = getattr(args, "is_causal", True)

        self.item_embedding = nn.Embedding(
            self.item_num + 1, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_size)
        # Per-position rotary correction applied to the input embedding.
        self.rotary_embedding = nn.Embedding(self.max_len, self.hidden_size // 2)
        nn.init.zeros_(self.rotary_embedding.weight)

        self.trm_encoder = _EulerTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=args.dropout,
            attn_dropout_prob=args.dropout,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            max_len=self.max_len,
            is_causal=self.is_causal,
            euler_init=self.euler_init,
            euler_bias=self.euler_bias,
            tep=self.tep,
            lamb=self.lamb,
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

    def _attention_mask(self, item_seq):
        attn_mask = (item_seq > 0).long().unsqueeze(1).unsqueeze(2)
        if self.is_causal:
            max_len = item_seq.size(-1)
            causal = torch.tril(
                torch.ones((1, 1, max_len, max_len), device=item_seq.device).long()
            )
            attn_mask = attn_mask * causal
        attn_mask = attn_mask.to(dtype=next(self.parameters()).dtype)
        return (1.0 - attn_mask) * -10000.0

    def forward(self, item_seq, tgt_seq, train_flag=True):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_emb = self.position_embedding(position_ids)
        rotary_emb = self.rotary_embedding(position_ids)

        input_emb = self.item_embedding(item_seq) + position_emb
        shape = input_emb.shape
        lam, theta = _euler_trans(input_emb)
        input_emb = torch.stack(
            [lam * torch.cos(theta + rotary_emb), lam * torch.sin(theta + rotary_emb)],
            dim=-1,
        ).reshape(*shape)

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        additive_mask = self._attention_mask(item_seq)
        out_seq = self.trm_encoder(input_emb, additive_mask)
        last_item = out_seq[:, -1, :]
        aux_loss = self.trm_encoder.aux_loss
        if not isinstance(aux_loss, torch.Tensor):
            aux_loss = torch.tensor(aux_loss, device=item_seq.device)
        return out_seq, last_item, aux_loss

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
