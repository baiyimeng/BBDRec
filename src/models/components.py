import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_square_subsequent_mask(sz: int, device):
    """Generate a square mask for the sequence."""
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=torch.float32, device=device),
        diagonal=1,
    )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)


class SublayerConnection(nn.Module):
    """Sandwich residual: PostNorm( x + Dropout( Sublayer( PreNorm(x) ) ) )."""

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Sandwich norm: norm2(x + dropout(sublayer(norm1(x))))."""
        return self.norm2(x + self.dropout(sublayer(self.norm1(x))))


class PositionwiseFeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_gate = nn.Linear(hidden_size, hidden_size * 3)
        self.w_up = nn.Linear(hidden_size, hidden_size * 3)
        self.w_down = nn.Linear(hidden_size * 3, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_gate.weight)
        nn.init.xavier_normal_(self.w_up.weight)
        nn.init.xavier_normal_(self.w_down.weight)

    def forward(self, hidden):
        gate = self.w_gate(hidden)
        up = self.w_up(hidden)
        return self.w_down(self.dropout(F.silu(gate) * up))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(3)]
        )
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.q_norm = RMSNorm(self.size_head)
        self.k_norm = RMSNorm(self.size_head)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)
        for l in self.linear_layers:
            nn.init.xavier_normal_(l.weight)

    def forward(self, q, k, v, padding_mask=None, is_causal=False):
        batch_size = q.shape[0]
        q, k, v = [
            l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
            for l, x in zip(self.linear_layers, (q, k, v))
        ]
        q = self.q_norm(q)
        k = self.k_norm(k)
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if padding_mask is not None:
            padding_mask = padding_mask.view(batch_size, 1, -1, 1).repeat(
                [1, corr.shape[1], 1, corr.shape[-1]]
            )
            corr = corr.masked_fill(padding_mask == 0, -1e9)
        if is_causal:
            causal_mask = generate_square_subsequent_mask(
                corr.shape[-1], device=corr.device
            )
            corr += (
                causal_mask.unsqueeze(0)
                .unsqueeze(0)
                .repeat([corr.shape[0], corr.shape[1], 1, 1])
            )
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(
            hidden.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.size_head)
        )
        return hidden


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout, is_causal):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadedAttention(
            heads=attn_heads, hidden_size=hidden_size, dropout=dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            hidden_size=hidden_size, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(
            hidden_size=hidden_size, dropout=dropout
        )
        self.output_sublayer = SublayerConnection(
            hidden_size=hidden_size, dropout=dropout
        )
        self.is_causal = is_causal

    def forward(self, hidden, padding_mask):
        hidden = self.input_sublayer(
            hidden,
            lambda _hidden: self.attention.forward(
                _hidden,
                _hidden,
                _hidden,
                padding_mask=padding_mask,
                is_causal=self.is_causal,
            ),
        )
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return hidden


class TransformerEncoder(nn.Module):
    def __init__(self, args, num_blocks, hidden_size=None, is_causal=None):
        super(TransformerEncoder, self).__init__()
        if hidden_size is not None:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        if is_causal is not None:
            self.is_causal = is_causal
        else:
            self.is_causal = args.is_causal
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    self.hidden_size,
                    self.heads,
                    self.dropout,
                    self.is_causal,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, hidden, padding_mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, padding_mask)
        return hidden


def get_timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output (must be even).
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps.unsqueeze(-1).float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
