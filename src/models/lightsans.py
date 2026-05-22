import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as fn


class _ItemToInterestAggregation(nn.Module):
    def __init__(self, seq_len, hidden_size, k_interests=5):
        super().__init__()
        self.k_interests = k_interests
        self.theta = nn.Parameter(torch.randn([hidden_size, k_interests]))

    def forward(self, input_tensor):
        D_matrix = torch.matmul(input_tensor, self.theta)
        D_matrix = nn.Softmax(dim=-2)(D_matrix)
        result = torch.einsum("nij, nik -> nkj", input_tensor, D_matrix)
        return result


class _LightMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        k_interests,
        hidden_size,
        seq_len,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})."
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attpooling_key = _ItemToInterestAggregation(
            seq_len, hidden_size, k_interests
        )
        self.attpooling_value = _ItemToInterestAggregation(
            seq_len, hidden_size, k_interests
        )

        self.attn_scale_factor = 2
        self.pos_q_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_k_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_scaling = (
            float(self.attention_head_size * self.attn_scale_factor) ** -0.5
        )
        self.pos_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, pos_emb):
        mixed_query = self.query(input_tensor)
        mixed_key = self.key(input_tensor)
        mixed_value = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query)
        key_layer = self.transpose_for_scores(self.attpooling_key(mixed_key))
        value_layer = self.transpose_for_scores(self.attpooling_value(mixed_value))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-2)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_item = torch.matmul(attention_probs, value_layer)

        value_layer_pos = self.transpose_for_scores(mixed_value)
        pos_emb = self.pos_ln(pos_emb).unsqueeze(0)
        pos_query_layer = (
            self.transpose_for_scores(self.pos_q_linear(pos_emb)) * self.pos_scaling
        )
        pos_key_layer = self.transpose_for_scores(self.pos_k_linear(pos_emb))

        abs_pos_bias = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))
        abs_pos_bias = abs_pos_bias / math.sqrt(self.attention_head_size)
        abs_pos_bias = nn.Softmax(dim=-2)(abs_pos_bias)
        context_pos = torch.matmul(abs_pos_bias, value_layer_pos)

        context_layer = context_item + context_pos
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            *context_layer.size()[:-2], self.all_head_size
        )
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


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

    def forward(self, input_tensor):
        hidden = self.dense_1(input_tensor)
        hidden = self.intermediate_act_fn(hidden)
        hidden = self.dense_2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.LayerNorm(hidden + input_tensor)
        return hidden


class _LightTransformerLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        k_interests,
        hidden_size,
        seq_len,
        inner_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super().__init__()
        self.multi_head_attention = _LightMultiHeadAttention(
            n_heads,
            k_interests,
            hidden_size,
            seq_len,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
        )
        self.feed_forward = _FeedForward(
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, pos_emb):
        attention_output = self.multi_head_attention(hidden_states, pos_emb)
        return self.feed_forward(attention_output)


class _LightTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        k_interests=5,
        hidden_size=64,
        seq_len=50,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        layer = _LightTransformerLayer(
            n_heads,
            k_interests,
            hidden_size,
            seq_len,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, pos_emb):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, pos_emb)
        return hidden_states


class LightSANs(nn.Module):
    def __init__(self, args):
        super(LightSANs, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.item_num = args.item_num
        self.max_len = args.max_len

        self.n_layers = getattr(args, "lightsans_n_layers", 2)
        self.n_heads = getattr(args, "lightsans_n_heads", 2)
        self.k_interests = getattr(args, "lightsans_k_interests", 5)
        self.inner_size = getattr(args, "lightsans_inner_size", self.hidden_size * 4)

        self.item_embedding = nn.Embedding(
            self.item_num + 1, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_size)
        self.trm_encoder = _LightTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            k_interests=self.k_interests,
            hidden_size=self.hidden_size,
            seq_len=self.max_len,
            inner_size=self.inner_size,
            hidden_dropout_prob=args.dropout,
            attn_dropout_prob=args.dropout,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
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

    def forward(self, item_seq, tgt_seq, train_flag=True):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_emb = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)

        out_seq = self.trm_encoder(item_emb, position_emb)
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
