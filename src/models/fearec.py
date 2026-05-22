import math

import torch
import torch.nn as nn
import torch.nn.functional as fn


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


class _HybridAttention(nn.Module):
    """Time + frequency domain hybrid attention layer (one transformer block)."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        layer_idx,
        config,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})."
            )

        self.factor = config["topk_factor"]
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        self.global_ratio = config["global_ratio"]
        self.n_layers = config["n_layers"]
        self.filter_mixer = "G" if self.global_ratio > (1 / self.n_layers) else "L"

        self.max_item_list_length = config["MAX_ITEM_LIST_LENGTH"]
        self.dual_domain = config["dual_domain"]
        # Avoid div-by-zero when n_layers == 1.
        denom = max(self.n_layers - 1, 1)
        self.slide_step = (
            (self.max_item_list_length // 2 + 1) * (1 - self.global_ratio)
        ) // denom
        self.local_ratio = 1 / self.n_layers
        self.filter_size = self.local_ratio * (self.max_item_list_length // 2 + 1)

        if self.filter_mixer == "G":
            self.w = self.global_ratio
            self.s = self.slide_step
        else:
            self.w = self.local_ratio
            self.s = self.filter_size

        left = int(
            ((self.max_item_list_length // 2 + 1) * (1 - self.w))
            - (layer_idx * self.s)
        )
        right = int(
            (self.max_item_list_length // 2 + 1) - layer_idx * self.s
        )
        left = max(left, 0)
        right = min(right, self.max_item_list_length // 2 + 1)
        if right <= left:
            right = left + 1

        self.q_index = list(range(left, right))
        self.k_index = list(range(left, right))
        self.v_index = list(range(left, right))

        self.std = config["std"]
        if self.std:
            self.time_q_index = list(self.q_index)
            self.time_k_index = list(self.k_index)
            self.time_v_index = list(self.v_index)
        else:
            full = list(range(self.max_item_list_length // 2 + 1))
            self.time_q_index = full
            self.time_k_index = full
            self.time_v_index = full

        self.use_filter = config.get("use_filter", False)
        if self.use_filter:
            # learnable complex-valued filter on the selected frequency bins
            self.complex_weight = nn.Parameter(
                torch.randn(
                    1,
                    self.num_attention_heads,
                    self.attention_head_size,
                    len(self.q_index),
                    2,
                )
                * 0.02
            )

        if self.dual_domain:
            self.spatial_ratio = config["spatial_ratio"]

    def _transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.view(*new_shape)

    def _time_delay_agg_training(self, values, corr):
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        top_k = max(int(self.factor * math.log(length)), 1)
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack(
            [mean_value[:, index[i]] for i in range(top_k)], dim=-1
        )
        tmp_corr = torch.softmax(weights, dim=-1)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def _time_delay_agg_inference(self, values, corr):
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        init_index = (
            torch.arange(length)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
            .to(values.device)
        )
        top_k = max(int(self.factor * math.log(length)), 1)
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                1
            ).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def forward(self, input_tensor, attention_mask):
        mixed_query = self.query_layer(input_tensor)
        mixed_key = self.key_layer(input_tensor)
        mixed_value = self.value_layer(input_tensor)

        queries = self._transpose_for_scores(mixed_query)
        keys = self._transpose_for_scores(mixed_key)
        values = self._transpose_for_scores(mixed_value)

        _, L, _, _ = queries.shape
        _, S, _, _ = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, : (L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies via FFT
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)

        q_fft_box = q_fft[:, :, :, self.q_index]
        k_fft_box = k_fft[:, :, :, self.k_index]
        res = q_fft_box * torch.conj(k_fft_box)

        if self.use_filter:
            weight = torch.view_as_complex(self.complex_weight)
            res = res * weight

        B, H, E = q_fft.shape[0], q_fft.shape[1], q_fft.shape[2]
        box_res = torch.zeros(
            B, H, E, L // 2 + 1, device=q_fft.device, dtype=torch.cfloat
        )
        box_res[:, :, :, self.q_index] = res

        corr = torch.fft.irfft(box_res, dim=-1, n=L)

        if self.training:
            V = self._time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)
        else:
            V = self._time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)

        context_layer = V.view(*V.size()[:-2], self.all_head_size)

        if self.dual_domain:
            # time-domain attention via the same fft basis but restricted to
            # ``time_*_index`` bins; the inverse rfft brings us back to a
            # complementary representation that we mix into the context.
            q_fft_box = q_fft[:, :, :, self.time_q_index]
            spatial_q = torch.zeros(
                B, H, E, L // 2 + 1, device=q_fft.device, dtype=torch.cfloat
            )
            spatial_q[:, :, :, self.time_q_index] = q_fft_box

            k_fft_box = k_fft[:, :, :, self.time_k_index]
            spatial_k = torch.zeros(
                B, H, E, L // 2 + 1, device=k_fft.device, dtype=torch.cfloat
            )
            spatial_k[:, :, :, self.time_k_index] = k_fft_box

            v_fft = torch.fft.rfft(values.permute(0, 2, 3, 1).contiguous(), dim=-1)
            v_fft_box = v_fft[:, :, :, self.time_v_index]
            spatial_v = torch.zeros(
                B, H, E, L // 2 + 1, device=v_fft.device, dtype=torch.cfloat
            )
            spatial_v[:, :, :, self.time_v_index] = v_fft_box

            queries_t = torch.fft.irfft(spatial_q, dim=-1, n=L)
            keys_t = torch.fft.irfft(spatial_k, dim=-1, n=L)
            values_t = torch.fft.irfft(spatial_v, dim=-1, n=L)

            queries_t = queries_t.permute(0, 1, 3, 2)
            keys_t = keys_t.permute(0, 1, 3, 2)
            values_t = values_t.permute(0, 1, 3, 2)

            attention_scores = torch.matmul(queries_t, keys_t.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.attn_dropout(attention_probs)
            qkv = torch.matmul(attention_probs, values_t)
            context_layer_spatial = qkv.permute(0, 2, 1, 3).contiguous()
            context_layer_spatial = context_layer_spatial.view(
                *context_layer_spatial.size()[:-2], self.all_head_size
            )
            context_layer = (
                1 - self.spatial_ratio
            ) * context_layer + self.spatial_ratio * context_layer_spatial

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class _FEABlock(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        layer_idx,
        config,
    ):
        super().__init__()
        self.hybrid_attention = _HybridAttention(
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
            layer_idx,
            config,
        )
        self.feed_forward = _FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.hybrid_attention(hidden_states, attention_mask)
        return self.feed_forward(attention_output)


class _FEAEncoder(nn.Module):
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
        config,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                _FEABlock(
                    n_heads,
                    hidden_size,
                    inner_size,
                    hidden_dropout_prob,
                    attn_dropout_prob,
                    hidden_act,
                    layer_norm_eps,
                    n,
                    config,
                )
                for n in range(n_layers)
            ]
        )

    def forward(self, hidden_states, attention_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class FEARec(nn.Module):
    def __init__(self, args):
        super(FEARec, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.item_num = args.item_num
        self.max_len = args.max_len

        self.n_layers = getattr(args, "fearec_n_layers", 2)
        self.n_heads = getattr(args, "fearec_n_heads", 4)
        self.inner_size = getattr(args, "fearec_inner_size", self.hidden_size * 4)

        config = {
            "topk_factor": getattr(args, "fearec_topk_factor", 5),
            "use_filter": getattr(args, "fearec_use_filter", False),
            "std": getattr(args, "fearec_std", True),
            "global_ratio": getattr(args, "fearec_global_ratio", 0.6),
            "n_layers": self.n_layers,
            "MAX_ITEM_LIST_LENGTH": self.max_len,
            "dual_domain": getattr(args, "fearec_dual_domain", True),
            "spatial_ratio": getattr(args, "fearec_spatial_ratio", 0.5),
        }

        self.item_embedding = nn.Embedding(
            self.item_num + 1, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_size)

        self.item_encoder = _FEAEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=args.dropout,
            attn_dropout_prob=args.dropout,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            config=config,
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
        """Causal mask combined with padding mask in additive form."""
        attention_mask = (item_seq > 0).long()
        extended = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        causal = torch.triu(
            torch.ones((1, max_len, max_len), device=item_seq.device), diagonal=1
        )
        causal = (causal == 0).unsqueeze(1).long()
        extended = extended * causal
        extended = extended.to(dtype=next(self.parameters()).dtype)
        extended = (1.0 - extended) * -10000.0
        return extended

    def forward(self, item_seq, tgt_seq, train_flag=True):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_emb = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        attention_mask = self._attention_mask(item_seq)
        out_seq = self.item_encoder(input_emb, attention_mask)
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
