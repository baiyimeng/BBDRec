import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import TransformerEncoder, get_timestep_embedding
from utils.diffusion import (
    _extract_into_tensor,
    create_named_schedule_sampler,
    exponential_mapping,
    get_named_beta_schedule,
    space_timesteps,
)


class Denoiser(nn.Module):
    """Denoising network for diffusion model."""

    def __init__(self, args):
        super(Denoiser, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        if args.diff_decoder == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                nn.SiLU(),
                nn.Linear(self.hidden_size * 4, self.hidden_size),
            )
        else:
            self.decoder = TransformerEncoder(
                args,
                num_blocks=2,
                hidden_size=self.hidden_size,
            )

        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, rep_item, x_t, t, mask_seq, mask_tgt):
        t = t.reshape(x_t.shape[0], -1)
        time_emb = self.time_embed(get_timestep_embedding(t, self.hidden_size))

        rep_diffu = rep_item + self.lambda_uncertainty * (x_t + time_emb)

        if isinstance(self.decoder, nn.Sequential):
            rep_diffu = self.decoder(rep_diffu)
            rep_diffu = self.dropout(rep_diffu)
        else:
            rep_diffu = self.decoder(rep_diffu, mask_seq)
            rep_diffu = self.dropout(rep_diffu)

        return rep_diffu


class BBDRec(nn.Module):
    def __init__(self, args):
        super(BBDRec, self).__init__()
        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        self.var_max = args.var_max
        self.eps = 1e-4

        self.b_t = np.linspace(
            self.eps,
            1.0 - self.eps,
            self.diffusion_steps,
            dtype=np.float64,
        )
        self.b_t_minus_1 = np.append(0, self.b_t[:-1])

        self.d_t = 2 * self.var_max * self.b_t * (1 - self.b_t)
        self.d_t_minus_1 = np.append(0.0, self.d_t[:-1])

        self.g_t = (1 - self.b_t) / (1 - self.b_t_minus_1)
        self.d_t_hat = self.d_t - self.d_t_minus_1 * (self.g_t) ** 2
        self.d_t_tilde = self.d_t_hat * self.d_t_minus_1 / self.d_t

        self.coef_t = self.d_t_minus_1 / self.d_t * self.g_t
        self.coef_0 = self.d_t_hat / self.d_t * (1 - self.b_t_minus_1)
        self.coef_T = (
            self.b_t_minus_1 - self.b_t * self.g_t * self.d_t_minus_1 / self.d_t
        )

        self.num_timesteps = int(self.b_t.shape[0])
        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_name, self.num_timesteps
        )
        self.rescale_timesteps = args.rescale_timesteps

        self.net = Denoiser(args)
        self.ag_encoder = TransformerEncoder(args, num_blocks=2)
        self.independent_diffusion = args.independent
        self.geodesic = args.geodesic

    def q_sample(self, x_start, t, item_rep, noise=None, mask=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        if self.geodesic:
            x_start = F.normalize(x_start, p=2, dim=-1)
        b_t = _extract_into_tensor(self.b_t, t, x_start.shape)
        d_t = _extract_into_tensor(self.d_t, t, x_start.shape)
        x_t = (1 - b_t) * x_start + b_t * item_rep + torch.sqrt(d_t) * noise
        if self.geodesic:
            # exp_x[v] = cos(||v||) * x + sin(||v||) * (v / ||v||)
            x_t = exponential_mapping(x_start, x_t)
        if mask == None:
            return x_t
        else:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return torch.where(mask == 0, x_start, x_t)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.coef_0, t, x_t.shape) * x_start
            + _extract_into_tensor(self.coef_t, t, x_t.shape) * x_t
        )
        assert posterior_mean.shape[0] == x_start.shape[0]
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq, mask_tag):
        out_seq = self.net(rep_item, x_t, self._scale_timesteps(t), mask_seq, mask_tag)
        x_0 = out_seq
        model_variance = _extract_into_tensor(self.d_t_tilde, t, x_t.shape)

        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq, mask_tag):
        model_mean, model_variance = self.p_mean_variance(
            item_rep, noise_x_t, t, mask_seq, mask_tag
        )
        noise = torch.randn_like(noise_x_t)
        nonzero_mask = (t != 0).float().unsqueeze(-1)
        sample_xt = model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
        if self.geodesic:
            sample_xt = F.normalize(sample_xt, p=2, dim=-1)
        return sample_xt

    def denoise_sample(self, seq, tgt, mask_seq, mask_tag):
        seq = self.ag_encoder(seq, mask_seq)
        noise_x_t = torch.concat([tgt[:, :-1], seq[:, -1:]], dim=1)
        indices = list(range(self.num_timesteps))[::-1]
        for i in indices:
            t = (
                torch.tensor([0] * (seq.shape[1] - 1) + [i], device=seq.device)
                .unsqueeze(0)
                .repeat(seq.shape[0], 1)
            )
            noise_x_t = torch.concat([tgt[:, :-1], noise_x_t[:, -1:]], dim=1)
            noise_x_t = self.p_sample(seq, noise_x_t, t, mask_seq, mask_tag)
        return noise_x_t

    def independent_diffuse(self, tgt, mask, item_rep, is_independent=False):
        if is_independent:
            t, weights = self.schedule_sampler.sample(
                tgt.shape[0] * tgt.shape[1], tgt.device
            )
            t = t * mask.reshape(-1).long()
            x_t = self.q_sample(
                tgt.reshape(-1, tgt.shape[-1]),
                t,
                mask=mask.reshape(-1),
                item_rep=item_rep.reshape(-1, item_rep.shape[-1]),
            ).reshape(*tgt.shape)
        else:
            t, weights = self.schedule_sampler.sample(tgt.shape[0], tgt.device)
            x_t = self.q_sample(tgt, t, mask=mask)
        return x_t, t

    def forward(self, item_rep, item_tag, mask_seq, mask_tag):
        item_rep = self.ag_encoder(item_rep, mask_seq)
        x_t, t = self.independent_diffuse(
            item_tag,
            mask_tag,
            item_rep,
            self.independent_diffusion,
        )
        denoised_seq = self.net(
            item_rep, x_t, self._scale_timesteps(t), mask_seq, mask_tag
        )
        losses = F.mse_loss(denoised_seq, item_tag, reduction="none") * (
            mask_tag / mask_tag.sum(1, keepdim=True)
        ).unsqueeze(-1)
        losses = losses.sum(1).mean()
        return denoised_seq, losses
