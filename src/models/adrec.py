import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import TransformerEncoder, get_timestep_embedding, RMSNorm
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

    def forward_with_cfg(self, rep_item, x_t, t, mask_seq, mask_tgt, cfg_scale=1.0):
        """Forward pass with classifier-free guidance."""
        cond_x_0 = self.forward(rep_item, x_t, t, mask_seq, mask_tgt)
        uncond_x_0 = self.forward(rep_item, x_t, t, mask_seq, mask_tgt, condition=False)
        x_0 = uncond_x_0 + cfg_scale * (cond_x_0 - uncond_x_0)
        return x_0

    def forward(self, rep_item, x_t, t, mask_seq, mask_tgt, condition=True):
        if condition is not True:
            rep_item = torch.zeros_like(rep_item)
        t = t.reshape(x_t.shape[0], -1)
        time_emb = self.time_embed(get_timestep_embedding(t, self.hidden_size))
        lambda_uncertainty = self.lambda_uncertainty

        rep_diffu = rep_item + lambda_uncertainty * (x_t + time_emb)

        if isinstance(self.decoder, nn.Sequential):
            rep_diffu = self.decoder(rep_diffu)
            rep_diffu = self.dropout(rep_diffu)
        else:
            rep_diffu = self.decoder(rep_diffu, mask_seq)
            rep_diffu = self.dropout(rep_diffu)
        return rep_diffu


class AdRec(nn.Module):
    def __init__(
        self,
        args,
    ):
        super(AdRec, self).__init__()
        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        self.use_timesteps = space_timesteps(
            self.diffusion_steps, [self.diffusion_steps]
        )

        betas = get_named_beta_schedule(args)
        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.num_timesteps = int(self.betas.shape[0])

        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_name, self.num_timesteps
        )
        self.rescale_timesteps = args.rescale_timesteps

        self.net = Denoiser(args)
        self.independent_diffusion = args.independent
        self.cfg_scale = args.cfg_scale
        self.geodesic = args.geodesic
        self.ag_encoder = TransformerEncoder(
            args, num_blocks=2
        )

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        if self.geodesic:
            x_start = F.normalize(x_start, p=2, dim=-1)
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
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
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        assert posterior_mean.shape[0] == x_start.shape[0]
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq, mask_tag):
        if self.cfg_scale == 1:
            out_seq = self.net(
                rep_item, x_t, self._scale_timesteps(t), mask_seq, mask_tag
            )
        else:
            out_seq = self.net.forward_with_cfg(
                rep_item,
                x_t,
                self._scale_timesteps(t),
                mask_seq,
                mask_tag,
                self.cfg_scale,
            )
        x_0 = out_seq
        model_log_variance = np.log(
            np.append(self.posterior_variance[1], self.betas[1:])
        )
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)

        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq, mask_tag):
        model_mean, model_log_variance = self.p_mean_variance(
            item_rep, noise_x_t, t, mask_seq, mask_tag
        )
        noise = torch.randn_like(noise_x_t)
        nonzero_mask = (t != 0).float().unsqueeze(-1)
        sample_xt = (
            model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        )
        if self.geodesic:
            sample_xt = F.normalize(sample_xt, p=2, dim=-1)
        return sample_xt

    def denoise_sample(self, seq, tgt, mask_seq, mask_tag):
        seq = self.ag_encoder(seq, mask_seq)
        noise_x_t = torch.randn_like(tgt)
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

    def independent_diffuse(self, tgt, mask, is_independent=False):
        if is_independent:
            t, weights = self.schedule_sampler.sample(
                tgt.shape[0] * tgt.shape[1], tgt.device
            )
            t = t * mask.reshape(-1).long()
            x_t = self.q_sample(
                tgt.reshape(-1, tgt.shape[-1]), t, mask=mask.reshape(-1)
            ).reshape(*tgt.shape)
        else:
            t, weights = self.schedule_sampler.sample(tgt.shape[0], tgt.device)
            x_t = self.q_sample(tgt, t, mask=mask)
        return x_t, t

    def forward(self, item_rep, item_tag, mask_seq, mask_tag):
        item_rep = self.ag_encoder(item_rep, mask_seq)
        x_t, t = self.independent_diffuse(
            item_tag, mask_tag, self.independent_diffusion
        )
        if self.cfg_scale != 1:
            mask = torch.rand([mask_seq.shape[0], 1, 1], device=item_rep.device) > 0.9
            item_rep = torch.where(mask, torch.zeros_like(item_rep), item_rep)
        denoised_seq = self.net(
            item_rep, x_t, self._scale_timesteps(t), mask_seq, mask_tag
        )
        losses = F.mse_loss(denoised_seq, item_tag, reduction="none") * (
            mask_tag / mask_tag.sum(1, keepdim=True)
        ).unsqueeze(-1)
        losses = losses.sum(1).mean()
        return denoised_seq, losses
