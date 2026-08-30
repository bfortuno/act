"""
Original Diffusion Policy (Chi et al., 2023) for the ACT repo.

Two interchangeable denoisers under one policy class, selected by ``--dp_denoiser``:

  * ``unet`` (default) - the canonical CNN 1-D temporal U-Net (``ConditionalUnet1D``)
    with FiLM global conditioning. The U-Net + helpers are vendored verbatim from
    ``real-stanford/diffusion_policy`` (``diffusion_policy/model/diffusion/
    conditional_unet1d.py`` + ``conv1d_components.py`` + ``positional_embedding.py``,
    MIT). Observation conditioning = per-camera ResNet-18 features (spatial-softmax
    keypoints or global average pool) concatenated with the proprio vector.

  * ``dit`` - reuse the *exact* gr00t-style DiT denoiser and visual cross-attention
    conditioning from ``detr/models/diffusion_flow.py`` (``DiffusionFlowModel``),
    but interpret its output as the DDPM noise prediction instead of a
    flow-matching velocity. Lets a working DiT-DDPM be compared one variable at a
    time against the (non-working) DiT flow-matching policy.

Generative process (both denoisers): standard DDPM (``diffusers`` ``DDPMScheduler``,
``num_train_timesteps`` = 100, cosine ``squaredcos_cap_v2`` betas, ``epsilon``
prediction, ``clip_sample=False`` because this repo z-scores actions rather than
min-max scaling them to [-1, 1]). Sampling: ``DDPMScheduler`` (100 steps) or
``DDIMScheduler`` (few steps), chosen by ``--dp_inference_scheduler``.

The full action chunk ``(B, T, action_dim)`` is predicted from a single observation
timestep (no obs-history stack - this repo's dataloader provides one).
"""

import math
from types import SimpleNamespace

import einops
import IPython
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from einops.layers.torch import Rearrange
from torch import nn

from .backbone import build_backbone
from .diffusion_flow import build_diffusion_flow

e = IPython.embed


# ---------------------------------------------------------------------------
# vendored from real-stanford/diffusion_policy (MIT) - do not edit
# ---------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871 - per-channel scale + bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """x: (B, in_channels, horizon); cond: (B, cond_dim) -> (B, out_channels, horizon)"""
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, sample, timestep, local_cond=None, global_cond=None, **kwargs):
        """sample: (B, T, input_dim); timestep: (B,) or int; global_cond: (B, global_cond_dim).
        Returns (B, T, input_dim). (Axis names b/h/t below are the upstream ones - the
        first rearrange just moves input_dim next to the batch for the 1-D convs.)"""
        sample = einops.rearrange(sample, "b h t -> b t h")

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        h_local = []
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder
            h_local.append(resnet(local_cond, global_feature))
            h_local.append(resnet2(local_cond, global_feature))

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x


# ---------------------------------------------------------------------------
# observation encoder (unet denoiser)
# ---------------------------------------------------------------------------


class SpatialSoftmax(nn.Module):
    """Per-channel spatial softmax -> expected (x, y) keypoint coords in [-1, 1]
    (Finn et al. 2016; the image encoder used by the original Diffusion Policy)."""

    def __init__(self, num_kp):
        super().__init__()
        self.num_kp = num_kp
        # derived constant grid, cached as plain (non-buffer) tensors so it is not in
        # state_dict / .buffers() - its (h, w)-dependent shape must not reach the EMA
        # buffer copy or a checkpoint reload.
        self._grid_hw = None
        self.pos_x = None
        self.pos_y = None

    def _maybe_build_grid(self, h, w, device):
        if self._grid_hw == (h, w) and self.pos_x is not None and self.pos_x.device == device:
            return
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing="ij",
        )
        self.pos_x = xs.reshape(h * w)
        self.pos_y = ys.reshape(h * w)
        self._grid_hw = (h, w)

    def forward(self, feat):
        # feat: (B, num_kp, h, w)
        b, k, h, w = feat.shape
        self._maybe_build_grid(h, w, feat.device)
        attn = F.softmax(feat.reshape(b, k, h * w), dim=-1)  # (B, K, hw)
        exp_x = (attn * self.pos_x).sum(-1)  # (B, K)
        exp_y = (attn * self.pos_y).sum(-1)  # (B, K)
        return torch.stack([exp_x, exp_y], dim=-1).reshape(b, k * 2)  # (B, 2K)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class DiffusionPolicyModel(nn.Module):
    def __init__(
        self,
        args,
        state_dim,
        action_dim,
        num_queries,
        camera_names,
        denoiser="unet",
        # unet denoiser
        obs_pool="spatial_softmax",
        num_kp=32,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=128,
        # diffusion process
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        action_scale=4.0,
        n_inference_steps=100,
        inference_scheduler="ddim",
    ):
        super().__init__()
        assert denoiser in ("unet", "dit")
        assert obs_pool in ("spatial_softmax", "avg")
        assert prediction_type in ("epsilon", "sample")
        assert inference_scheduler in ("ddim", "ddpm")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.denoiser = denoiser
        self.obs_pool = obs_pool
        self.num_kp = num_kp
        self.down_dims = list(down_dims)
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        # DDPM/DDIM sampling with a cosine schedule + few train timesteps is numerically
        # unstable unless pred_x0 is clipped to [-1, 1] (stock Diffusion Policy does this,
        # relying on its actions being min-max scaled to [-1, 1]). This repo z-scores
        # actions instead, so divide by ``action_scale`` (~4 sigma -> [-1, 1]) around the
        # diffusion process and undo it on sampling; clip then only touches real outliers.
        self.action_scale = action_scale
        self.n_inference_steps = n_inference_steps
        self.inference_scheduler = inference_scheduler
        # gr00t DiT groups the gripper as the last channel of each bimanual arm block
        self.dit_arch = getattr(args, "dit_arch", "cross_attn")

        n_cam = len(camera_names)

        if denoiser == "unet":
            self.backbones = nn.ModuleList([build_backbone(args) for _ in camera_names])
            feat_ch = self.backbones[0].num_channels  # 512 for resnet18
            if obs_pool == "spatial_softmax":
                self.kp_heads = nn.ModuleList(
                    [nn.Conv2d(feat_ch, num_kp, kernel_size=1) for _ in camera_names]
                )
                self.spatial_softmax = SpatialSoftmax(num_kp)
                pool_out = 2 * num_kp
            else:
                self.kp_heads = None
                self.spatial_softmax = None
                pool_out = feat_ch
            self.global_cond_dim = n_cam * pool_out + state_dim
            self.cond_norm = nn.LayerNorm(self.global_cond_dim)
            self.net = ConditionalUnet1D(
                input_dim=action_dim,
                global_cond_dim=self.global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=self.down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=True,
            )
        else:  # dit: reuse the flow-matching model's DiT + visual cross-attn conditioning
            self.dit = build_diffusion_flow(args)  # a DiffusionFlowModel
            self.global_cond_dim = None

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule=beta_schedule,
            variance_type="fixed_small",
            clip_sample=True,  # bounds pred_x0; actions are pre-divided by action_scale
            prediction_type=prediction_type,
        )

    # ------------------------------------------------------------------
    def encode_obs(self, qpos, image):
        """image (B, n_cam, 3, H, W) already ImageNet-normalized.

        unet -> global_cond (B, global_cond_dim).
        dit  -> (state_token, vis_tokens, vis_kpm) tuple from DiffusionFlowModel.
        """
        if self.denoiser == "dit":
            return self.dit.encode_obs(qpos, image, self.training)

        parts = []
        for cam_id in range(len(self.camera_names)):
            features, _pos = self.backbones[cam_id](image[:, cam_id])
            feat = features[0]  # (B, C, h, w)
            if self.obs_pool == "spatial_softmax":
                parts.append(self.spatial_softmax(self.kp_heads[cam_id](feat)))
            else:
                parts.append(F.adaptive_avg_pool2d(feat, 1).flatten(1))
        parts.append(qpos)
        return self.cond_norm(torch.cat(parts, dim=-1))

    def _predict_noise(self, x_t, t, cond):
        """x_t (B, T, A); t (B,) long; cond = encode_obs(...) output. -> (B, T, A)."""
        if self.denoiser == "unet":
            return self.net(x_t, t, global_cond=cond)
        state_token, vis_tokens, vis_kpm = cond
        t_cont = t.float() / self.num_train_timesteps  # -> [0, 1) for the DiT time encoder
        return self.dit.denoise(x_t, t_cont, state_token, vis_tokens, vis_kpm)

    def _make_infer_scheduler(self):
        if self.inference_scheduler == "ddpm" or self.n_inference_steps >= self.num_train_timesteps:
            sched = DDPMScheduler.from_config(self.train_scheduler.config)
        else:
            sched = DDIMScheduler.from_config(self.train_scheduler.config)
        steps = min(self.n_inference_steps, self.num_train_timesteps)
        sched.set_timesteps(steps)
        return sched

    def _sample(self, cond, batch_size, device, generator=None):
        """Returns a chunk in the *scaled* action space (caller multiplies by action_scale)."""
        sched = self._make_infer_scheduler()
        x = torch.randn(
            batch_size, self.num_queries, self.action_dim, device=device, generator=generator
        )
        for t in sched.timesteps:
            model_out = self._predict_noise(x, t.to(device).expand(batch_size), cond)
            step_kw = {}
            if isinstance(sched, DDIMScheduler):
                step_kw["eta"] = 0.0
            x = sched.step(model_out, t, x, generator=generator, **step_kw).prev_sample
        return x

    # ------------------------------------------------------------------
    def compute_loss(self, qpos, image, actions, is_pad):
        cond = self.encode_obs(qpos, image)
        bs = actions.shape[0]

        x0 = actions / self.action_scale
        noise = torch.randn_like(x0)
        timesteps = torch.randint(0, self.num_train_timesteps, (bs,), device=actions.device).long()
        noisy = self.train_scheduler.add_noise(x0, noise, timesteps)
        pred = self._predict_noise(noisy, timesteps, cond)
        target = noise if self.prediction_type == "epsilon" else x0

        m = (~is_pad)[..., None].float()
        loss = (F.mse_loss(pred, target, reduction="none") * m).sum() / (
            m.sum() * self.action_dim + 1e-6
        )
        out = {"loss": loss}
        # Validation-only: MSE between the actually-sampled chunk and the ground truth
        # (an end-task signal, unlike the training noise MSE). Seeded so model selection
        # is stable across epochs. Costs one full sampling loop per val batch.
        if not self.training:
            g = torch.Generator(device=actions.device).manual_seed(0)
            with torch.no_grad():
                a_hat = self._sample(cond, bs, actions.device, generator=g) * self.action_scale
            out["sample_mse"] = (F.mse_loss(a_hat, actions, reduction="none") * m).sum() / (
                m.sum() * self.action_dim + 1e-6
            )
        return out

    @torch.no_grad()
    def sample(self, qpos, image):
        cond = self.encode_obs(qpos, image)
        bs = qpos.shape[0]
        return self._sample(cond, bs, qpos.device, generator=None) * self.action_scale

    def forward(self, qpos, image, actions=None, is_pad=None):
        if actions is not None:
            return self.compute_loss(qpos, image, actions, is_pad)
        return self.sample(qpos, image)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def build_diffusion_policy(args):
    model = DiffusionPolicyModel(
        args,
        state_dim=getattr(args, "state_dim", 14),
        action_dim=getattr(args, "action_dim", getattr(args, "state_dim", 14)),
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        denoiser=getattr(args, "denoiser", "unet"),
        obs_pool=getattr(args, "obs_pool", "spatial_softmax"),
        num_kp=getattr(args, "num_kp", 32),
        down_dims=getattr(args, "down_dims", [256, 512, 1024]),
        kernel_size=getattr(args, "kernel_size", 5),
        n_groups=getattr(args, "n_groups", 8),
        diffusion_step_embed_dim=getattr(args, "diffusion_step_embed_dim", 128),
        num_train_timesteps=getattr(args, "num_train_timesteps", 100),
        beta_schedule=getattr(args, "beta_schedule", "squaredcos_cap_v2"),
        prediction_type=getattr(args, "prediction_type", "epsilon"),
        action_scale=getattr(args, "action_scale", 4.0),
        n_inference_steps=getattr(args, "n_inference_steps", 100),
        inference_scheduler=getattr(args, "inference_scheduler", "ddim"),
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters / 1e6:.2f}M")
    return model


# ---------------------------------------------------------------------------
# self-test:  python -m detr.models.diffusion_policy
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    cams = ["top", "left_wrist"]

    def _args(state_dim, **kw):
        base = dict(
            backbone="resnet18",
            lr_backbone=1e-5,
            masks=False,
            dilation=False,
            position_embedding="sine",
            hidden_dim=256,
            dim_feedforward=1024,
            nheads=8,
            dropout=0.1,
            num_queries=16,
            camera_names=cams,
            state_dim=state_dim,
            action_dim=state_dim,
            # unet
            denoiser="unet",
            obs_pool="spatial_softmax",
            num_kp=16,
            down_dims=[64, 128, 256],
            kernel_size=5,
            n_groups=8,
            diffusion_step_embed_dim=64,
            # diffusion
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            action_scale=4.0,
            n_inference_steps=10,
            inference_scheduler="ddim",
            # dit path (consumed by build_diffusion_flow)
            dit_arch="cross_attn",
            dit_layers=2,
            state_dropout_prob=0.0,
            cam_dropout_prob=0.0,
            gripper_rescale=False,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    # small images keep the CPU self-test fast; ResNet18 stride-32 still yields a valid
    # feature map (192x256 -> 6x8) for spatial-softmax and the sine pos embedding.
    H, W = 192, 256

    configs = []
    for denoiser in ("unet", "dit"):
        for state_dim in (16, 20):
            pools = ("spatial_softmax", "avg") if denoiser == "unet" else ("spatial_softmax",)
            for pool in pools:
                configs.append((denoiser, state_dim, pool))

    for denoiser, state_dim, pool in configs:
        m = build_diffusion_policy(_args(state_dim, denoiser=denoiser, obs_pool=pool))
        n_cam = len(cams)
        if denoiser == "unet":
            pool_out = 2 * 16 if pool == "spatial_softmax" else m.backbones[0].num_channels
            assert m.global_cond_dim == n_cam * pool_out + state_dim, (
                m.global_cond_dim,
                n_cam * pool_out + state_dim,
            )

        B, T = 2, 16
        qpos = torch.randn(B, state_dim)
        img = torch.randn(B, n_cam, 3, H, W)
        act = torch.rand(B, T, state_dim)
        is_pad = torch.zeros(B, T, dtype=torch.bool)
        is_pad[0, 10:] = True

        m.train()
        out = m.compute_loss(qpos, img, act, is_pad)
        assert out["loss"].ndim == 0 and torch.isfinite(out["loss"]), (denoiser, state_dim, pool)
        out["loss"].backward()
        net_grads = [
            p.grad
            for n, p in m.named_parameters()
            if p.grad is not None and "backbone" not in n and "dit.backbones" not in n
        ]
        assert net_grads and all(torch.isfinite(g).all() for g in net_grads)

        m.eval()
        s = m.sample(qpos, img)
        assert s.shape == (B, m.num_queries, state_dim), s.shape
        assert torch.isfinite(s).all()

        out2 = m.compute_loss(qpos, img, act, is_pad)
        out3 = m.compute_loss(qpos, img, act, is_pad)
        assert "sample_mse" in out2 and torch.isfinite(out2["sample_mse"])
        assert torch.allclose(out2["sample_mse"], out3["sample_mse"]), (
            "sample_mse not deterministic"
        )

        print(
            f"ok  {denoiser:5s} pool={pool:14s} state_dim={state_dim:2d}  "
            f"loss={out['loss'].item():.4f} sample_mse={out2['sample_mse'].item():.4f}"
        )

    # overfit one fixed batch -> the sampled chunk must collapse onto the target.
    # This is the regression guard for the train/sample wiring (add_noise, scheduler
    # step, action_scale, clip_sample). Uses prediction_type="sample" and a short
    # schedule so it converges in a few hundred CPU steps; the epsilon-MSE loss has a
    # noise floor well above 0 and is a poor overfit signal. Backbones are frozen -
    # this tests the DDPM loop, not the CNN feature extractor.
    torch.manual_seed(0)
    m = build_diffusion_policy(
        _args(
            20,
            denoiser="unet",
            down_dims=[64, 128],
            diffusion_step_embed_dim=64,
            num_train_timesteps=50,
            prediction_type="sample",
            n_inference_steps=25,
        )
    )
    for n, p in m.named_parameters():
        if "backbone" in n:
            p.requires_grad_(False)
    B, T = 2, 16
    qpos = torch.randn(B, 20)
    img = torch.randn(B, len(cams), 3, H, W)
    act = torch.rand(B, T, 20)  # ~U[0,1], var ~0.083 -> predicting the mean gives sample_mse ~0.083
    is_pad = torch.zeros(B, T, dtype=torch.bool)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-3)
    m.train()
    for step in range(500):
        loss = m.compute_loss(qpos, img, act, is_pad)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  overfit step {step:4d}  x0_loss={loss.item():.5f}")
    m.eval()
    mse = m.compute_loss(qpos, img, act, is_pad)["sample_mse"].item()
    print(f"  overfit final sample_mse={mse:.5f}  (predict-the-mean baseline ~0.083)")
    assert mse < 0.02, f"failed to overfit one batch (sample_mse={mse})"

    print("all diffusion_policy self-tests passed")
