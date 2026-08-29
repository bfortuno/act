# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Diffusion / flow-matching policy for the ACT repo.

The denoiser is a DiT modelled on NVIDIA gr00t-n1d7
(``gr00t/model/gr00t_n1d7/gr00t_n1d7.py`` + ``gr00t/model/modules/dit.py``,
Apache-2.0), scaled down and re-wired to use ACT's ResNet CNN backbone
(``detr/models/backbone.py``) instead of a VLM.

Flow-matching formulation (pi0 / rectified flow, faithful to gr00t-n1d7):
  * timestep  t ~ (1 - Beta(alpha, beta)) * noise_s          -> t in [0, noise_s]
  * interpolation   x_t = (1 - t) * noise + t * action
  * regression target (velocity)   v = action - noise
  * masked MSE loss on v
  * sampling: explicit forward Euler, ``n_inference_steps`` steps, t: 0 -> 1

Two conditioning designs, switched by ``dit_arch``:
  * ``cross_attn`` - blocks alternate self-attention over [state, action] tokens
    and cross-attention to the flattened CNN feature-map tokens. Faithful to
    gr00t's interleaved DiT.
  * ``concat`` - one token stream [state, visual, action], pure self-attention.

Regularization (training only):
  * state dropout  - whole proprio vector zeroed with prob ``state_dropout_prob``
    (ported from gr00t ``state_dropout_prob``).
  * camera dropout - each camera's visual tokens masked out of attention with
    prob ``cam_dropout_prob`` (>= 1 camera always kept).
"""

import math
from types import SimpleNamespace

import IPython
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from .backbone import build_backbone

e = IPython.embed


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sinusoidal_embedding(values, dim, max_period=10000.0):
    """(...,) -> (..., dim) log-spaced [cos, sin] embedding (diffusers/OpenAI style)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=values.device) / half
    )
    args = values.float()[..., None] * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


class TimestepEncoder(nn.Module):
    """Discretized flow timestep -> (B, hidden) conditioning vector for adaLN."""

    def __init__(self, hidden_dim, num_timestep_buckets=1000, freq_dim=256):
        super().__init__()
        self.num_timestep_buckets = num_timestep_buckets
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_cont):
        # t_cont: (B,) continuous in [0, 1]; bucket then embed (gr00t dit.py:61-71)
        buckets = (t_cont * self.num_timestep_buckets).clamp(0, self.num_timestep_buckets - 1)
        return self.mlp(sinusoidal_embedding(buckets, self.freq_dim))


class AdaLN(nn.Module):
    """adaLN-zero: LayerNorm modulated by (shift, scale) plus a residual `gate`, all
    predicted from the timestep embedding. `proj` is zero-initialised, so at init
    shift = scale = gate = 0 and the sub-layer this wraps is an identity map
    (Peebles & Xie, "Scalable Diffusion Models with Transformers")."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, temb):
        shift, scale, gate = self.proj(F.silu(temb)).chunk(3, dim=-1)
        mod = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return mod, gate[:, None]


class DiTBlock(nn.Module):
    """adaLN-zero(attn) + adaLN-zero(FF) residual block. mode: 'self' or 'cross'.
    With zero-init adaLN every block starts as x -> x."""

    def __init__(self, hidden_dim, n_heads, mlp_ratio, dropout, mode):
        super().__init__()
        assert mode in ("self", "cross")
        self.mode = mode
        self.norm1 = AdaLN(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = AdaLN(hidden_dim)
        inner = int(hidden_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, temb, memory=None, key_padding_mask=None):
        mod1, gate1 = self.norm1(x, temb)
        kv = memory if self.mode == "cross" else mod1
        attn_out, _ = self.attn(mod1, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + gate1 * attn_out
        mod2, gate2 = self.norm2(x, temb)
        x = x + gate2 * self.ff(mod2)
        return x


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class DiffusionFlowModel(nn.Module):
    def __init__(
        self,
        backbones,
        state_dim,
        action_dim,
        num_queries,
        camera_names,
        hidden_dim=512,
        dim_feedforward=2048,
        nheads=8,
        dit_layers=8,
        dit_arch="cross_attn",
        dropout=0.1,
        n_inference_steps=4,
        state_dropout_prob=0.0,
        cam_dropout_prob=0.0,
        gripper_rescale=False,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
    ):
        super().__init__()
        assert dit_arch in ("cross_attn", "concat")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.hidden_dim = hidden_dim
        self.dit_arch = dit_arch
        self.n_inference_steps = n_inference_steps
        self.state_dropout_prob = state_dropout_prob
        self.cam_dropout_prob = cam_dropout_prob
        self.noise_s = noise_s
        self.num_timestep_buckets = num_timestep_buckets
        # near-binary {0,1} gripper channels routed through the Gaussian flow path get
        # mode-averaged; rescale them to [-1, 1] for the flow and undo it on sampling.
        # Gripper is always the last channel of each (bimanual) arm block.
        self.gripper_rescale = gripper_rescale
        self.gripper_idx = [action_dim // 2 - 1, action_dim - 1]
        self.register_buffer(
            "_beta_ab", torch.tensor([noise_beta_alpha, noise_beta_beta]), persistent=False
        )

        mlp_ratio = dim_feedforward / hidden_dim

        # --- vision ---
        self.backbones = nn.ModuleList(backbones)
        self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        self.cam_embed = nn.Embedding(len(camera_names), hidden_dim)
        self.vis_norm = nn.LayerNorm(hidden_dim)

        # --- proprio state ---
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- action / flow-time encoder (gr00t MultiEmbodimentActionEncoder, 1 embodiment) ---
        self.act_in = nn.Linear(action_dim, hidden_dim)
        self.act_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act_pos = nn.Embedding(num_queries, hidden_dim)

        # --- timestep conditioning ---
        self.time_enc = TimestepEncoder(hidden_dim, num_timestep_buckets)

        # --- DiT blocks ---
        blocks = []
        for idx in range(dit_layers):
            mode = ("cross" if idx % 2 == 0 else "self") if dit_arch == "cross_attn" else "self"
            blocks.append(DiTBlock(hidden_dim, nheads, mlp_ratio, dropout, mode))
        self.blocks = nn.ModuleList(blocks)

        self.norm_out = AdaLN(hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)
        # zero-init output head: the model predicts v = 0 at init (adaLN-zero recipe)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    def _beta_dist(self):
        return Beta(self._beta_ab[0], self._beta_ab[1])

    def encode_obs(self, qpos, image, training):
        """qpos (B, state_dim), image (B, n_cam, 3, H, W) already ImageNet-normalized.

        Returns (state_token (B,1,D), vis_tokens (B,N,D), vis_kpm (B,N) bool or None).
        """
        bs = qpos.shape[0]
        n_cam = len(self.camera_names)

        cam_tokens = []
        tokens_per_cam = None
        for cam_id in range(n_cam):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            feat = self.input_proj(features[0])  # (B, D, h, w)
            pos = pos[0]  # (1, D, h, w)
            h, w = feat.shape[-2:]
            tokens_per_cam = h * w
            feat = feat.flatten(2).permute(0, 2, 1)  # (B, hw, D)
            pos = pos.flatten(2).permute(0, 2, 1)  # (1, hw, D)
            tok = feat + pos + self.cam_embed.weight[cam_id][None, None]
            cam_tokens.append(tok)
        vis_tokens = torch.cat(cam_tokens, dim=1)  # (B, n_cam*hw, D)
        vis_tokens = self.vis_norm(vis_tokens)

        vis_kpm = None
        if training and self.cam_dropout_prob > 0 and n_cam > 1:
            keep = torch.rand(bs, n_cam, device=qpos.device) > self.cam_dropout_prob
            # guarantee >= 1 camera per sample
            none_kept = ~keep.any(dim=1)
            if none_kept.any():
                rand_cam = torch.randint(0, n_cam, (bs,), device=qpos.device)
                keep[none_kept, rand_cam[none_kept]] = True
            drop = ~keep  # (B, n_cam) True == masked
            vis_kpm = drop.repeat_interleave(tokens_per_cam, dim=1)  # (B, N)

        if training and self.state_dropout_prob > 0:
            drop_s = torch.rand(bs, 1, device=qpos.device) < self.state_dropout_prob
            qpos = qpos * (~drop_s)
        state_token = self.state_encoder(qpos)[:, None, :]

        return state_token, vis_tokens, vis_kpm

    def denoise(self, x_t, t_cont, state_token, vis_tokens, vis_kpm):
        """x_t (B, T, action_dim), t_cont (B,) -> predicted velocity (B, T, action_dim)."""
        bs, T, _ = x_t.shape
        temb = self.time_enc(t_cont)  # (B, D)

        time_feat = sinusoidal_embedding(
            (t_cont * self.num_timestep_buckets), self.hidden_dim
        )  # (B, D)
        a = self.act_in(x_t)  # (B, T, D)
        a = torch.cat([a, time_feat[:, None, :].expand(-1, T, -1)], dim=-1)
        a = self.act_mlp(a)
        a = a + self.act_pos.weight[None, :T, :]

        if self.dit_arch == "cross_attn":
            h = torch.cat([state_token, a], dim=1)  # (B, 1+T, D)
            self_kpm = None
        else:
            h = torch.cat([state_token, vis_tokens, a], dim=1)  # (B, 1+N+T, D)
            if vis_kpm is not None:
                pad = torch.zeros(bs, 1, dtype=torch.bool, device=h.device)
                tail = torch.zeros(bs, T, dtype=torch.bool, device=h.device)
                self_kpm = torch.cat([pad, vis_kpm, tail], dim=1)
            else:
                self_kpm = None

        for blk in self.blocks:
            if blk.mode == "cross":
                h = blk(h, temb, memory=vis_tokens, key_padding_mask=vis_kpm)
            else:
                h = blk(h, temb, key_padding_mask=self_kpm)

        h, _ = self.norm_out(h[:, -T:], temb)  # final adaLN modulation (gate unused, no residual)
        return self.head(h)

    # ------------------------------------------------------------------
    def compute_loss(self, qpos, image, actions, is_pad):
        state_token, vis_tokens, vis_kpm = self.encode_obs(qpos, image, self.training)

        x1 = actions  # (B, T, action_dim); normalized by the dataloader
        if self.gripper_rescale:
            x1 = x1.clone()
            x1[..., self.gripper_idx] = x1[..., self.gripper_idx] * 2.0 - 1.0  # [0,1] -> [-1,1]
        bs, T, _ = x1.shape
        x0 = torch.randn_like(x1)
        u = self._beta_dist().sample((bs,)).to(x1.device)
        t = (1.0 - u) * self.noise_s  # (B,) in [0, noise_s]
        t_b = t[:, None, None]
        x_t = (1.0 - t_b) * x0 + t_b * x1
        v_tgt = x1 - x0

        v_pred = self.denoise(x_t, t, state_token, vis_tokens, vis_kpm)

        m = (~is_pad)[..., None].float()
        loss = (F.mse_loss(v_pred, v_tgt, reduction="none") * m).sum() / (
            m.sum() * self.action_dim + 1e-6
        )
        out = {"loss": loss}
        # Validation-only diagnostic: how far the actually-sampled chunk lands from the
        # ground truth (an end-task signal, unlike the training velocity MSE). Skipped
        # in training since it costs n_inference_steps extra forward passes per batch.
        if not self.training:
            with torch.no_grad():
                a_hat = self._euler_sample(state_token, vis_tokens)
            out["sample_mse"] = (F.mse_loss(a_hat, x1, reduction="none") * m).sum() / (
                m.sum() * self.action_dim + 1e-6
            )
        return out

    def _euler_sample(self, state_token, vis_tokens):
        """Forward-Euler integrate the velocity field from Gaussian noise. Returns the
        raw model space (gripper in [-1,1] when gripper_rescale). No dropout / camera
        mask (obs must be encoded with training=False)."""
        bs = state_token.shape[0]
        x = torch.randn(bs, self.num_queries, self.action_dim, device=state_token.device)
        dt = 1.0 / self.n_inference_steps
        for i in range(self.n_inference_steps):
            t = torch.full((bs,), i * dt, device=state_token.device)
            x = x + dt * self.denoise(x, t, state_token, vis_tokens, None)
        return x

    @torch.no_grad()
    def sample(self, qpos, image):
        state_token, vis_tokens, _ = self.encode_obs(qpos, image, training=False)
        x = self._euler_sample(state_token, vis_tokens)
        if self.gripper_rescale:
            x = x.clone()
            x[..., self.gripper_idx] = (x[..., self.gripper_idx] + 1.0) / 2.0  # [-1,1] -> [0,1]
        return x

    def forward(self, qpos, image, actions=None, is_pad=None):
        if actions is not None:
            return self.compute_loss(qpos, image, actions, is_pad)
        return self.sample(qpos, image)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def build_diffusion_flow(args):
    backbones = [build_backbone(args) for _ in args.camera_names]

    model = DiffusionFlowModel(
        backbones,
        state_dim=getattr(args, "state_dim", 14),
        action_dim=getattr(args, "action_dim", getattr(args, "state_dim", 14)),
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        hidden_dim=args.hidden_dim,
        dim_feedforward=args.dim_feedforward,
        nheads=args.nheads,
        dit_layers=getattr(args, "dit_layers", 8),
        dit_arch=getattr(args, "dit_arch", "cross_attn"),
        dropout=args.dropout,
        n_inference_steps=getattr(args, "n_inference_steps", 4),
        state_dropout_prob=getattr(args, "state_dropout_prob", 0.0),
        cam_dropout_prob=getattr(args, "cam_dropout_prob", 0.0),
        gripper_rescale=getattr(args, "gripper_rescale", False),
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters / 1e6:.2f}M")
    return model


# ---------------------------------------------------------------------------
# self-test:  python -m detr.models.diffusion_flow
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
            dit_layers=4,
            n_inference_steps=4,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    for dit_arch in ("cross_attn", "concat"):
        for state_dim in (16, 20):  # quat-bimanual, rot6d-bimanual
            for sdp, cdp in ((0.0, 0.0), (0.5, 0.5)):
                m = build_diffusion_flow(
                    _args(
                        state_dim,
                        dit_arch=dit_arch,
                        state_dropout_prob=sdp,
                        cam_dropout_prob=cdp,
                        gripper_rescale=True,
                    )
                )
                B, T = 2, 16
                qpos = torch.randn(B, state_dim)
                img = torch.randn(B, len(cams), 3, 480, 640)
                act = torch.rand(B, T, state_dim)  # gripper channels want [0,1]
                is_pad = torch.zeros(B, T, dtype=torch.bool)
                is_pad[0, 10:] = True

                # adaLN-zero: every DiT block is identity and the head predicts 0 at init
                m.eval()
                with torch.no_grad():
                    st, vt, _ = m.encode_obs(qpos, img, training=False)
                    v0 = m.denoise(torch.randn(B, T, state_dim), torch.rand(B), st, vt, None)
                assert v0.abs().max() < 1e-5, ("head not zero at init", v0.abs().max().item())

                m.train()
                out = m.compute_loss(qpos, img, act, is_pad)
                assert torch.isfinite(out["loss"]), (dit_arch, state_dim, sdp, cdp)
                out["loss"].backward()

                m.eval()
                s = m.sample(qpos, img)
                assert s.shape == (B, m.num_queries, state_dim), s.shape
                assert torch.isfinite(s).all()
                # untrained -> _euler_sample returns ~N(0,1); the [-1,1]->[0,1] un-rescale
                # shifts the gripper channels to ~mean 0.5 (without it they'd sit near 0)
                g = s[..., m.gripper_idx]
                assert 0.2 < g.mean() < 0.8, ("gripper rescale not applied?", g.mean().item())
                out2 = m.compute_loss(qpos, img, act, is_pad)  # eval -> has sample_mse
                assert "sample_mse" in out2 and torch.isfinite(out2["sample_mse"])
                print(
                    f"ok  {dit_arch:10s} state_dim={state_dim:2d} "
                    f"sdp={sdp} cdp={cdp}  loss={out['loss'].item():.4f} "
                    f"sample_mse={out2['sample_mse'].item():.4f}"
                )
    print("all diffusion_flow self-tests passed")
