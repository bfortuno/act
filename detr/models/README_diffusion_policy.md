# DiffusionPolicy

The **original Diffusion Policy** (Chi et al., 2023, `real-stanford/diffusion_policy`)
wired into the ACT training / eval stack as `--policy_class DiffusionPolicy`, next to
`ACT` / `CNNMLP` / `DiffusionFlow`.

It exists as a **known-good generative-BC baseline** for this repo: a reference that
should train and roll out in the task-space `rot6d` / `relative` regime where the
`DiffusionFlow` (rectified-flow) policy currently scores 0/50, so the flow failure can be
localised by comparison.

| | |
|---|---|
| Model | `detr/models/diffusion_policy.py` — `DiffusionPolicyModel`, `build_diffusion_policy` |
| Wrapper | `policy.py` — `DiffusionPolicy` |
| Builder / optimizer | `detr/main.py` — `build_DiffusionPolicy_model_and_optimizer` |
| Self-test | `python -m detr.models.diffusion_policy` |

---

## Two denoisers, one policy class — `--dp_denoiser`

| `--dp_denoiser` | Denoiser | Obs conditioning |
|---|---|---|
| `unet` *(default)* | CNN 1-D temporal U-Net `ConditionalUnet1D` (vendored verbatim from `real-stanford/diffusion_policy`, MIT) | FiLM global vector: per-camera ResNet-18 features (spatial-softmax keypoints or global avg pool) ⊕ proprio |
| `dit` | the **exact** gr00t-style adaLN-zero DiT from `diffusion_flow.py` (`DiTBlock` / `AdaLN` / `TimestepEncoder`, imported — that file is untouched) | visual cross-attention tokens, identical to `DiffusionFlow` |

`unet` is the canonical, hyperparameter-robust Diffusion Policy. `dit` swaps **only** the
generative process (rectified-flow velocity + 4-step Euler → DDPM ε + ancestral sampling)
and the EMA relative to `DiffusionFlow`, so a working `dit` DDPM vs the broken `dit` flow
isolates the objective/sampler/EMA as the cause.

---

## Generative process

Standard DDPM (`diffusers==0.11.1`, `DDPMScheduler`):

- `num_train_timesteps = 100` (`--dp_num_train_timesteps`)
- `beta_schedule = "squaredcos_cap_v2"` (cosine) (`--dp_beta_schedule`)
- `prediction_type = "epsilon"` (`--dp_prediction_type {epsilon,sample}`)
- **`clip_sample = True`**, with actions pre-divided by `--dp_action_scale` (default `4.0`).
  The cosine schedule at only 100 train steps has terminal `alpha_bar ≈ 0`; reconstructing
  `x0` from `eps` then divides by `sqrt(alpha_bar)`, so any `eps` error is hugely amplified
  and DDPM ancestral sampling **diverges** (empirically to `1e5+`) unless `pred_x0` is
  clamped to `[-1, 1]` — this is why stock DP sets `clip_sample=True`. Stock DP min-max
  scales actions to `[-1, 1]`; this repo z-scores them, so `DiffusionPolicy` divides the
  (z-scored) action chunk by `action_scale ≈ 4σ` around the diffusion process and multiplies
  the sample back, keeping ~everything inside `[-1, 1]` so the clamp only ever touches real
  outliers. Verified: `clip_sample=False` → sample MSE `~8e4`; `clip_sample=True` +
  `action_scale` → sample MSE `< 1` at the same (under-)trained checkpoint.
- gripper channels are **not** separately rescaled (DDPM + the clamp handle the bimodal
  open/close fine).

Training, per sample:
```
x0        = expert action chunk / action_scale       (B, T, A)   (dataloader-normalized, then /~4sigma)
noise     ~ N(0, I)
t         ~ U{0, ..., 99}
x_t       = sqrt(a_bar_t) x0 + sqrt(1 - a_bar_t) noise      (scheduler.add_noise)
eps_hat   = denoiser(x_t, t, obs)
loss      = masked_MSE(eps_hat, noise)                       (is_pad steps excluded)
```

Inference — `--dp_inference_scheduler {ddim,ddpm}` + `--n_inference_steps` (reused ACT flag):
```
x = randn(B, T, A)
for t in scheduler.timesteps:                # ddpm: 100 steps; ddim: --n_inference_steps
    x = scheduler.step(denoiser(x, t, obs), t, x).prev_sample
return x                                     # (B, T, A) normalized action chunk
```
`ddpm` (or `n_inference_steps >= num_train_timesteps`) reuses the training scheduler;
`ddim` builds a `DDIMScheduler` from the same config with `eta = 0`. These two flags are
**not** frozen by `run_config.json` — they stay tunable at eval.

`compute_loss` returns `{"loss"}` in training and `{"loss", "sample_mse"}` in validation,
where `sample_mse` is the masked MSE between the **seeded** Euler/ancestral sample and the
ground-truth chunk. `train_bc` selects `policy_best` on `sample_mse` (already wired for
`DiffusionFlow`); the seeding (`torch.Generator().manual_seed(0)`) makes that selection
stable across epochs — unlike `DiffusionFlow`, whose `sample_mse` draws fresh noise each
validation.

---

## Weight EMA (with warmup)

`DiffusionPolicy` keeps `self.ema_model = deepcopy(self.model)` (a real `nn.Module`, so it
serializes into the checkpoint). `ema_step()` is called by `train_bc` after every optimizer
step; the decay **ramps from ~0** via a power schedule:
```
decay = 1 - (1 + step / ema_inv_gamma) ** (-ema_power)
decay = clip(decay, ema_min_value, ema_decay)      # ema_decay is the ceiling
```
`--ema_decay` (default `0.9999`, `0` disables), `--ema_power` (`0.75`), `--ema_inv_gamma`
(`1.0`), `--ema_min_value` (`0.0`). Ramp: ≈0.41 @ step 1, 0.84 @ 10, 0.97 @ 100, 0.994 @ 1e3.

This fixes the `DiffusionFlow` EMA bug: its decay is a fixed `0.9999` with no warmup, and
because `EpisodicDataset.__len__` is the *episode count* an "epoch" is only a few optimizer
steps, so `0.9999**steps` stays ≈0.5 for thousands of steps and val / selection / eval all
sample from a half-random EMA copy. Validation and eval sample from the EMA weights;
training uses the raw weights.

---

## CLI reference

Added by this policy (present in both `imitate_episodes.py` and `detr/main.py:get_args_parser`):

| Flag | Default | Meaning |
|---|---|---|
| `--dp_denoiser {unet,dit}` | `unet` | denoiser architecture |
| `--dp_down_dims D [D ...]` | `256 512 1024` | U-Net channel schedule (`unet`); depth 3 ⇒ `chunk_size % 4 == 0` |
| `--dp_kernel_size` | `5` | U-Net Conv1d kernel |
| `--dp_n_groups` | `8` | U-Net GroupNorm groups |
| `--dp_diffusion_step_embed_dim` | `128` | U-Net timestep embedding width |
| `--dp_obs_pool {spatial_softmax,avg}` | `spatial_softmax` | image-feature pooling (`unet`) |
| `--dp_num_kp` | `32` | spatial-softmax keypoints per camera |
| `--dp_num_train_timesteps` | `100` | DDPM training steps |
| `--dp_beta_schedule` | `squaredcos_cap_v2` | DDPM beta schedule |
| `--dp_prediction_type {epsilon,sample}` | `epsilon` | regression target |
| `--dp_action_scale` | `4.0` | divide the (z-scored) action chunk by this before diffusion so `clip_sample` at `[-1,1]` is valid |
| `--dp_inference_scheduler {ddim,ddpm}` | `ddim` | sampler (tunable at eval) |
| `--ema_power` | `0.75` | EMA warmup exponent |
| `--ema_inv_gamma` | `1.0` | EMA warmup inverse-gamma |
| `--ema_min_value` | `0.0` | EMA decay floor |

Reused from the ACT / DiffusionFlow CLI: `--chunk_size` (**required** — action horizon `T`),
`--n_inference_steps` (sampler steps; **override its default of 4**), `--ema_decay`,
`--hidden_dim` / `--dim_feedforward` / `--nheads` / `--dit_arch` / `--dit_layers` (`dit`
denoiser only), `--lr`, `--seed`, `--batch_size`, `--num_epochs`, `--task_name`,
`--dataset_dir`, `--num_episodes`, `--temporal_agg`, `--task_space`,
`--rot_repr {quat,rpy,rot6d}`, `--action_repr {absolute,delta,relative}`.

The CNN backbone keeps its own optimizer param-group at `lr_backbone = 1e-5`
(hardcoded in `imitate_episodes.main`, same as ACT).

---

## Usage (task-space, rot6d + relative — the target regime)

Headless first: `source wsl_gl.sh headless`.

### Train — CNN U-Net (Stage 1)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted --dataset_dir /path/to/data --num_episodes 400 \
  --policy_class DiffusionPolicy --dp_denoiser unet \
  --task_space --rot_repr rot6d --action_repr relative --chunk_size 32 \
  --batch_size 64 --lr 1e-4 --seed 0 --num_epochs 3000 \
  --dp_num_train_timesteps 100 --n_inference_steps 100 --dp_inference_scheduler ddpm \
  --ema_decay 0.9999 --ckpt_dir ckpts/dp_unet_ts_rot6d_rel
```

### Train — gr00t DiT + DDPM (Stage 2)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted --dataset_dir /path/to/data --num_episodes 400 \
  --policy_class DiffusionPolicy --dp_denoiser dit --dit_arch cross_attn --dit_layers 8 \
  --task_space --rot_repr rot6d --action_repr relative --chunk_size 32 \
  --hidden_dim 512 --dim_feedforward 2048 --nheads 8 \
  --batch_size 64 --lr 1e-4 --seed 0 --num_epochs 3000 \
  --dp_num_train_timesteps 100 --n_inference_steps 100 --dp_inference_scheduler ddpm \
  --ema_decay 0.9999 --ckpt_dir ckpts/dp_dit_ts_rot6d_rel
```

### Evaluate

```bash
uv run python imitate_episodes.py --eval \
  --task_name sim_transfer_cube_scripted --dataset_dir /path/to/data --num_episodes 400 \
  --policy_class DiffusionPolicy --dp_denoiser unet \
  --task_space --rot_repr rot6d --action_repr relative --chunk_size 32 \
  --n_inference_steps 100 --dp_inference_scheduler ddpm --num_epochs 1 --seed 0 \
  --batch_size 64 --lr 1e-4 --ckpt_dir ckpts/dp_unet_ts_rot6d_rel   # optionally --temporal_agg
```

Architecture (`denoiser`, `down_dims`, `kernel_size`, `num_train_timesteps`,
`prediction_type`, `obs_pool`, `num_kp`, dims, `state/action_dim`, `num_queries`,
`ema_decay`) is restored from `<ckpt_dir>/run_config.json`; only `--n_inference_steps` /
`--dp_inference_scheduler` remain tunable at eval. Task-space eval uses
`eval_bc_task_space` (de-normalize → `ee_transforms.invert_action_chunk` → SO(3) temporal
agg → `ee_sim_env`); joint-space uses the ACT chunking path in `eval_bc`.

Watch `<ckpt_dir>/train_val_sample_mse_seed_0.png` (primary) and
`train_val_loss_seed_0.png` (ε-MSE). Rollout success is printed and written to
`result_policy_best.txt`; videos to `video*.mp4`.

---

## Integration notes / contract

`DiffusionPolicy` matches the `ACTPolicy` interface:

- `__call__(qpos, image, actions=None, is_pad=None)` — ImageNet-normalizes `image`
  `(B, n_cam, 3, H, W)` internally; training returns a dict with a scalar `loss`;
  inference returns `(B, T, A)`.
- `configure_optimizers()` returns the stored `AdamW`; `ema_step()` is called by `train_bc`
  after each optimizer step.
- `state_dict` carries `model.*` and (unless `--ema_decay 0`) `ema_model.*` plus
  `ema_step_count`.

Consumes the dataloader output as-is (actions arrive normalized, per-chunk-step stats for
`delta` / `relative`). Nothing in `utils.py` / `constants.py` / `diffusion_flow.py` was
changed.

## Provenance

- `ConditionalUnet1D`, `ConditionalResidualBlock1D`, `Conv1dBlock`, `Downsample1d`,
  `Upsample1d`, `SinusoidalPosEmb` — vendored verbatim from **`real-stanford/diffusion_policy`**
  (`diffusion_policy/model/diffusion/{conditional_unet1d,conv1d_components,positional_embedding}.py`,
  MIT).
- `DDPMScheduler` / `DDIMScheduler` — `diffusers==0.11.1` (the pin from the DP repo).
- `DiTBlock` / `AdaLN` / `TimestepEncoder` for `--dp_denoiser dit` — imported from
  `detr/models/diffusion_flow.py` (gr00t-n1d7 lineage, Apache-2.0).
- The CNN backbone, sine positional embedding and the training/eval harness are ACT / DETR.
