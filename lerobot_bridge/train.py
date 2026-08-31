"""Loop 3-4: thin trainer that drives upstream LeRobot ACT / Diffusion Policy on the
rot6d / relative (or any) task-space representation.

State + action reach the policy already z-scored (per-chunk-step for delta/relative) and
images already ImageNet-normalized by ``EEReprDataset``; the policy is therefore
configured with IDENTITY normalization and LeRobot's processor pipeline is bypassed.

    cd lerobot_bridge
    uv run python train.py --dataset-root _fixtures/ts --out-dir outputs/smoke_act \
        --policy act --rot-repr rot6d --action-repr relative --chunk-size 32 \
        --steps 100 --batch-size 8 --device cuda
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from torch.utils.data import DataLoader

import _shared  # noqa: F401
from ee_repr_dataset import EEReprDataset, stats_filename


def build_datasets(args):
    full = EEReprDataset(
        args.dataset_root,
        rot_repr=args.rot_repr,
        action_repr=args.action_repr,
        chunk_size=args.chunk_size,
    )
    n_ep = full.info["num_episodes"]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_ep)
    n_val = max(1, int(round(args.val_fraction * n_ep)))
    val_eps = sorted(perm[:n_val].tolist())
    train_eps = sorted(perm[n_val:].tolist())

    common = dict(rot_repr=args.rot_repr, action_repr=args.action_repr, chunk_size=args.chunk_size)
    train_ds = EEReprDataset(args.dataset_root, episodes=train_eps, **common)
    val_ds = EEReprDataset(args.dataset_root, episodes=val_eps, **common)
    return full, train_ds, val_ds, train_eps, val_eps


def _hw(full: EEReprDataset) -> tuple[int, int]:
    cam0 = full.camera_keys[0]
    shape = full.ds.meta.features[cam0]["shape"]  # (H, W, C)
    return int(shape[0]), int(shape[1])


def make_features(full: EEReprDataset):
    h, w = _hw(full)
    inp = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(full.state_dim,)),
    }
    for cam in full.camera_keys:
        inp[cam] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))
    out = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(full.action_dim,))}
    return inp, out


IDENTITY_MAP = {
    "VISUAL": NormalizationMode.IDENTITY,
    "STATE": NormalizationMode.IDENTITY,
    "ACTION": NormalizationMode.IDENTITY,
}


def build_policy(args, full: EEReprDataset):
    inp, out = make_features(full)
    if args.policy == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        cfg = ACTConfig(
            input_features=inp,
            output_features=out,
            normalization_mapping=IDENTITY_MAP,
            chunk_size=args.chunk_size,
            n_action_steps=args.chunk_size,
            n_obs_steps=1,
            dim_model=args.dim_model,
            n_heads=args.n_heads,
            dim_feedforward=args.dim_feedforward,
            n_encoder_layers=args.enc_layers,
            n_decoder_layers=args.dec_layers,
            use_vae=not args.no_vae,
            kl_weight=args.kl_weight,
            dropout=args.dropout,
            optimizer_lr=args.lr,
            optimizer_lr_backbone=args.lr_backbone,
            device=args.device,
        )
        return cfg, ACTPolicy(cfg)

    if args.policy == "diffusion":
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        cfg = DiffusionConfig(
            input_features=inp,
            output_features=out,
            normalization_mapping=IDENTITY_MAP,
            horizon=args.chunk_size,
            n_action_steps=args.chunk_size,
            n_obs_steps=1,
            num_train_timesteps=args.dp_train_timesteps,
            beta_schedule=args.dp_beta_schedule,
            prediction_type=args.dp_prediction_type,
            clip_sample=True,
            optimizer_lr=args.lr,
            device=args.device,
        )
        return cfg, DiffusionPolicy(cfg)

    raise ValueError(args.policy)


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def adapt_batch(batch: dict, policy_kind: str) -> dict:
    """Diffusion's ``compute_loss`` wants ``observation.state`` as (B, n_obs_steps, D);
    EEReprDataset emits a single obs so we add the time axis. ACT wants it as (B, D)."""
    if policy_kind == "diffusion":
        s = batch["observation.state"]
        if s.ndim == 2:
            batch = {**batch, "observation.state": s.unsqueeze(1)}
    return batch


@torch.no_grad()
def evaluate(policy, loader, device, policy_kind, max_batches: int | None = None) -> float:
    policy.eval()
    losses = []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = adapt_batch(to_device(batch, device), policy_kind)
        loss, _ = policy.forward(batch)
        losses.append(loss.item())
    policy.train()
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--policy", choices=["act", "diffusion"], default="act")
    p.add_argument("--rot-repr", default="rot6d")
    p.add_argument("--action-repr", default="relative")
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--wandb", action="store_true")
    # ACT arch
    p.add_argument("--dim-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--dim-feedforward", type=int, default=3200)
    p.add_argument("--enc-layers", type=int, default=4)
    p.add_argument("--dec-layers", type=int, default=1)
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-vae", action="store_true")
    # Diffusion
    p.add_argument("--dp-train-timesteps", type=int, default=100)
    p.add_argument("--dp-beta-schedule", default="squaredcos_cap_v2")
    p.add_argument("--dp-prediction-type", default="epsilon")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    full, train_ds, val_ds, train_eps, val_eps = build_datasets(args)
    print(
        f"episodes: {len(train_eps)} train / {len(val_eps)} val   frames: {len(train_ds)} / {len(val_ds)}"
    )
    print(f"state_dim={full.state_dim} cameras={full.camera_keys}")

    cfg, policy = build_policy(args, full)
    policy.to(device).train()
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"policy={args.policy}  trainable params={n_params / 1e6:.1f}M")

    optim = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=args.lr,
        weight_decay=getattr(cfg, "optimizer_weight_decay", 1e-4) or 1e-4,
    )

    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **{**dl_kwargs, "drop_last": False})

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="lerobot-bridge", name=args.out_dir.name, config=vars(args))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    step = 0
    for batch in itertools.cycle(train_loader):
        step += 1
        if step > args.steps:
            break
        batch = adapt_batch(to_device(batch, device), args.policy)
        loss, loss_dict = policy.forward(batch)
        loss_dict = loss_dict or {}
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optim.step()

        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            print(f"step {step:6d}  loss {loss.item():.4f}  {loss_dict}  {sps:.1f} it/s")
            if run:
                run.log(
                    {"train/loss": loss.item(), **{f"train/{k}": v for k, v in loss_dict.items()}},
                    step=step,
                )

        if step % args.val_every == 0 or step == args.steps:
            vloss = evaluate(policy, val_loader, device, args.policy, args.val_batches)
            print(f"  [val] step {step}  loss {vloss:.4f}  (best {best_val:.4f})")
            if run:
                run.log({"val/loss": vloss}, step=step)
            if vloss < best_val:
                best_val = vloss
                _save(
                    policy,
                    args,
                    cfg,
                    train_eps,
                    val_eps,
                    full,
                    tag="best",
                    val_loss=vloss,
                    step=step,
                )

    _save(policy, args, cfg, train_eps, val_eps, full, tag="last", val_loss=best_val, step=step)
    if run:
        run.finish()
    print(f"done. best val loss {best_val:.4f}  ->  {args.out_dir}")


def _save(
    policy, args, cfg, train_eps, val_eps, full, *, tag: str, val_loss: float, step: int
) -> None:
    ckpt_dir = args.out_dir / tag
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    policy.save_pretrained(ckpt_dir)
    src_stats = (
        args.dataset_root
        / "meta"
        / stats_filename(args.rot_repr, args.action_repr, args.chunk_size)
    )
    shutil.copy(src_stats, ckpt_dir / src_stats.name)
    (ckpt_dir / "run_config.json").write_text(
        json.dumps(
            {
                "policy": args.policy,
                "rot_repr": args.rot_repr,
                "action_repr": args.action_repr,
                "chunk_size": args.chunk_size,
                "state_dim": full.state_dim,
                "camera_keys": full.camera_keys,
                "task_space": full.task_space,
                "dataset_root": str(args.dataset_root.resolve()),
                "task_name": full.info["task_name"],
                "fps": full.info["fps"],
                "train_episodes": train_eps,
                "val_episodes": val_eps,
                "val_loss": val_loss,
                "step": step,
                "imagenet_normalize": True,
            },
            indent=2,
        )
    )
    print(f"  saved {tag} -> {ckpt_dir}")


if __name__ == "__main__":
    main()
