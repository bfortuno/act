"""Loop 5: roll out a trained LeRobot ACT / Diffusion policy in ``ee_sim_env``.

Mirrors ``imitate_episodes.py:eval_bc_task_space`` - denormalize the policy's chunk,
``ee_transforms.invert_action_chunk`` back to canonical absolute EE poses, optional
SO(3) temporal aggregation, then ``env.step``. Writes the same artifact layout
(``result_*.txt``, ``video*.mp4``) into the checkpoint dir.

    cd lerobot_bridge
    MUJOCO_GL=egl uv run --group sim python eval_sim.py \
        --ckpt-dir outputs/smoke_act/best --num-rollouts 2

NOTE: the bridge runs a current mujoco/dm-control (no cp312 wheel for the root repo's
mujoco 2.3.7); sim dynamics may differ slightly from the in-repo baselines. For an
exact comparison, evaluate the same checkpoint through a two-process bridge against the
root py3.8 env (see README).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _shared  # noqa: F401
import ee_transforms
from _eval_utils import aggregate_abs_poses, exp_weights
from _sim_compat import make_ee_sim_env
from constants import SIM_TASK_CONFIGS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_policy(ckpt_dir: Path, kind: str, device, dp_inference_steps: int | None):
    if kind == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy

        policy = ACTPolicy.from_pretrained(ckpt_dir)
    elif kind == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        policy = DiffusionPolicy.from_pretrained(ckpt_dir)
        if dp_inference_steps is not None:
            policy.config.num_inference_steps = dp_inference_steps
    else:
        raise ValueError(kind)
    return policy.to(device).eval()


def load_stats(ckpt_dir: Path, rot_repr: str, action_repr: str, chunk: int) -> dict:
    from ee_repr_dataset import stats_filename

    npz = np.load(ckpt_dir / stats_filename(rot_repr, action_repr, chunk))
    return {
        k: npz[k].astype(np.float32) for k in ("qpos_mean", "qpos_std", "action_mean", "action_std")
    }


def denorm_chunk(norm_chunk: np.ndarray, stats: dict) -> np.ndarray:
    am, as_ = stats["action_mean"], stats["action_std"]
    if am.ndim == 2:
        idx = np.minimum(np.arange(norm_chunk.shape[0]), am.shape[0] - 1)
        am, as_ = am[idx], as_[idx]
    return norm_chunk * as_ + am


def make_batch(ee_pose16, images_hwc, cfg, stats, device):
    state = ee_transforms.transform_state(ee_pose16, cfg["rot_repr"]).astype(np.float32)
    state = (state - stats["qpos_mean"]) / stats["qpos_std"]
    batch = {"observation.state": torch.from_numpy(state).float().unsqueeze(0).to(device)}
    if cfg["policy"] == "diffusion":
        batch["observation.state"] = batch["observation.state"].unsqueeze(1)  # (1, 1, D)
    for cam_key in cfg["camera_keys"]:
        short = cam_key.split("observation.images.")[-1]
        img = images_hwc[short].astype(np.float32).transpose(2, 0, 1) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        batch[cam_key] = torch.from_numpy(img).float().unsqueeze(0).to(device)
    return batch


def save_video(frames_per_cam: list[dict], dt: float, path: Path) -> None:
    try:
        import imageio.v2 as imageio
    except Exception:
        print(f"  (imageio unavailable, skipping {path.name})")
        return
    cams = list(frames_per_cam[0].keys())
    stacked = [np.concatenate([f[c] for c in cams], axis=1) for f in frames_per_cam]
    imageio.mimsave(path, stacked, fps=int(round(1.0 / dt)))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--task-name", default=None, help="default: from run_config.json")
    ap.add_argument("--num-rollouts", type=int, default=50)
    ap.add_argument("--temporal-agg", action="store_true")
    ap.add_argument("--agg-k", type=float, default=0.01)
    ap.add_argument("--dp-inference-steps", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--max-videos", type=int, default=10)
    args = ap.parse_args()

    cfg = json.loads((args.ckpt_dir / "run_config.json").read_text())
    task_name = args.task_name or cfg["task_name"]
    rot_repr, action_repr, chunk = cfg["rot_repr"], cfg["action_repr"], cfg["chunk_size"]
    episode_len = SIM_TASK_CONFIGS[task_name]["episode_len"]
    device = torch.device(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = load_policy(args.ckpt_dir, cfg["policy"], device, args.dp_inference_steps)
    stats = load_stats(args.ckpt_dir, rot_repr, action_repr, chunk)
    env = make_ee_sim_env(task_name)
    env_max_reward = env.task.max_reward
    query_frequency = 1 if args.temporal_agg else chunk
    C = ee_transforms.CANONICAL_DIM

    highest_rewards, returns = [], []
    for rollout_id in range(args.num_rollouts):
        if hasattr(policy, "reset"):
            policy.reset()
        ts = env.reset()
        all_time = np.zeros([episode_len, episode_len + chunk, C]) if args.temporal_agg else None
        all_filled = (
            np.zeros([episode_len, episode_len + chunk], dtype=bool) if args.temporal_agg else None
        )
        abs_chunk = None
        rewards, frames = [], []
        for t in range(episode_len):
            obs = ts.observation
            ee_pose = np.asarray(obs["ee_pose"], dtype=np.float64)
            if not args.no_video and rollout_id < args.max_videos:
                frames.append({k: v.copy() for k, v in obs["images"].items()})

            if t % query_frequency == 0:
                batch = make_batch(ee_pose, obs["images"], cfg, stats, device)
                norm_chunk = policy.predict_action_chunk(batch)[0].float().cpu().numpy()
                abs_chunk = ee_transforms.invert_action_chunk(
                    denorm_chunk(norm_chunk, stats), ee_pose, action_repr, rot_repr
                )

            if args.temporal_agg:
                all_time[t, t : t + chunk] = abs_chunk
                all_filled[t, t : t + chunk] = True
                mask = all_filled[:, t]
                cands = all_time[mask, t]
                action16 = aggregate_abs_poses(cands, exp_weights(len(cands), args.agg_k))
            else:
                action16 = abs_chunk[t % query_frequency]

            ts = env.step(action16)
            rewards.append(ts.reward if ts.reward is not None else 0.0)

        rewards = np.asarray(rewards, dtype=np.float64)
        highest_rewards.append(rewards.max())
        returns.append(rewards.sum())
        ok = rewards.max() == env_max_reward
        print(
            f"rollout {rollout_id:3d}  return={rewards.sum():.0f}  max={rewards.max():.0f}/{env_max_reward}  success={ok}"
        )
        if frames:
            save_video(frames, 0.02, args.ckpt_dir / f"video{rollout_id}.mp4")

    highest = np.asarray(highest_rewards)
    success_rate = float(np.mean(highest == env_max_reward))
    summary = [f"Success rate: {success_rate}", f"Average return: {float(np.mean(returns))}", ""]
    for r in range(int(env_max_reward) + 1):
        n = int((highest >= r).sum())
        summary.append(
            f"Reward >= {r}: {n}/{args.num_rollouts} = {100 * n / args.num_rollouts:.1f}%"
        )
    text = "\n".join(summary)
    print("\n" + text)

    tag = args.ckpt_dir.name
    (args.ckpt_dir / f"result_{tag}.txt").write_text(
        text + "\n\n" + repr(returns) + "\n" + repr(highest_rewards)
    )
    (args.ckpt_dir / "eval_result.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "policy": cfg["policy"],
                "num_rollouts": args.num_rollouts,
                "temporal_agg": args.temporal_agg,
                "success_rate": success_rate,
                "avg_return": float(np.mean(returns)),
                "highest_rewards": [float(x) for x in highest_rewards],
                "mujoco_note": "bridge uses current mujoco/dm-control, not root repo's 2.3.7",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
