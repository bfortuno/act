"""Record scripted sim demonstrations straight into a LeRobot dataset v3 (no HDF5).

Same rollout and same stored quantities as the repo-root
``record_sim_episodes.py --task_space`` + ``convert_dataset.py`` pair, in one pass:

* ``observation.state``  achieved EE pose  [xyz, quat_wxyz, grip] x2  (16-dim canonical)
* ``action``             commanded EE target (scripted policy output)   (16-dim canonical)
* ``observation.images.<cam>``  mp4 video

Task-space only. The rollout runs in this project's mujoco (see README sim-fidelity
caveat), i.e. the same simulator ``eval_sim.py`` evaluates in.

Usage::

    cd lerobot_bridge
    MUJOCO_GL=egl uv run --group sim python record_dataset.py \
        --task-name sim_transfer_cube_scripted --out-dir outputs/data/ts_cube \
        --num-episodes 400 --seed 0
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import _shared  # noqa: F401  (repo root on sys.path)
import ee_transforms
from _sim_compat import make_ee_sim_env
from constants import CAMERA_HEIGHT, CAMERA_WIDTH, DT, SIM_TASK_CONFIGS
from convert_dataset import build_features
from scripted_policy import InsertionPolicy, PickAndTransferPolicy

POLICY_CLS = {
    "sim_transfer_cube_scripted": PickAndTransferPolicy,
    "sim_insertion_scripted": InsertionPolicy,
}
# with --only-success, give up after this many rollouts per requested episode
MAX_ATTEMPTS_FACTOR = 10


# strict success: max reward on every one of the last HOLD_STEPS steps (0.5 s at 50 Hz)
HOLD_STEPS = 25


def held_to_end(rewards, max_reward) -> bool:
    """Strict success: the task's final stage still holds over the last ``HOLD_STEPS``.

    The env reward only says the final stage (e.g. cube held by the receiving gripper,
    off the table) is true *right now*; ``max(rewards) == max_reward`` therefore also
    accepts episodes where the object is dropped afterwards. Requiring it from the first
    time it is reached would be too strict: the contact-based reward flickers for a few
    steps around the handover even in clean episodes, but not once the object is held.
    """
    return bool((np.asarray(rewards)[-HOLD_STEPS:] == max_reward).all())


def record(
    task_name: str,
    out_dir: Path,
    num_episodes: int,
    repo_id: str,
    seed: int,
    camera_height: int | None,
    camera_width: int | None,
    only_success: bool,
    overwrite: bool,
) -> dict:
    if task_name not in POLICY_CLS:
        raise NotImplementedError(f"no scripted policy for {task_name}")
    task_config = SIM_TASK_CONFIGS[task_name]
    episode_len = task_config["episode_len"]
    camera_names = list(task_config["camera_names"])
    # resolution priority: CLI flag > per-task config > global default
    cam_h = camera_height or task_config.get("camera_height", CAMERA_HEIGHT)
    cam_w = camera_width or task_config.get("camera_width", CAMERA_WIDTH)
    fps = round(1.0 / DT)

    info = {
        "state_dim": ee_transforms.CANONICAL_DIM,
        "action_dim": ee_transforms.CANONICAL_DIM,
        "camera_names": camera_names,
        "image_hw": (cam_h, cam_w),
        "has_qvel": False,
        "task_space": True,
        "rot_repr": "quat",
        "sim": True,
        "fps": fps,
        "task_name": task_name,
        "source_dir": None,  # recorded directly: norm stats come from the dataset itself
        "seed": seed,
    }

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists (pass --overwrite)")
        shutil.rmtree(out_dir)

    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=build_features(info), root=out_dir, use_videos=True
    )

    # object poses (utils.sample_*_pose) draw from the global numpy RNG
    np.random.seed(seed)
    env = make_ee_sim_env(task_name, camera_height=cam_h, camera_width=cam_w)
    max_reward = env.task.max_reward

    episode_lens: list[int] = []
    success: list[int] = []
    attempts = 0
    while len(episode_lens) < num_episodes:
        if attempts >= num_episodes * MAX_ATTEMPTS_FACTOR:
            raise RuntimeError(
                f"only {len(episode_lens)}/{num_episodes} successful episodes after {attempts} rollouts"
            )
        attempts += 1
        t0 = time.time()
        ts = env.reset()
        policy = POLICY_CLS[task_name](False)  # inject_noise=False, as record_sim_episodes.py
        rewards = []
        for _step in range(episode_len):
            action = np.asarray(policy(ts))
            # obs[t] is the observation the policy saw when it produced action[t]
            frame = {
                "observation.state": ee_transforms.canonicalize_quats(
                    ts.observation["ee_pose"]
                ).astype(np.float32),
                "action": ee_transforms.canonicalize_quats(action).astype(np.float32),
                "task": task_name,
            }
            for cam in camera_names:
                frame[f"observation.images.{cam}"] = ts.observation["images"][cam]
            ds.add_frame(frame)
            ts = env.step(action)
            rewards.append(ts.reward)
        t_roll = time.time() - t0

        ok = held_to_end(rewards, max_reward)
        if only_success and not ok:
            ds.clear_episode_buffer()
            print(f"  rollout {attempts}: Failed, discarded  (rollout {t_roll:.1f}s)")
            continue
        t0 = time.time()
        ds.save_episode()
        episode_lens.append(episode_len)
        success.append(int(ok))
        print(
            f"  episode {len(episode_lens) - 1}: {'Successful' if ok else 'Failed'}"
            f"  (rollout {t_roll:.1f}s, save {time.time() - t0:.1f}s)"
        )

    ds.finalize()

    info["num_episodes"] = len(episode_lens)
    info["episode_lens"] = episode_lens
    info["total_frames"] = int(sum(episode_lens))
    info["success"] = success
    sidecar = out_dir / "meta" / "act_bridge.json"
    sidecar.write_text(json.dumps(info, indent=2))
    print(f"wrote {sidecar}")
    print(f"Success: {sum(success)} / {len(success)}  ({attempts} rollouts)")
    return info


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--task-name", required=True, choices=sorted(POLICY_CLS))
    p.add_argument("--out-dir", type=Path, required=True, help="output LeRobot dataset root")
    p.add_argument("--num-episodes", type=int, required=True, help="episodes to save")
    p.add_argument(
        "--repo-id", default=None, help="local dataset id (default: act_bridge/<out dir name>)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera-height", type=int, default=None)
    p.add_argument("--camera-width", type=int, default=None)
    p.add_argument(
        "--only-success",
        action="store_true",
        help="discard rollouts that fail or drop the object after succeeding, and keep going "
        "until --num-episodes are saved",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    record(
        args.task_name,
        args.out_dir,
        args.num_episodes,
        args.repo_id or f"act_bridge/{args.out_dir.name}",
        args.seed,
        args.camera_height,
        args.camera_width,
        args.only_success,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
