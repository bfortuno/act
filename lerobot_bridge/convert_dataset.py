"""Loop 1: convert the ACT-fork per-episode HDF5 dataset to LeRobot dataset v3.

The converted dataset stores **exactly what the HDF5 holds** - canonical raw poses,
no rot6d / relative transform (that happens later in ``ee_repr_dataset.py``). A sidecar
``meta/act_bridge.json`` carries the bits LeRobot's schema has no slot for
(task-space flag, rotation representation, source dir, per-episode lengths).

Usage::

    cd lerobot_bridge
    uv run python convert_dataset.py \
        --source-dir ../data/ts --out-dir _fixtures/ts \
        --task-name sim_transfer_cube_scripted --verify

    # task-space (16-dim) and joint-space (14-dim) HDF5 are both handled.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import _shared  # noqa: F401  (repo root on sys.path)
from constants import DT

CANONICAL_STATE_NAMES_16 = [
    "l_x",
    "l_y",
    "l_z",
    "l_qw",
    "l_qx",
    "l_qy",
    "l_qz",
    "l_grip",
    "r_x",
    "r_y",
    "r_z",
    "r_qw",
    "r_qx",
    "r_qy",
    "r_qz",
    "r_grip",
]
JOINT_STATE_NAMES_14 = [
    "left_j0",
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_grip",
    "right_j0",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_grip",
]


def _state_names(dim: int) -> list[str]:
    if dim == 16:
        return list(CANONICAL_STATE_NAMES_16)
    if dim == 14:
        return list(JOINT_STATE_NAMES_14)
    return [f"s{i}" for i in range(dim)]


def _episode_paths(source_dir: Path, num_episodes: int | None) -> list[Path]:
    paths = []
    i = 0
    while (source_dir / f"episode_{i}.hdf5").exists():
        paths.append(source_dir / f"episode_{i}.hdf5")
        i += 1
    if not paths:
        raise FileNotFoundError(f"no episode_*.hdf5 under {source_dir}")
    if num_episodes is not None:
        paths = paths[:num_episodes]
    return paths


def _probe(first_hdf5: Path) -> dict:
    with h5py.File(first_hdf5, "r") as f:
        attrs = dict(f.attrs)
        state_dim = int(f["/observations/qpos"].shape[1])
        action_dim = int(f["/action"].shape[1])
        cams = sorted(f["/observations/images"].keys())
        h, w = f[f"/observations/images/{cams[0]}"].shape[1:3]
        has_qvel = "/observations/qvel" in f
    return {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "camera_names": cams,
        "image_hw": (int(h), int(w)),
        "has_qvel": has_qvel,
        "task_space": bool(attrs.get("task_space", False)),
        "rot_repr": str(attrs.get("rot_repr", "quat")),
        "sim": bool(attrs.get("sim", True)),
    }


def build_features(info: dict) -> dict:
    h, w = info["image_hw"]
    feats: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (info["state_dim"],),
            "names": _state_names(info["state_dim"]),
        },
        "action": {
            "dtype": "float32",
            "shape": (info["action_dim"],),
            "names": _state_names(info["action_dim"]),
        },
    }
    if info["has_qvel"]:
        feats["observation.qvel"] = {
            "dtype": "float32",
            "shape": (info["state_dim"],),
            "names": _state_names(info["state_dim"]),
        }
    for cam in info["camera_names"]:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return feats


def convert(
    source_dir: Path,
    out_dir: Path,
    task_name: str,
    num_episodes: int | None,
    repo_id: str,
    fps: int,
    overwrite: bool,
) -> dict:
    ep_paths = _episode_paths(source_dir, num_episodes)
    info = _probe(ep_paths[0])
    info["num_episodes"] = len(ep_paths)
    info["fps"] = fps
    info["task_name"] = task_name
    info["source_dir"] = str(source_dir.resolve())

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists (pass --overwrite)")
        shutil.rmtree(out_dir)

    features = build_features(info)
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features, root=out_dir, use_videos=True
    )

    episode_lens: list[int] = []
    for ep_path in ep_paths:
        with h5py.File(ep_path, "r") as f:
            qpos = f["/observations/qpos"][()]
            action = f["/action"][()]
            qvel = f["/observations/qvel"][()] if info["has_qvel"] else None
            imgs = {c: f[f"/observations/images/{c}"][()] for c in info["camera_names"]}
        t_len = int(action.shape[0])
        episode_lens.append(t_len)
        for t in range(t_len):
            frame = {
                "observation.state": qpos[t].astype(np.float32),
                "action": action[t].astype(np.float32),
                "task": task_name,
            }
            if qvel is not None:
                frame["observation.qvel"] = qvel[t].astype(np.float32)
            for cam in info["camera_names"]:
                frame[f"observation.images.{cam}"] = imgs[cam][t]
            ds.add_frame(frame)
        ds.save_episode()
        print(f"  {ep_path.name}: {t_len} frames")

    ds.finalize()

    info["episode_lens"] = episode_lens
    info["total_frames"] = int(sum(episode_lens))
    sidecar = out_dir / "meta" / "act_bridge.json"
    sidecar.write_text(json.dumps(info, indent=2))
    print(f"wrote {sidecar}")
    return info


def verify(out_dir: Path, source_dir: Path, repo_id: str) -> None:
    info = json.loads((out_dir / "meta" / "act_bridge.json").read_text())
    ds = LeRobotDataset(repo_id, root=out_dir)
    assert len(ds) == info["total_frames"], (len(ds), info["total_frames"])

    # frame 0 of episode 0: state must match the HDF5 qpos exactly
    with h5py.File(source_dir / "episode_0.hdf5", "r") as f:
        hdf5_qpos0 = f["/observations/qpos"][0].astype(np.float32)
        cam0 = info["camera_names"][0]
        hdf5_img0 = f[f"/observations/images/{cam0}"][0]  # (H, W, 3) uint8

    s0 = np.asarray(ds[0]["observation.state"], dtype=np.float32)
    err = float(np.abs(s0 - hdf5_qpos0).max())
    assert err < 1e-5, f"state mismatch {err}"

    img = np.asarray(ds[0][f"observation.images.{cam0}"])  # (3, H, W) float [0,1]
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img_u8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    mad = float(np.abs(img_u8.astype(np.int16) - hdf5_img0.astype(np.int16)).mean())
    assert mad < 2.0, f"image mean-abs-diff too high after mp4 round-trip: {mad}"

    print(f"verify OK  frames={len(ds)}  state_err={err:.2e}  img_mad={mad:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source-dir", type=Path, required=True, help="dir with episode_*.hdf5")
    p.add_argument("--out-dir", type=Path, required=True, help="output LeRobot dataset root")
    p.add_argument("--task-name", required=True, help="sim task / natural-language task string")
    p.add_argument("--num-episodes", type=int, default=None)
    p.add_argument(
        "--repo-id", default=None, help="local dataset id (default: act_bridge/<out dir name>)"
    )
    p.add_argument("--fps", type=int, default=round(1.0 / DT))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify", action="store_true", help="reload and check round-trip fidelity")
    args = p.parse_args()

    repo_id = args.repo_id or f"act_bridge/{args.out_dir.name}"
    convert(
        args.source_dir,
        args.out_dir,
        args.task_name,
        args.num_episodes,
        repo_id,
        args.fps,
        args.overwrite,
    )
    if args.verify:
        verify(args.out_dir, args.source_dir, repo_id)


if __name__ == "__main__":
    main()
