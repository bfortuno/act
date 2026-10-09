"""Loop 2: representation wrapper + normalization stats.

The converted LeRobot dataset stores canonical raw poses. LeRobot policies here are
trained on the same ``(rot_repr, action_repr)`` features as the in-repo
``utils.EpisodicDataset``:

* ``transform_state`` / ``transform_action_chunk`` from the repo-root ``ee_transforms``
* z-score with stats from the repo-root ``utils.get_norm_stats`` (per-chunk-step for
  ``delta`` / ``relative``; rotation + gripper channels pass through unnormalized)

State and action come out **already normalized**; the policy is configured with IDENTITY
normalization for STATE / ACTION (LeRobot's ``Normalize`` has no per-chunk-step mode).
Images are returned raw (CHW float [0, 1]) for the policy to ImageNet-normalize.

Frame-indexed (one item per frame), unlike ``EpisodicDataset`` which is episode-indexed
with a random ``start_ts`` - this gives ~``episode_len`` x more optimizer steps/epoch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import _shared  # noqa: F401  (repo root on sys.path)
import ee_transforms
from utils import get_norm_stats, task_space_norm_stats_from_arrays

_STAT_KEYS = ("qpos_mean", "qpos_std", "action_mean", "action_std")

# ImageNet stats - the in-repo detr/ACT policy normalizes camera frames with these
# (policy.py). LeRobot's ACT model does NOT normalize internally, so we do it here and
# set the policy's VISUAL normalization to IDENTITY.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def stats_filename(rot_repr: str, action_repr: str, chunk_size: int) -> str:
    return f"ee_repr_stats_{rot_repr}_{action_repr}_c{chunk_size}.npz"


def load_canonical_episodes(dataset_root: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-episode ``(state, action)`` arrays read straight from the dataset's parquet
    files (no video decode), in episode order."""
    cols = ["episode_index", "frame_index", "observation.state", "action"]
    files = sorted((dataset_root / "data").glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    episodes = []
    for _, ep in df.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
        state = np.stack(ep["observation.state"].to_numpy())
        action = np.stack(ep["action"].to_numpy())
        episodes.append((state, action))
    return episodes


def compute_ee_repr_stats(
    dataset_root: Path,
    *,
    rot_repr: str = "rot6d",
    action_repr: str = "relative",
    chunk_size: int = 32,
    overwrite: bool = False,
) -> dict:
    """Compute the repo-root norm stats and cache the arrays next to the dataset.

    Converted datasets run ``get_norm_stats`` against the ORIGINAL HDF5 dir (from the
    sidecar). Directly recorded datasets (``record_dataset.py``, no ``source_dir``) feed
    the same math from the dataset's own parquet columns."""
    info = json.loads((dataset_root / "meta" / "act_bridge.json").read_text())
    out = dataset_root / "meta" / stats_filename(rot_repr, action_repr, chunk_size)
    if out.exists() and not overwrite:
        return load_ee_repr_stats(dataset_root, rot_repr, action_repr, chunk_size)

    if info.get("source_dir"):
        stats = get_norm_stats(
            info["source_dir"],
            info["num_episodes"],
            task_space=info["task_space"],
            action_repr=action_repr,
            rot_repr=rot_repr,
            chunk_size=chunk_size,
        )
    else:
        if not info["task_space"]:
            raise NotImplementedError("stats without a source HDF5 dir are task-space only")
        stats = task_space_norm_stats_from_arrays(
            load_canonical_episodes(dataset_root), action_repr, rot_repr, chunk_size
        )
    payload = {k: np.asarray(stats[k], dtype=np.float32) for k in _STAT_KEYS}
    payload["meta"] = np.array(
        json.dumps(
            {
                "task_space": info["task_space"],
                "rot_repr": rot_repr,
                "action_repr": action_repr,
                "chunk_size": chunk_size,
            }
        )
    )
    np.savez(out, **payload)
    print(
        f"wrote {out}  qpos_mean{payload['qpos_mean'].shape}  action_mean{payload['action_mean'].shape}"
    )
    return load_ee_repr_stats(dataset_root, rot_repr, action_repr, chunk_size)


def load_ee_repr_stats(
    dataset_root: Path, rot_repr: str, action_repr: str, chunk_size: int
) -> dict:
    npz = np.load(dataset_root / "meta" / stats_filename(rot_repr, action_repr, chunk_size))
    d = {k: npz[k].astype(np.float32) for k in _STAT_KEYS}
    d.update(json.loads(str(npz["meta"])))
    return d


class EEReprDataset(torch.utils.data.Dataset):
    """Wraps a ``LeRobotDataset`` (with an ``action`` delta-timestamp window) and applies
    the ``(rot_repr, action_repr)`` transform + normalization per frame."""

    def __init__(
        self,
        dataset_root: Path | str,
        *,
        repo_id: str | None = None,
        rot_repr: str = "rot6d",
        action_repr: str = "relative",
        chunk_size: int = 32,
        episodes: list[int] | None = None,
        image_transforms=None,
        imagenet_normalize: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.imagenet_normalize = imagenet_normalize
        self.info = json.loads((self.dataset_root / "meta" / "act_bridge.json").read_text())
        self.task_space = bool(self.info["task_space"])
        self.rot_repr = rot_repr
        self.action_repr = action_repr
        self.chunk_size = chunk_size

        repo_id = repo_id or f"act_bridge/{self.dataset_root.name}"
        fps = self.info["fps"]
        delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
        self.ds = LeRobotDataset(
            repo_id,
            root=self.dataset_root,
            delta_timestamps=delta_timestamps,
            episodes=episodes,
            image_transforms=image_transforms,
        )
        self.camera_keys = [k for k in self.ds.meta.features if k.startswith("observation.images.")]

        self.stats = compute_ee_repr_stats(
            self.dataset_root,
            rot_repr=rot_repr,
            action_repr=action_repr,
            chunk_size=chunk_size,
        )
        self._qpos_mean = self.stats["qpos_mean"]
        self._qpos_std = self.stats["qpos_std"]
        self._action_mean = self.stats["action_mean"]
        self._action_std = self.stats["action_std"]

        self.state_dim = int(self._qpos_mean.shape[-1])
        self.action_dim = self.state_dim

    def __len__(self) -> int:
        return len(self.ds)

    def _norm_action(self, act: np.ndarray) -> np.ndarray:
        mean, std = self._action_mean, self._action_std
        if mean.ndim == 2:  # (chunk, D) per-chunk-step stats
            idx = np.minimum(np.arange(act.shape[0]), mean.shape[0] - 1)
            mean, std = mean[idx], std[idx]
        return (act - mean) / std

    def __getitem__(self, idx: int) -> dict:
        raw = self.ds[idx]
        state = np.asarray(raw["observation.state"], dtype=np.float64)  # (C,) canonical
        action = np.asarray(raw["action"], dtype=np.float64)  # (chunk, C) canonical
        is_pad = torch.as_tensor(raw["action_is_pad"], dtype=torch.bool)

        if self.task_space:
            ref = state
            qpos_feat = ee_transforms.transform_state(state, self.rot_repr)
            act_feat = ee_transforms.transform_action_chunk(
                action, ref, self.action_repr, self.rot_repr
            )
        else:
            qpos_feat = state
            act_feat = action

        qpos_feat = (qpos_feat - self._qpos_mean) / self._qpos_std
        act_feat = self._norm_action(act_feat)

        out = {
            "observation.state": torch.from_numpy(qpos_feat.astype(np.float32)),
            "action": torch.from_numpy(act_feat.astype(np.float32)),
            "action_is_pad": is_pad,
        }
        for cam in self.camera_keys:
            img = (
                raw[cam].float()
                if torch.is_tensor(raw[cam])
                else torch.as_tensor(raw[cam], dtype=torch.float32)
            )
            if self.imagenet_normalize:
                img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
            out[cam] = img
        return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--rot-repr", default="rot6d")
    p.add_argument("--action-repr", default="relative")
    p.add_argument("--chunk-size", type=int, default=32)
    a = p.parse_args()
    ds = EEReprDataset(
        a.dataset_root,
        rot_repr=a.rot_repr,
        action_repr=a.action_repr,
        chunk_size=a.chunk_size,
    )
    s = ds[0]
    print("len", len(ds), "state_dim", ds.state_dim)
    for k, v in s.items():
        print(f"  {k:26s} {tuple(v.shape)} {v.dtype}")
