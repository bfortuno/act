"""Loop 2 exit test: EEReprDataset parity with a direct ee_transforms + stats
computation from the source HDF5, plus action-chunk invertibility.

    cd lerobot_bridge && uv run python test_loop2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

import _shared  # noqa: F401
import ee_transforms
from ee_repr_dataset import EEReprDataset, load_ee_repr_stats


def _unit_quat_canonical(vec16: np.ndarray) -> np.ndarray:
    """Unit-normalize the wxyz quat sub-block of each arm in a canonical (..., 16) array.

    Some raw recorded action quaternions have norm < 1 (scripted-policy slerp without
    renormalization). Any rot-repr round-trip recovers the *normalized* rotation, so
    invertibility must be checked against unit-normalized quats.
    """
    out = np.array(vec16, dtype=np.float64)
    for base in (3, 11):  # [xyz(3) | quat_wxyz(4) | grip(1)] per arm
        q = out[..., base : base + 4]
        out[..., base : base + 4] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return out


def _ref_sample(hdf5_path: Path, start_ts: int, chunk: int, rot_repr, action_repr, stats):
    with h5py.File(hdf5_path, "r") as f:
        qpos = f["/observations/qpos"][start_ts].astype(np.float64)
        act = f["/action"][()].astype(np.float64)
    T = act.shape[0]
    end = min(start_ts + chunk, T)
    window = act[start_ts:end]
    if end - start_ts < chunk:  # LeRobot pads by repeating the last frame
        window = np.concatenate([window, np.repeat(window[-1:], chunk - (end - start_ts), 0)])

    qpos_feat = ee_transforms.transform_state(qpos, rot_repr)
    act_feat = ee_transforms.transform_action_chunk(window, qpos, action_repr, rot_repr)

    qpos_feat = (qpos_feat - stats["qpos_mean"]) / stats["qpos_std"]
    am, as_ = stats["action_mean"], stats["action_std"]
    if am.ndim == 2:
        idx = np.minimum(np.arange(chunk), am.shape[0] - 1)
        am, as_ = am[idx], as_[idx]
    act_feat = (act_feat - am) / as_
    return qpos_feat, act_feat, qpos, window, end - start_ts


def check_taskspace():
    root = Path("_fixtures/ts")
    rot_repr, action_repr, chunk = "rot6d", "relative", 32
    ds = EEReprDataset(root, rot_repr=rot_repr, action_repr=action_repr, chunk_size=chunk)
    stats = load_ee_repr_stats(root, rot_repr, action_repr, chunk)
    hdf5 = (
        Path(json.loads((root / "meta" / "act_bridge.json").read_text())["source_dir"])
        / "episode_0.hdf5"
    )

    for start_ts in (0, 10, 137, 395):  # 395 -> only 5 real steps, 27 padded
        item = ds[start_ts]  # episode 0 occupies indices 0..399
        q_ref, a_ref, canon_ref, canon_window, n_real = _ref_sample(
            hdf5, start_ts, chunk, rot_repr, action_repr, stats
        )
        q_err = np.abs(item["observation.state"].numpy() - q_ref).max()
        a_err = np.abs(item["action"].numpy() - a_ref).max()
        n_pad_expected = chunk - n_real
        n_pad_got = int(item["action_is_pad"].sum())
        assert q_err < 1e-4, f"start={start_ts} state parity {q_err}"
        assert a_err < 1e-4, f"start={start_ts} action parity {a_err}"
        assert n_pad_got == n_pad_expected, f"start={start_ts} pad {n_pad_got}!={n_pad_expected}"

        # invertibility on the non-padded rows
        am, as_ = stats["action_mean"], stats["action_std"]
        idx = np.minimum(np.arange(chunk), am.shape[0] - 1)
        denorm = item["action"].numpy() * as_[idx] + am[idx]
        recon = ee_transforms.invert_action_chunk(denorm, canon_ref, action_repr, rot_repr)
        inv_err = np.abs(
            _unit_quat_canonical(recon[:n_real]) - _unit_quat_canonical(canon_window[:n_real])
        ).max()
        assert inv_err < 1e-4, f"start={start_ts} invert {inv_err}"
        print(
            f"  ts start={start_ts:3d}  state_err={q_err:.2e}  act_err={a_err:.2e}  invert_err={inv_err:.2e}  pad={n_pad_got}"
        )


def check_jointspace():
    root = Path("_fixtures/js")
    ds = EEReprDataset(root, rot_repr="rot6d", action_repr="relative", chunk_size=32)
    assert ds.state_dim == 14, ds.state_dim
    assert not ds.task_space
    stats = load_ee_repr_stats(root, "rot6d", "relative", 32)
    hdf5 = (
        Path(json.loads((root / "meta" / "act_bridge.json").read_text())["source_dir"])
        / "episode_0.hdf5"
    )
    with h5py.File(hdf5, "r") as f:
        qpos = f["/observations/qpos"][12].astype(np.float64)
        act = f["/action"][12:44].astype(np.float64)
    item = ds[12]
    q_ref = (qpos - stats["qpos_mean"]) / stats["qpos_std"]
    a_ref = (act - stats["action_mean"]) / stats["action_std"]
    q_err = np.abs(item["observation.state"].numpy() - q_ref).max()
    a_err = np.abs(item["action"].numpy() - a_ref).max()
    assert q_err < 1e-4 and a_err < 1e-4, (q_err, a_err)
    print(f"  js start= 12  state_err={q_err:.2e}  act_err={a_err:.2e}  (passthrough)")


if __name__ == "__main__":
    print("task-space rot6d/relative:")
    check_taskspace()
    print("joint-space passthrough:")
    check_jointspace()
    print("Loop 2 OK")
