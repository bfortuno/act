"""Helpers copied verbatim (behaviour-preserving) from the repo-root
``imitate_episodes.py`` so ``eval_sim.py`` does not have to import that torch-2.0 module.

* ``aggregate_abs_poses``  == ``imitate_episodes._aggregate_abs_poses``
* ``exp_weights``          == the ``k`` schedule used in ``eval_bc_task_space``
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import _shared  # noqa: F401
import ee_transforms


def aggregate_abs_poses(cands: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Temporal-aggregate a set of canonical 16-dim absolute EE poses.

    xyz + gripper are weighted-averaged; each arm's quaternion is averaged with
    scipy ``Rotation.mean`` (re-orthonormalized) to stay on SO(3).
    """
    out = np.zeros(ee_transforms.CANONICAL_DIM)
    for arm in range(2):
        base = arm * ee_transforms.ARM_CANONICAL_DIM
        block = cands[:, base : base + ee_transforms.ARM_CANONICAL_DIM]
        out[base : base + 3] = (block[:, 0:3] * weights[:, None]).sum(0)
        out[base + 7] = float((block[:, 7] * weights).sum())
        xyzw = block[:, 3:7][:, [1, 2, 3, 0]]
        mean_xyzw = Rotation.from_quat(xyzw).mean(weights).as_quat()
        out[base + 3 : base + 7] = mean_xyzw[[3, 0, 1, 2]]
    return out


def exp_weights(n: int, k: float = 0.01) -> np.ndarray:
    w = np.exp(-k * np.arange(n))
    return w / w.sum()
