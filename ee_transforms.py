"""Task-space (end-effector) action / rotation transforms for ACT.

Canonical recorded pose vector (per timestep, 16-dim, bimanual):

    [ l_xyz(3), l_quat_wxyz(4), l_grip(1),  r_xyz(3), r_quat_wxyz(4), r_grip(1) ]

`qpos` (achieved EE pose) and `action` (commanded EE target) are both stored in this
canonical form by ``record_sim_episodes.py --task_space``. The dataset loader and the
rollout code convert to/from a chosen *rotation representation* (``quat`` / ``rpy`` /
``rot6d``) and a chosen *action representation* (``absolute`` / ``delta`` / ``relative``,
UMI-style, see Fig. 6 of the reference) via the functions here.

Conventions:
- Quaternions are ``wxyz`` (MuJoCo / pyquaternion order); scipy uses ``xyzw`` internally.
- ``rot6d`` is the first two columns of the rotation matrix, flattened column-major
  ``[r00, r10, r20, r01, r11, r21]`` (Zhou et al., "On the Continuity of Rotation
  Representations in Neural Networks"); decoded back with Gram-Schmidt.
- ``rpy`` is extrinsic-xyz Euler angles in radians.
- Rotation (and gripper) channels are never normalized; only translation channels are.
"""

import numpy as np
from scipy.spatial.transform import Rotation

CANONICAL_DIM = 16  # bimanual canonical pose vector
ARM_CANONICAL_DIM = 8  # xyz(3) + quat_wxyz(4) + grip(1)
ROT6D_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

ROT_DIM = {"quat": 4, "rpy": 3, "rot6d": 6}
ACTION_REPRS = ("absolute", "delta", "relative")
ROT_REPRS = tuple(ROT_DIM.keys())


def arm_dim(rot_repr):
    """Per-arm feature dim for a rotation representation: xyz(3) + rot + grip(1)."""
    return 3 + ROT_DIM[rot_repr] + 1


def state_dim(rot_repr):
    """Bimanual state / action feature dim (quat->16, rpy->14, rot6d->20)."""
    return 2 * arm_dim(rot_repr)


# --------------------------------------------------------------------------------------
# rotation-matrix <-> representation
# --------------------------------------------------------------------------------------


def _canonicalize_quat_wxyz(q):
    """Force a consistent hemisphere (w >= 0) to avoid double-cover sign flips."""
    q = np.asarray(q, dtype=np.float64)
    return np.where(q[..., :1] < 0, -q, q)


def quat_wxyz_to_mat(q):
    """(..., 4) wxyz -> (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    flat = q.reshape(-1, 4)
    xyzw = np.concatenate([flat[:, 1:4], flat[:, 0:1]], axis=-1)
    mats = Rotation.from_quat(xyzw).as_matrix()
    return mats.reshape(q.shape[:-1] + (3, 3))


def mat_to_quat_wxyz(R):
    """(..., 3, 3) -> (..., 4) wxyz, canonicalized to w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    xyzw = Rotation.from_matrix(flat).as_quat()
    wxyz = np.concatenate([xyzw[:, 3:4], xyzw[:, 0:3]], axis=-1)
    wxyz = _canonicalize_quat_wxyz(wxyz)
    return wxyz.reshape(R.shape[:-2] + (4,))


def mat_to_rpy(R):
    """(..., 3, 3) -> (..., 3) extrinsic-xyz Euler radians."""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    e = Rotation.from_matrix(flat).as_euler("xyz")
    return e.reshape(R.shape[:-2] + (3,))


def rpy_to_mat(e):
    """(..., 3) extrinsic-xyz Euler radians -> (..., 3, 3)."""
    e = np.asarray(e, dtype=np.float64)
    flat = e.reshape(-1, 3)
    mats = Rotation.from_euler("xyz", flat).as_matrix()
    return mats.reshape(e.shape[:-1] + (3, 3))


def mat_to_rot6d(R):
    """(..., 3, 3) -> (..., 6): first two columns, column-major."""
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_mat(x):
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt."""
    x = np.asarray(x, dtype=np.float64)
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def mat_to_rot(R, rot_repr):
    if rot_repr == "quat":
        return mat_to_quat_wxyz(R)
    if rot_repr == "rpy":
        return mat_to_rpy(R)
    if rot_repr == "rot6d":
        return mat_to_rot6d(R)
    raise ValueError(f"unknown rot_repr {rot_repr!r}")


def rot_to_mat(x, rot_repr):
    if rot_repr == "quat":
        return quat_wxyz_to_mat(x)
    if rot_repr == "rpy":
        return rpy_to_mat(x)
    if rot_repr == "rot6d":
        return rot6d_to_mat(x)
    raise ValueError(f"unknown rot_repr {rot_repr!r}")


# --------------------------------------------------------------------------------------
# canonical-vector helpers
# --------------------------------------------------------------------------------------


def _arm_slices():
    return (slice(0, ARM_CANONICAL_DIM), slice(ARM_CANONICAL_DIM, CANONICAL_DIM))


def split_canonical_arm(vec):
    """(..., 8) canonical arm -> (xyz (...,3), quat_wxyz (...,4), grip (...,))."""
    vec = np.asarray(vec, dtype=np.float64)
    return vec[..., 0:3], vec[..., 3:7], vec[..., 7]


def canonicalize_quats(vec16):
    """Return a copy of a canonical (..., 16) vector with both quats hemisphere-fixed."""
    out = np.array(vec16, dtype=np.float64)
    for s in _arm_slices():
        q = out[..., s][..., 3:7]
        out[..., s.start + 3 : s.start + 7] = _canonicalize_quat_wxyz(q)
    return out


# --------------------------------------------------------------------------------------
# state transform
# --------------------------------------------------------------------------------------


def transform_state(ee_pose16, rot_repr):
    """Canonical (..., 16) achieved pose -> (..., state_dim(rot_repr)) feature vector.

    State always uses the *absolute* framing; only the rotation representation changes.
    """
    ee_pose16 = np.asarray(ee_pose16, dtype=np.float64)
    outs = []
    for s in _arm_slices():
        xyz, quat, grip = split_canonical_arm(ee_pose16[..., s])
        rot = mat_to_rot(quat_wxyz_to_mat(quat), rot_repr)
        outs.append(np.concatenate([xyz, rot, grip[..., None]], axis=-1))
    return np.concatenate(outs, axis=-1)


# --------------------------------------------------------------------------------------
# action transforms  (Fig. 6: absolute / delta / relative)
# --------------------------------------------------------------------------------------


def _transform_arm_chunk(xyz_t, R_t, grip_t, ref_xyz, ref_R, action_repr, rot_repr):
    if action_repr == "absolute":
        out_xyz, out_R = xyz_t, R_t
    elif action_repr == "relative":
        out_xyz = xyz_t - ref_xyz
        out_R = np.einsum("ij,tjk->tik", ref_R.T, R_t)
    elif action_repr == "delta":
        prev_xyz = np.concatenate([ref_xyz[None], xyz_t[:-1]], axis=0)
        prev_R = np.concatenate([ref_R[None], R_t[:-1]], axis=0)
        out_xyz = xyz_t - prev_xyz
        out_R = np.einsum("tij,tjk->tik", np.swapaxes(prev_R, -1, -2), R_t)
    else:
        raise ValueError(f"unknown action_repr {action_repr!r}")
    rot = mat_to_rot(out_R, rot_repr)
    return np.concatenate([out_xyz, rot, grip_t[:, None]], axis=-1)


def _invert_arm_chunk(feat, ref_xyz, ref_R, action_repr, rot_repr):
    d = ROT_DIM[rot_repr]
    out_xyz = feat[:, 0:3]
    out_R = rot_to_mat(feat[:, 3 : 3 + d], rot_repr)
    grip_t = feat[:, 3 + d]
    if action_repr == "absolute":
        xyz_t, R_t = out_xyz, out_R
    elif action_repr == "relative":
        xyz_t = out_xyz + ref_xyz
        R_t = np.einsum("ij,tjk->tik", ref_R, out_R)
    elif action_repr == "delta":
        T = feat.shape[0]
        xyz_t = np.empty((T, 3))
        R_t = np.empty((T, 3, 3))
        prev_xyz, prev_R = ref_xyz, ref_R
        for t in range(T):
            prev_xyz = prev_xyz + out_xyz[t]
            prev_R = prev_R @ out_R[t]
            xyz_t[t], R_t[t] = prev_xyz, prev_R
    else:
        raise ValueError(f"unknown action_repr {action_repr!r}")
    quat_t = mat_to_quat_wxyz(R_t)
    return np.concatenate([xyz_t, quat_t, grip_t[:, None]], axis=-1)


def transform_action_chunk(abs_chunk, ref, action_repr, rot_repr):
    """(T, 16) canonical absolute poses + (16,) reference pose -> (T, state_dim)."""
    abs_chunk = np.asarray(abs_chunk, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    outs = []
    for s in _arm_slices():
        xyz_t, quat_t, grip_t = split_canonical_arm(abs_chunk[:, s])
        ref_xyz, ref_quat, _ = split_canonical_arm(ref[s])
        outs.append(
            _transform_arm_chunk(
                xyz_t,
                quat_wxyz_to_mat(quat_t),
                grip_t,
                ref_xyz,
                quat_wxyz_to_mat(ref_quat),
                action_repr,
                rot_repr,
            )
        )
    return np.concatenate(outs, axis=-1)


def invert_action_chunk(pred, ref, action_repr, rot_repr):
    """(T, state_dim) predicted features + (16,) reference -> (T, 16) canonical absolute."""
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    ad = arm_dim(rot_repr)
    outs = []
    for i, s in enumerate(_arm_slices()):
        feat = pred[:, i * ad : (i + 1) * ad]
        ref_xyz, ref_quat, _ = split_canonical_arm(ref[s])
        outs.append(
            _invert_arm_chunk(feat, ref_xyz, quat_wxyz_to_mat(ref_quat), action_repr, rot_repr)
        )
    return np.concatenate(outs, axis=-1)


# --------------------------------------------------------------------------------------
# normalization mask
# --------------------------------------------------------------------------------------


def normalizable_mask(rot_repr):
    """bool (state_dim,): True only on translation (xyz) channels.

    Rotation channels are already bounded and the gripper is already in [0, 1]; both are
    passed through un-normalized (matching gr00t-h/open_h).
    """
    ad = arm_dim(rot_repr)
    m = np.zeros(2 * ad, dtype=bool)
    for i in range(2):
        m[i * ad : i * ad + 3] = True
    return m


# --------------------------------------------------------------------------------------
# self-test: round-trip transform / invert for all 9 combos
# --------------------------------------------------------------------------------------


def _random_canonical(n, rng):
    xyz = rng.uniform([-0.3, 0.3, 0.0], [0.3, 0.8, 0.5], size=(n, 3))
    xyzw = Rotation.random(n, random_state=rng).as_quat()
    quat = _canonicalize_quat_wxyz(np.concatenate([xyzw[:, 3:4], xyzw[:, 0:3]], axis=-1))
    grip = rng.uniform(0, 1, size=(n, 1))
    return np.concatenate([xyz, quat, grip], axis=-1)


def _self_test():
    rng = np.random.default_rng(0)
    T = 12
    left = _random_canonical(T, rng)
    right = _random_canonical(T, rng)
    abs_chunk = np.concatenate([left, right], axis=-1)
    ref = np.concatenate([_random_canonical(1, rng)[0], _random_canonical(1, rng)[0]])

    for action_repr in ACTION_REPRS:
        for rot_repr in ROT_REPRS:
            feat = transform_action_chunk(abs_chunk, ref, action_repr, rot_repr)
            assert feat.shape == (T, state_dim(rot_repr)), (action_repr, rot_repr, feat.shape)
            back = invert_action_chunk(feat, ref, action_repr, rot_repr)
            # compare xyz + grip directly; rotation via matrix (quat double-cover safe)
            for s in _arm_slices():
                a0, q0, g0 = split_canonical_arm(abs_chunk[:, s])
                a1, q1, g1 = split_canonical_arm(back[:, s])
                assert np.allclose(a0, a1, atol=1e-6), (action_repr, rot_repr, "xyz")
                assert np.allclose(g0, g1, atol=1e-6), (action_repr, rot_repr, "grip")
                R0 = quat_wxyz_to_mat(q0)
                R1 = quat_wxyz_to_mat(q1)
                assert np.allclose(R0, R1, atol=1e-5), (action_repr, rot_repr, "rot")
            # state transform round-trips too
            st = transform_state(abs_chunk[:1], rot_repr)
            assert st.shape == (1, state_dim(rot_repr))
            print(f"ok  {action_repr:9s} {rot_repr:6s}  feat_dim={feat.shape[1]}")
    print("all round-trip checks passed")


if __name__ == "__main__":
    _self_test()
