"""Failure Prediction Models (FPMs) for pre-emptive, FPM-gated DAgger.

An FPM scores a *proposed* action chunk before it is executed:

    f(o_t, L, A_{t:t+H}) -> p_hat_t in [0, 1],   p_hat_t ~ P(irreversible failure within H)

Everything here is consumed by ``collect_preemptive_demos.py``. See
README_preemptive_dagger.md for the workflow.

Contents
- ``FPMContext``                  what an FPM gets to see at a check
- ``FailurePredictor``            base class / plugin contract
- ``TransferCubeFailureMonitor``  stateful, privileged ground-truth failure detector
                                  (gives frame-accurate tau_fail for labels, and powers
                                  the lookahead oracle)
- ``LookaheadOracleFPM``          privileged upper bound: simulates the pending chunk
- ``ChunkConsistencyFPM``         training-free black-box ACC baseline
- ``PluginFPM``                   load a user FPM from ``pkg.module:factory``
- ``conformal_threshold``         split-conformal threshold from successful rollouts
"""

import argparse
import dataclasses
import importlib
import json
from typing import Any, Optional

import numpy as np

import ee_transforms

# --------------------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class FPMContext:
    """Everything an FPM may look at when scoring the learner at step ``t``.

    Action chunks follow the learner's own representation: ``rel_chunk`` is exactly what the
    policy emitted after de-normalization (UMI-style relative to ``ref_pose``, the achieved
    EE pose at query time, in the learner's ``rot_repr``); ``abs_chunk`` is the same chunk
    inverted to canonical absolute 16-dim poses [xyz, quat_wxyz, grip] x2.
    """

    t: int  # current env step
    observation: dict  # ts.observation (images per camera, ee_pose, ...)
    language: str  # task instruction (sim tasks: the task name)
    chunk_t: int  # env step at which the current chunk was queried
    ref_pose: np.ndarray  # (16,) achieved EE pose at chunk_t (relative-action reference)
    norm_chunk: np.ndarray  # (H, D) normalized policy output
    rel_chunk: np.ndarray  # (H, D) de-normalized, learner action_repr (relative)
    abs_chunk: np.ndarray  # (H, 16) canonical absolute poses
    pending_abs: np.ndarray  # (h<=H, 16) part of the plan still to be executed from t
    action_repr: str = "relative"
    rot_repr: str = "rot6d"
    physics: Any = None  # privileged sim handle; None for black-box FPMs
    env: Any = None  # privileged; the dm_control Environment
    monitor: Optional["TransferCubeFailureMonitor"] = None  # privileged online monitor


class FailurePredictor:
    """Plugin contract. Subclass, or return any object with ``__call__(ctx) -> float``
    (and optionally ``reset(env)``) from a ``PluginFPM`` factory."""

    name = "base"
    privileged = False

    def reset(self, env):  # called once per rollout, after env.reset()
        pass

    def __call__(self, ctx: FPMContext) -> float:
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# ground-truth failure monitor (privileged)
# --------------------------------------------------------------------------------------

_BOX = "red_box"
_TABLE = "table"


def contact_flags(physics):
    """(touch_left_arm, touch_right_arm, touch_table) for the red box.

    Unlike ``TransferCubeEETask.get_reward`` (which checks one finger geom per arm), either
    finger of an arm counts: a firm grasp often registers contact on only the other finger
    for a few steps, which would read as a spurious 'released'. Success is still judged by
    the env reward."""
    left = right = table = False
    for i in range(physics.data.ncon):
        c = physics.data.contact[i]
        n1 = physics.model.id2name(c.geom1, "geom") or ""
        n2 = physics.model.id2name(c.geom2, "geom") or ""
        if _BOX not in (n1, n2):
            continue
        other = n2 if n1 == _BOX else n1
        if other == _TABLE:
            table = True
        elif "gripper_finger" in other:
            if other.startswith("vx300s_left/"):
                left = True
            elif other.startswith("vx300s_right/"):
                right = True
    return left, right, table


def cube_pose(physics):
    return physics.data.qpos[16:23].copy()  # xyz + quat_wxyz (env_state)


def reward_from_flags(touch_l, touch_r, touch_table):
    r = 0
    if touch_r:
        r = 1
    if touch_r and not touch_table:
        r = 2
    if touch_l:
        r = 3
    if touch_l and not touch_table:
        r = 4
    return r


class TransferCubeFailureMonitor:
    """Stateful detector of *irreversible* failures in sim_transfer_cube.

    Events (first one wins, latched):
      drop          cube back on the table, touched by no finger, after it was lifted
      missed_grasp  right gripper opened earlier, is now commanded closed, its EE is >6cm
                    above the cube, cube still on the table with no right-finger contact,
                    and the cube was never lifted -> the arm is lifting an empty hand
      off_table     cube z dropped >3cm below its resting height (fell off the table)

    ``update`` is called once per env step with the action that was just applied. The
    monitor is cheap to ``clone()``, which the lookahead oracle uses to continue the
    current history inside a simulated future.
    """

    MISSED_GRASP_HEIGHT = 0.06
    OFF_TABLE_DROP = 0.03

    def __init__(self):
        self.reset()

    def reset(self, physics=None):
        # z_rest (cube resting height on the table) is latched on the first step the cube
        # touches only the table -- NOT at reset: the cube spawns at z=0.05 and falls.
        self.r_max = 0
        self.right_opened = False
        self.z_rest = None
        self.event = None
        self.t_event = None

    def clone(self):
        m = TransferCubeFailureMonitor.__new__(TransferCubeFailureMonitor)
        m.__dict__.update(self.__dict__)
        return m

    def update(self, physics, action, t):
        """Returns the event name if a failure is detected at this step, else None."""
        if self.event is not None:
            return self.event
        touch_l, touch_r, touch_table = contact_flags(physics)
        self.r_max = max(self.r_max, reward_from_flags(touch_l, touch_r, touch_table))
        box = cube_pose(physics)
        if self.z_rest is None and touch_table and not touch_l and not touch_r:
            self.z_rest = float(box[2])

        # achieved right gripper opening (normalized 0 closed .. 1 open) + EE poses
        ee = _ee_pose(physics)
        if ee[15] > 0.5:
            self.right_opened = True

        event = None
        if self.z_rest is not None and box[2] < self.z_rest - self.OFF_TABLE_DROP:
            event = "off_table"
        elif self.r_max >= 2 and touch_table and not touch_l and not touch_r:
            event = "drop"
        elif (
            self.r_max < 2
            and self.right_opened
            and action is not None
            and float(action[15]) < 0.3
            # commanded (mocap) height, the frame actions live in: a correct grasp is at
            # cube_z - 0.015, hover at +0.08 (gripper_link ee_pose sits ~0.13m higher)
            and float(physics.data.mocap_pos[1][2]) > box[2] + self.MISSED_GRASP_HEIGHT
            and touch_table
            and not touch_r
        ):
            event = "missed_grasp"
        if event is not None:
            self.event, self.t_event = event, t
        return event


def _ee_pose(physics):
    from ee_sim_env import BimanualViperXEETask

    return BimanualViperXEETask.get_ee_pose(physics)


# --------------------------------------------------------------------------------------
# physics snapshot helpers
# --------------------------------------------------------------------------------------


def snapshot_physics(physics):
    d = physics.data
    return {
        "state": physics.get_state().copy(),
        "time": float(d.time),
        "mocap_pos": d.mocap_pos.copy(),
        "mocap_quat": d.mocap_quat.copy(),
        "ctrl": d.ctrl.copy(),
        "qacc_warmstart": d.qacc_warmstart.copy(),
        "act": d.act.copy(),
    }


def restore_physics(physics, snap):
    d = physics.data
    with physics.reset_context():
        physics.set_state(snap["state"])
        d.time = snap["time"]
        np.copyto(d.mocap_pos, snap["mocap_pos"])
        np.copyto(d.mocap_quat, snap["mocap_quat"])
        np.copyto(d.ctrl, snap["ctrl"])
        np.copyto(d.act, snap["act"])
    # reset_context runs mj_forward, which overwrites the warmstart; put it back last so
    # the next real mj_step is bit-identical to one without a lookahead in between
    np.copyto(d.qacc_warmstart, snap["qacc_warmstart"])


# --------------------------------------------------------------------------------------
# built-in FPMs
# --------------------------------------------------------------------------------------


class LookaheadOracleFPM(FailurePredictor):
    """Privileged sim upper bound: simulate the pending chunk forward (no rendering),
    run the failure monitor (continuing the real history), restore the physics.

    Score = gamma ** k, k = steps until the first predicted failure (0 if none within
    ``horizon``). The threshold therefore acts as a time-to-failure gate:
    p > thr  <=>  failure predicted within log(thr)/log(gamma) steps.
    """

    name = "lookahead_oracle"
    privileged = True

    def __init__(self, horizon=None, gamma=0.98, hold_last=True):
        self.horizon = horizon
        self.gamma = gamma
        self.hold_last = hold_last  # keep commanding the last pose if chunk < horizon

    def __call__(self, ctx):
        env, physics = ctx.env, ctx.physics
        if env is None or physics is None:
            raise ValueError("LookaheadOracleFPM needs privileged ctx.env / ctx.physics")
        actions = np.asarray(ctx.pending_abs)
        H = self.horizon or len(actions)
        if len(actions) < H and self.hold_last:
            actions = np.concatenate([actions, np.repeat(actions[-1:], H - len(actions), 0)])
        actions = actions[:H]

        monitor = (ctx.monitor or TransferCubeFailureMonitor()).clone()
        snap = snapshot_physics(physics)
        task, n_sub = env.task, env._n_sub_steps
        k_fail = None
        try:
            for k, a in enumerate(actions):
                task.before_step(a, physics)
                physics.step(n_sub)
                if monitor.update(physics, a, ctx.t + k) is not None:
                    k_fail = k
                    break
        finally:
            restore_physics(physics, snap)
        return 0.0 if k_fail is None else float(self.gamma**k_fail)


def _chunk_distance(a, b, w_rot=0.05, w_grip=0.05):
    """Mean per-step distance between two (h, 16) canonical chunks: xyz L2 (m) +
    w_rot * geodesic angle (rad) + w_grip * |gripper diff|, summed over both arms."""
    d = np.zeros(len(a))
    for s in (slice(0, 8), slice(8, 16)):
        A, B = a[:, s], b[:, s]
        d += np.linalg.norm(A[:, :3] - B[:, :3], axis=-1)
        dot = np.abs(np.sum(A[:, 3:7] * B[:, 3:7], axis=-1))
        dot /= np.linalg.norm(A[:, 3:7], axis=-1) * np.linalg.norm(B[:, 3:7], axis=-1) + 1e-9
        d += w_rot * 2 * np.arccos(np.clip(dot, -1.0, 1.0))
        d += w_grip * np.abs(A[:, 7] - B[:, 7])
    return float(d.mean())


class ChunkConsistencyFPM(FailurePredictor):
    """Training-free, black-box Action Chunk Consistency (ACC) score.

    Compares the overlap of the newest chunk with the previous one (in absolute space, so
    relative-action reference changes don't count as disagreement). Oscillating
    receding-horizon plans are a known precursor of failure. Needs overlapping chunks:
    use --temporal_agg or --query_every < chunk_size, otherwise it always returns 0.
    """

    name = "chunk_consistency"

    def __init__(self, scale=0.01, min_overlap=4):
        self.scale = scale
        self.min_overlap = min_overlap
        self.prev = None  # (chunk_t, abs_chunk)

    def reset(self, env):
        self.prev = None

    def __call__(self, ctx):
        score = 0.0
        if self.prev is not None and self.prev[0] < ctx.chunk_t:
            pt, pchunk = self.prev
            off = ctx.chunk_t - pt
            n = min(len(pchunk) - off, len(ctx.abs_chunk))
            if n >= self.min_overlap:
                d = _chunk_distance(pchunk[off : off + n], ctx.abs_chunk[:n])
                score = 1.0 - float(np.exp(-d / self.scale))
        if self.prev is None or self.prev[0] != ctx.chunk_t:
            self.prev = (ctx.chunk_t, np.array(ctx.abs_chunk))
        return score


class PluginFPM(FailurePredictor):
    """Wrap a user FPM given as ``pkg.module:factory``; ``factory(**kwargs)`` must return
    a callable ``(FPMContext) -> float`` (optionally with ``reset(env)`` and a
    ``privileged`` attribute). This is where a trained dual-stream guardian plugs in."""

    def __init__(self, spec, kwargs=None):
        mod_name, _, attr = spec.partition(":")
        if not attr:
            raise ValueError(f"--fpm plugin must look like 'pkg.module:factory', got {spec!r}")
        factory = getattr(importlib.import_module(mod_name), attr)
        self.impl = factory(**(kwargs or {}))
        self.name = f"plugin:{spec}"
        self.privileged = bool(getattr(self.impl, "privileged", False))

    def reset(self, env):
        if hasattr(self.impl, "reset"):
            self.impl.reset(env)

    def __call__(self, ctx):
        return float(np.clip(self.impl(ctx), 0.0, 1.0))


def make_fpm(name, kwargs=None):
    kwargs = kwargs or {}
    if name in ("oracle", "lookahead_oracle"):
        return LookaheadOracleFPM(**kwargs)
    if name in ("acc", "chunk_consistency"):
        return ChunkConsistencyFPM(**kwargs)
    if ":" in name:
        return PluginFPM(name, kwargs)
    raise ValueError(f"unknown FPM {name!r} (oracle | acc | pkg.module:factory)")


# --------------------------------------------------------------------------------------
# labels + conformal calibration
# --------------------------------------------------------------------------------------


def window_labels(T, tau_fail, H):
    """Per-step FPM targets: 1 in [tau-H, tau], -1 (ignore) in (tau-2H, tau-H),
    0 elsewhere; all 0 for a successful rollout (tau_fail None)."""
    y = np.zeros(T, dtype=np.int8)
    if tau_fail is None:
        return y
    lo1 = max(0, tau_fail - H)
    y[max(0, tau_fail - 2 * H + 1) : lo1] = -1
    y[lo1 : min(T, tau_fail + 1)] = 1
    return y


def conformal_threshold(max_scores_successful, alpha):
    """Split conformal: threshold q such that a new *successful* rollout's max score
    exceeds q (false alarm) with probability <= alpha (exchangeability assumed).
    Returns inf if there are too few calibration rollouts for this alpha."""
    s = np.sort(np.asarray(max_scores_successful, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or k > n:
        return float("inf")
    return float(s[k - 1])


def _calibrate_cli(args):
    data = np.load(args.scores, allow_pickle=True)
    scores = [np.asarray(s, dtype=float) for s in data["scores"]]  # (T,) each, NaN = not evaluated
    success = np.asarray(data["success"], bool)
    tau = list(data["tau_fail"])  # -1 if none
    maxes = np.array([np.nanmax(s) if np.isfinite(s).any() else 0.0 for s in scores])
    q = conformal_threshold(maxes[success], args.alpha)
    print(f"calibration rollouts: {success.sum()} successful / {len(success)} total")
    print(f"alpha={args.alpha}  ->  threshold q={q:.6g}  (trigger if p_hat > q)")

    fail_idx = np.flatnonzero(~success)
    if len(fail_idx) and np.isfinite(q):
        caught, leads = 0, []
        for i in fail_idx:
            s = np.nan_to_num(scores[i], nan=0.0)
            t_end = int(tau[i]) if int(tau[i]) >= 0 else len(s)
            cross = np.flatnonzero(s[: t_end + 1] > q)
            if len(cross):
                caught += 1
                leads.append(t_end - int(cross[0]))
        print(
            f"failed rollouts flagged before/at tau_fail: {caught}/{len(fail_idx)}"
            + (f", median lead {np.median(leads):.0f} steps" if leads else "")
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"alpha": args.alpha, "threshold": q, "n_success": int(success.sum())}, f)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="conformal threshold from an --observe_only run")
    c.add_argument("--scores", required=True, help="<dataset_dir>/fpm_scores.npz")
    c.add_argument("--alpha", type=float, default=0.1, help="per-episode false-alarm budget")
    c.add_argument("--out", default=None, help="optional json output")
    a = p.parse_args()
    if a.cmd == "calibrate":
        _calibrate_cli(a)

# keep ee_transforms imported for plugin authors who want to convert chunks
__all__ = [
    "FPMContext",
    "FailurePredictor",
    "TransferCubeFailureMonitor",
    "LookaheadOracleFPM",
    "ChunkConsistencyFPM",
    "PluginFPM",
    "make_fpm",
    "window_labels",
    "conformal_threshold",
    "contact_flags",
    "cube_pose",
    "ee_transforms",
]
