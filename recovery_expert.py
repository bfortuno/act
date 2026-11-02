"""Deterministic replanning oracle expert for sim_transfer_cube (pre-emptive DAgger).

``PickAndTransferPolicy`` (scripted_policy.py) is open-loop from t=0, so it cannot take
over from an arbitrary mid-episode state. ``TransferCubeRecoveryExpert.plan`` replans from
the *current* state at takeover step t0, using the same geometry and style as the
scripted demos (fixed meet point [0, 0.5, 0.25], the same grasp/meet quaternions,
piecewise-linear waypoint interpolation, the same segment durations), so the recovery data
stays unimodal/deterministic and differs from the base demos only where recovery
requires it.

Phases (from privileged contacts + cube pose):
  pick      cube on the table and not grasped (includes after a drop): retract up if low,
            hover above the cube, descend, close, carry to meet, hand over, retreat
  carry     right gripper holds the cube: go to meet, hand over, retreat
  handover  left finger touches the airborne cube: left closes + retreats, then right opens
            + retreats
  off_table cube fell off the table -> unrecoverable (plan returns None)
"""

import numpy as np
from pyquaternion import Quaternion

from fpm import contact_flags, cube_pose

MEET_XYZ = np.array([0.0, 0.5, 0.25])  # scripted_policy.PickAndTransferPolicy
INIT_RIGHT_QUAT = Quaternion([1.0, 0.0, 0.0, 0.0])  # ee_sim_env initialize_robots
PICK_QUAT = (INIT_RIGHT_QUAT * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)).elements
MEET_LEFT_QUAT = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90).elements
LEFT_RETREAT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# segment durations copied from the deterministic scripted schedule
DUR_DOWN = 40  # 90 -> 130
DUR_CLOSE = 40  # 130 -> 170
DUR_CARRY = 30  # 170 -> 200
DUR_MEET = 20  # 200 -> 220
DUR_HANDOVER = 90  # right open 220 -> 310 (left closes 260 -> 310)
LEFT_FINE_AFTER_RIGHT_MEET = 40  # left at fine meet 260 = right meet 220 + 40
DUR_RETREAT = 50  # 310 -> 360
# reach speeds (m/step) matching the scripted schedule: sleep->hover ~0.35m in 90 steps,
# grasp->approach-meet ~0.25m in 30 steps
REACH_SPEED = 0.004
CARRY_SPEED = 0.008
MIN_REACH = 20

# refuse plans that would need more speed-up than this to fit the remaining episode:
# pick includes a fast 25cm carry (keep <= 2x); carry/handover are short motions (2.5x)
MIN_TIME_SCALE = {"pick": 0.5, "carry": 0.4, "handover": 0.4}
END_MARGIN = 10  # steps of final hold guaranteed before episode end


def _wp(t, xyz, quat, grip):
    q = np.asarray(quat, float)
    return {
        "t": float(t),
        "xyz": np.asarray(xyz, float),
        "quat": q / np.linalg.norm(q),
        "gripper": float(grip),
    }


def _reach(a, b, speed=REACH_SPEED, min_steps=MIN_REACH):
    return max(min_steps, int(np.ceil(np.linalg.norm(np.asarray(b) - np.asarray(a)) / speed)))


def detect_phase(physics, last_action, z_rest, monitor=None, t=None):
    """Who holds the cube. Contacts flicker for single steps even in a firm grasp (the
    cube can be airborne with zero registered contacts), so with a ``monitor`` the
    decision uses contact *history* (touched within the last 15 steps) and the cube's
    height (airborne = somebody holds it) instead of this step's contacts alone."""
    touch_l, touch_r, touch_table = contact_flags(physics)
    box = cube_pose(physics)
    if z_rest is not None and box[2] < z_rest - 0.03:
        return "off_table"
    if monitor is not None and t is not None:
        touch_l = touch_l or monitor.touched_recently("left", t)
        touch_r = touch_r or monitor.touched_recently("right", t)
    airborne = z_rest is not None and box[2] > z_rest + 0.02
    if touch_l and (airborne or not touch_table):
        return "handover"
    right_cmd_closed = last_action is not None and float(last_action[15]) < 0.5
    if touch_r and (airborne or not touch_table or right_cmd_closed):
        return "carry"
    if airborne:  # held, but by nobody we saw recently: the right arm carries by default
        return "carry"
    return "pick"


class TransferCubeRecoveryExpert:
    def __init__(self, left_wps, right_wps, t0, phase, time_scale):
        self.left = left_wps
        self.right = right_wps
        self.t0 = t0
        self.phase = phase
        self.time_scale = time_scale
        self.k = 0

    # ------------------------------------------------------------------ planning
    @classmethod
    def plan(cls, physics, obs, last_action, t0, episode_len, z_rest=None, monitor=None):
        """Build a recovery plan from the current state, or None if unrecoverable.
        Pass the episode's failure ``monitor`` for flicker-robust phase detection."""
        if monitor is not None and z_rest is None:
            z_rest = monitor.z_rest
        phase = detect_phase(physics, last_action, z_rest, monitor=monitor, t=t0 - 1)
        if phase == "off_table":
            return None

        box = cube_pose(physics)[:3]
        ml, mr = obs["mocap_pose_left"], obs["mocap_pose_right"]
        g_l = float(last_action[7]) if last_action is not None else 0.0
        g_r = float(last_action[15]) if last_action is not None else 0.0
        S_l = _wp(0, ml[:3], ml[3:], g_l)
        S_r = _wp(0, mr[:3], mr[3:], g_r)

        if phase == "handover":
            left, right = cls._plan_handover(S_l, S_r)
        else:
            right, t_rmeet = (cls._plan_pick if phase == "pick" else cls._plan_carry)(S_r, box)
            left = cls._plan_left_receive(S_l, t_rmeet)
            right = right + cls._handover_tail_right(t_rmeet)
            _fill_none_quats(right)  # tail keeps the meet orientation

        # fit into the remaining episode budget (uniform time-scaling, deterministic)
        horizon = episode_len - t0
        end = max(left[-1]["t"], right[-1]["t"])
        budget = horizon - END_MARGIN
        scale = 1.0
        if end > budget:
            scale = budget / end
            if scale < MIN_TIME_SCALE[phase]:
                return None
            for wp in left + right:
                wp["t"] *= scale
        for traj in (left, right):
            last = traj[-1]
            traj.append(_wp(horizon, last["xyz"], last["quat"], last["gripper"]))  # hold
        return cls(left, right, t0, phase, scale)

    @staticmethod
    def _plan_pick(S, box):
        wps = [S]
        t = 0.0
        cur = S["xyz"]
        hover = box + np.array([0, 0, 0.08])
        if cur[2] < box[2] + 0.06:  # low near the table/cube: lift straight up first
            up = np.array([cur[0], cur[1], hover[2]])
            t += _reach(cur, up)
            wps.append(_wp(t, up, S["quat"], 1))
            cur = up
        t += _reach(cur, hover)
        wps.append(_wp(t, hover, PICK_QUAT, 1))
        grasp = box + np.array([0, 0, -0.015])
        t += DUR_DOWN
        wps.append(_wp(t, grasp, PICK_QUAT, 1))
        t += DUR_CLOSE
        wps.append(_wp(t, grasp, PICK_QUAT, 0))
        t += DUR_CARRY
        wps.append(_wp(t, MEET_XYZ + np.array([0.05, 0, 0]), PICK_QUAT, 0))
        t += DUR_MEET
        wps.append(_wp(t, MEET_XYZ, PICK_QUAT, 0))
        return wps, t

    @staticmethod
    def _plan_carry(S, box):
        # keep the current grasp orientation and gripper command: re-orienting or
        # re-closing a held cube (e.g. the scripted handover had already started opening)
        # risks a drop
        wps = [S]
        t = 0.0
        g = S["gripper"]
        approach = MEET_XYZ + np.array([0.05, 0, 0])
        if np.linalg.norm(S["xyz"] - MEET_XYZ) > 0.06:
            t += _reach(S["xyz"], approach, speed=CARRY_SPEED)
            wps.append(_wp(t, approach, S["quat"], g))
        t += DUR_MEET
        wps.append(_wp(t, MEET_XYZ, S["quat"], g))
        return wps, t

    @staticmethod
    def _handover_tail_right(t_rmeet):
        # right: open at meet over the handover window, then retreat; quat None = inherit
        # the meet waypoint's orientation (resolved by _fill_none_quats)
        return [
            {"t": t_rmeet + DUR_HANDOVER, "xyz": MEET_XYZ.copy(), "quat": None, "gripper": 1.0},
            {
                "t": t_rmeet + DUR_HANDOVER + DUR_RETREAT,
                "xyz": MEET_XYZ + np.array([0.1, 0, 0]),
                "quat": None,
                "gripper": 1.0,
            },
        ]

    @staticmethod
    def _plan_left_receive(S, t_rmeet):
        t_fine = t_rmeet + LEFT_FINE_AFTER_RIGHT_MEET
        t_close = t_rmeet + DUR_HANDOVER
        coarse = MEET_XYZ + np.array([-0.1, 0, -0.02])
        fine = MEET_XYZ + np.array([0.02, 0, -0.02])
        wps = [S]
        near = min(np.linalg.norm(S["xyz"] - coarse), np.linalg.norm(S["xyz"] - fine)) < 0.03
        if not near:
            t_c = min(_reach(S["xyz"], coarse), t_fine - MIN_REACH)
            if t_c > 5:
                wps.append(_wp(t_c, coarse, MEET_LEFT_QUAT, 1))
        wps.append(_wp(max(t_fine, wps[-1]["t"] + 5), fine, MEET_LEFT_QUAT, 1))
        wps.append(_wp(max(t_close, wps[-1]["t"] + 5), fine, MEET_LEFT_QUAT, 0))
        wps.append(_wp(wps[-1]["t"] + DUR_RETREAT, coarse, LEFT_RETREAT_QUAT, 0))
        return wps

    @staticmethod
    def _plan_handover(S_l, S_r):
        t_close = 30
        left = [
            S_l,
            _wp(t_close, S_l["xyz"], S_l["quat"], 0),
            _wp(t_close + DUR_RETREAT, S_l["xyz"] + np.array([-0.12, 0, 0]), S_l["quat"], 0),
        ]
        right = [
            S_r,
            _wp(t_close, S_r["xyz"], S_r["quat"], S_r["gripper"]),  # hold until left closed
            _wp(t_close + 30, S_r["xyz"], S_r["quat"], 1),
            _wp(t_close + 30 + DUR_RETREAT, S_r["xyz"] + np.array([0.1, 0, 0]), S_r["quat"], 1),
        ]
        return left, right

    # ------------------------------------------------------------------ execution
    @staticmethod
    def _query(wps, t):
        ts = np.array([w["t"] for w in wps])
        t = float(np.clip(t, ts[0], ts[-1]))
        i = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(wps) - 2))
        a, b = wps[i], wps[i + 1]
        f = 0.0 if b["t"] == a["t"] else (t - a["t"]) / (b["t"] - a["t"])
        xyz = a["xyz"] + (b["xyz"] - a["xyz"]) * f
        quat = a["quat"] + (b["quat"] - a["quat"]) * f  # linear, as in BasePolicy.interpolate
        quat = quat / np.linalg.norm(quat)
        grip = a["gripper"] + (b["gripper"] - a["gripper"]) * f
        return np.concatenate([xyz, quat, [grip]])

    def __call__(self, ts=None):
        a = np.concatenate([self._query(self.left, self.k), self._query(self.right, self.k)])
        self.k += 1
        return a

    def describe(self):
        return {"phase": self.phase, "t0": int(self.t0), "time_scale": float(self.time_scale)}


def _fill_none_quats(traj):
    for i, wp in enumerate(traj):
        if wp["quat"] is None:
            wp["quat"] = traj[i - 1]["quat"]
