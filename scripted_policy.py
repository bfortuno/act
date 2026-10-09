import IPython
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

e = IPython.embed

# Per-episode discrete "funnel" via-point offsets (world xyz, meters). One mode is chosen
# per episode and applied to every transit segment, so episodes form separable behavioral
# clusters instead of washing out into continuous noise.
VIA_MODE_OFFSETS = {
    "direct": np.array([0.0, 0.0, 0.0]),
    "arc_high": np.array([0.0, 0.0, 0.06]),
    "arc_left": np.array([-0.05, 0.0, 0.02]),
    "arc_right": np.array([0.05, 0.0, 0.02]),
}


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        self.left_motion = None
        self.right_motion = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def _min_jerk(u):
        """Quintic minimum-jerk time-warp: u in [0,1] -> s in [0,1], s'(0)=s'(1)=0."""
        u = np.clip(u, 0.0, 1.0)
        return 6 * u**5 - 15 * u**4 + 10 * u**3

    @staticmethod
    def _catmull_rom_tangents(ts, pts):
        """Non-uniform Catmull-Rom finite-difference tangents; zero velocity at the two
        true endpoints so the arm doesn't overshoot past the start/hold pose."""
        n = len(ts)
        m = np.zeros_like(pts)
        for i in range(1, n - 1):
            m[i] = (pts[i + 1] - pts[i - 1]) / (ts[i + 1] - ts[i - 1])
        return m

    def _build_motion(self, waypoints):
        """Precompute a queryable smooth motion (position spline + slerp + gripper) over
        the full waypoint list (authored "bottleneck" waypoints plus any inserted "via"
        via-points)."""
        wp_ts = np.array([wp["t"] for wp in waypoints], dtype=float)
        xyz = np.stack([np.asarray(wp["xyz"], dtype=float) for wp in waypoints])
        quat_wxyz = np.stack([np.asarray(wp["quat"], dtype=float) for wp in waypoints])
        grip = np.array([wp["gripper"] for wp in waypoints], dtype=float)

        tangents = self._catmull_rom_tangents(wp_ts, xyz)
        pos_spline = CubicHermiteSpline(wp_ts, xyz, tangents, axis=0)

        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        slerp = Slerp(wp_ts, Rotation.from_quat(quat_xyzw))

        # outer segments used for the minimum-jerk time-warp: only the originally
        # authored waypoints ("bottleneck") are warp boundaries, so inserted via-points
        # are passed through at full speed instead of causing a stop-start at every point
        bottleneck_ts = np.array(
            [wp["t"] for wp in waypoints if wp.get("kind", "bottleneck") == "bottleneck"],
            dtype=float,
        )

        return {
            "t": wp_ts,
            "pos": pos_spline,
            "slerp": slerp,
            "grip": grip,
            "bottleneck_ts": bottleneck_ts,
        }

    def _query_motion(self, motion, t):
        bts = motion["bottleneck_ts"]
        t_clamped = float(np.clip(t, bts[0], bts[-1]))
        i = int(np.clip(np.searchsorted(bts, t_clamped, side="right") - 1, 0, len(bts) - 2))
        t0, t1 = bts[i], bts[i + 1]
        u = 0.0 if t1 == t0 else (t_clamped - t0) / (t1 - t0)
        s = self._min_jerk(u)
        warped_t = t0 + s * (t1 - t0)

        xyz = motion["pos"](warped_t)
        quat_xyzw = motion["slerp"]([warped_t]).as_quat()[0]
        quat = np.concatenate([quat_xyzw[3:4], quat_xyzw[0:3]])
        gripper = np.interp(warped_t, motion["t"], motion["grip"])
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)
            self.left_motion = self._build_motion(self.left_trajectory)
            self.right_motion = self._build_motion(self.right_trajectory)

        left_xyz, left_quat, left_gripper = self._query_motion(self.left_motion, self.step_count)
        right_xyz, right_quat, right_gripper = self._query_motion(
            self.right_motion, self.step_count
        )

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        init_mocap_pose_left = ts_first.observation["mocap_pose_left"]

        box_info = np.array(ts_first.observation["env_state"])
        box_xyz = box_info[:3]
        box_quat_wxyz = box_info[3:7]
        # print(f"Generate trajectory for {box_xyz=}")

        # yaw-aware grasp approach: box_info is MuJoCo/pyquaternion wxyz, scipy wants xyzw
        box_yaw = Rotation.from_quat(box_quat_wxyz[[1, 2, 3, 0]]).as_euler("xyz")[2]
        gripper_pick_quat = (
            Quaternion(axis=[0.0, 0.0, 1.0], radians=box_yaw)
            * Quaternion(init_mocap_pose_right[3:])
            * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
        )

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        # randomized handover rendezvous point (was a fixed constant); x centered on the
        # arm-base symmetry line, y/z bounded around the previously-fixed [0, 0.5, 0.25]
        meet_xyz = np.array(
            [
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(0.45, 0.55),
                np.random.uniform(0.22, 0.28),
            ]
        )

        self.left_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_left[:3],
                "quat": init_mocap_pose_left[3:],
                "gripper": 0,
                "kind": "bottleneck",
            },  # sleep
            {
                "t": 100,
                "xyz": meet_xyz + np.array([-0.1, 0, -0.02]),
                "quat": meet_left_quat.elements,
                "gripper": 1,
                "kind": "bottleneck",
            },  # approach meet position
            {
                "t": 260,
                "xyz": meet_xyz + np.array([0.02, 0, -0.02]),
                "quat": meet_left_quat.elements,
                "gripper": 1,
                "kind": "bottleneck",
            },  # move to meet position
            {
                "t": 310,
                "xyz": meet_xyz + np.array([0.02, 0, -0.02]),
                "quat": meet_left_quat.elements,
                "gripper": 0,
                "kind": "bottleneck",
            },  # close gripper
            {
                "t": 360,
                "xyz": meet_xyz + np.array([-0.1, 0, -0.02]),
                "quat": np.array([1, 0, 0, 0]),
                "gripper": 0,
                "kind": "bottleneck",
            },  # move left
            {
                "t": 400,
                "xyz": meet_xyz + np.array([-0.1, 0, -0.02]),
                "quat": np.array([1, 0, 0, 0]),
                "gripper": 0,
                "kind": "bottleneck",
            },  # stay
        ]

        self.right_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_right[:3],
                "quat": init_mocap_pose_right[3:],
                "gripper": 0,
                "kind": "bottleneck",
            },  # sleep
            {
                "t": 90,
                "xyz": box_xyz + np.array([0, 0, 0.08]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
                "kind": "bottleneck",
            },  # approach the cube
            {
                "t": 130,
                "xyz": box_xyz + np.array([0, 0, -0.015]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
                "kind": "bottleneck",
            },  # go down
            {
                "t": 170,
                "xyz": box_xyz + np.array([0, 0, -0.015]),
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
                "kind": "bottleneck",
            },  # close gripper
            {
                "t": 200,
                "xyz": meet_xyz + np.array([0.05, 0, 0]),
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
                "kind": "bottleneck",
            },  # approach meet position
            {
                "t": 220,
                "xyz": meet_xyz,
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
                "kind": "bottleneck",
            },  # move to meet position
            {
                "t": 310,
                "xyz": meet_xyz,
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
                "kind": "bottleneck",
            },  # open gripper
            {
                "t": 360,
                "xyz": meet_xyz + np.array([0.1, 0, 0]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
                "kind": "bottleneck",
            },  # move to right
            {
                "t": 400,
                "xyz": meet_xyz + np.array([0.1, 0, 0]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
                "kind": "bottleneck",
            },  # stay
        ]

        # discrete "funnel" via-point diversity: one mode per episode, applied to every
        # transit segment (grasp/meet/release bottlenecks above stay exact)
        # Insertions are applied per trajectory in descending idx_a order so that an
        # earlier insertion's index shift never invalidates a later (smaller-index) call.
        mode = np.random.choice(list(VIA_MODE_OFFSETS.keys()))
        offset = VIA_MODE_OFFSETS[mode]
        self._insert_via_point(self.right_trajectory, 6, 7, offset)  # release -> retreat
        self._insert_via_point(self.right_trajectory, 3, 4, offset)  # grasp -> approach meet
        self._insert_via_point(self.right_trajectory, 0, 1, offset)  # sleep -> approach cube
        self._insert_via_point(self.left_trajectory, 3, 4, offset)  # receive -> retreat
        self._insert_via_point(self.left_trajectory, 1, 2, offset)  # approach meet -> move to meet
        self._insert_via_point(self.left_trajectory, 0, 1, offset)  # sleep -> approach meet

        # per-episode timing jitter, synchronized at the shared handover instant (t=310)
        jitter_r = np.random.uniform(0.85, 1.15)
        self._apply_timing_jitter(self.left_trajectory, jitter_r)
        self._apply_timing_jitter(self.right_trajectory, jitter_r)

    @staticmethod
    def _insert_via_point(trajectory, idx_a, idx_b, offset):
        a, b = trajectory[idx_a], trajectory[idx_b]
        t_mid = (a["t"] + b["t"]) / 2.0
        xyz_mid = (np.asarray(a["xyz"]) + np.asarray(b["xyz"])) / 2.0 + offset
        grip_mid = (a["gripper"] + b["gripper"]) / 2.0
        via = {"t": t_mid, "xyz": xyz_mid, "quat": a["quat"], "gripper": grip_mid, "kind": "via"}
        trajectory.insert(idx_a + 1, via)

    @staticmethod
    def _apply_timing_jitter(trajectory, r, anchor=310.0, total=400.0):
        for wp in trajectory:
            t = wp["t"]
            if t <= anchor:
                wp["t"] = t * r
            else:
                frac = (t - anchor) / (total - anchor)
                wp["t"] = anchor * r + frac * (total - anchor * r)
        trajectory[-1]["t"] = total


class InsertionPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        init_mocap_pose_left = ts_first.observation["mocap_pose_left"]

        peg_info = np.array(ts_first.observation["env_state"])[:7]
        peg_xyz = peg_info[:3]
        peg_info[3:]

        socket_info = np.array(ts_first.observation["env_state"])[7:]
        socket_xyz = socket_info[:3]
        socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(
            axis=[0.0, 1.0, 0.0], degrees=-60
        )

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(
            axis=[0.0, 1.0, 0.0], degrees=60
        )

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_left[:3],
                "quat": init_mocap_pose_left[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 120,
                "xyz": socket_xyz + np.array([0, 0, 0.08]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 1,
            },  # approach the cube
            {
                "t": 170,
                "xyz": socket_xyz + np.array([0, 0, -0.03]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 1,
            },  # go down
            {
                "t": 220,
                "xyz": socket_xyz + np.array([0, 0, -0.03]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0,
            },  # close gripper
            {
                "t": 285,
                "xyz": meet_xyz + np.array([-0.1, 0, 0]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0,
            },  # approach meet position
            {
                "t": 340,
                "xyz": meet_xyz + np.array([-0.05, 0, 0]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0,
            },  # insertion
            {
                "t": 400,
                "xyz": meet_xyz + np.array([-0.05, 0, 0]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0,
            },  # insertion
        ]

        self.right_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_right[:3],
                "quat": init_mocap_pose_right[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 120,
                "xyz": peg_xyz + np.array([0, 0, 0.08]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 1,
            },  # approach the cube
            {
                "t": 170,
                "xyz": peg_xyz + np.array([0, 0, -0.03]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 1,
            },  # go down
            {
                "t": 220,
                "xyz": peg_xyz + np.array([0, 0, -0.03]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0,
            },  # close gripper
            {
                "t": 285,
                "xyz": meet_xyz + np.array([0.1, 0, lift_right]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0,
            },  # approach meet position
            {
                "t": 340,
                "xyz": meet_xyz + np.array([0.05, 0, lift_right]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0,
            },  # insertion
            {
                "t": 400,
                "xyz": meet_xyz + np.array([0.05, 0, lift_right]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0,
            },  # insertion
        ]


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]["episode_len"]
    if "sim_transfer_cube" in task_name:
        env = make_ee_sim_env("sim_transfer_cube")
    elif "sim_insertion" in task_name:
        env = make_ee_sim_env("sim_insertion")
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation["images"]["angle"])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for _step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation["images"]["angle"])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == "__main__":
    test_task_name = "sim_transfer_cube_scripted"
    test_policy(test_task_name)
