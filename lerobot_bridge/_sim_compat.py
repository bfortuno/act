"""``make_ee_sim_env`` for this project's mujoco (3.3), matched to the root repo's 2.3.7.

The EE envs drive each arm through a mocap weld (``assets/bimanual_viperx_ee_*.xml``,
default ``torquescale=1``). With the unmodified model, mujoco 3.3 tracks the commanded
*rotation* far more loosely than 2.3.7 (up to ~80 deg off during the scripted
transfer-cube rollout): the gripper arrives mis-oriented and the scripted policy never
picks the cube (0/8 vs 8/8 under 2.3.7).

Setting the weld ``torquescale`` to 20 restores the 2.3.7 behaviour: over full 400-step
scripted rollouts (8 seeds, including the grasp and handover) the achieved EE pose
matches 2.3.7 to < 0.1 mm / 0.01 deg, and neighbouring values (19, 21) are measurably
worse. Use this factory instead of ``ee_sim_env.make_ee_sim_env`` everywhere in the
bridge.
"""

from __future__ import annotations

import _shared  # noqa: F401
import ee_sim_env

# eq_data layout for a weld: [anchor(3), relpose pos(3), relpose quat(4), torquescale(1)]
_WELD_TORQUESCALE_IDX = 10
WELD_TORQUESCALE = 20.0


def make_ee_sim_env(task_name: str, **kwargs):
    env = ee_sim_env.make_ee_sim_env(task_name, **kwargs)
    env.physics.model.eq_data[:, _WELD_TORQUESCALE_IDX] = WELD_TORQUESCALE
    return env
