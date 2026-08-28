import pathlib

### Task parameters
DATA_DIR = "<put your data dir here>"
SIM_TASK_CONFIGS = {
    "sim_transfer_cube_scripted": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_scripted",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top", "left_wrist", "right_wrist"],
    },
    "sim_transfer_cube_human": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_human",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top", "left_wrist", "right_wrist"],
    },
    "sim_insertion_scripted": {
        "dataset_dir": DATA_DIR + "/sim_insertion_scripted",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top", "left_wrist", "right_wrist"],
    },
    "sim_insertion_human": {
        "dataset_dir": DATA_DIR + "/sim_insertion_human",
        "num_episodes": 50,
        "episode_len": 500,
        "camera_names": ["top", "left_wrist", "right_wrist"],
    },
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + "/assets/"  # note: absolute path

# End-effector body names (used for task-space / EE-pose recording)
EE_LINK_NAMES = ["vx300s_left/gripper_link", "vx300s_right/gripper_link"]

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################


def MASTER_GRIPPER_POSITION_NORMALIZE_FN(x):
    return (x - MASTER_GRIPPER_POSITION_CLOSE) / (
        MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
    )


def PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x):
    return (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
        PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
    )


def MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(x):
    return (
        x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
        + MASTER_GRIPPER_POSITION_CLOSE
    )


def PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(x):
    return (
        x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
        + PUPPET_GRIPPER_POSITION_CLOSE
    )


def MASTER2PUPPET_POSITION_FN(x):
    return PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))


def MASTER_GRIPPER_JOINT_NORMALIZE_FN(x):
    return (x - MASTER_GRIPPER_JOINT_CLOSE) / (
        MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE
    )


def PUPPET_GRIPPER_JOINT_NORMALIZE_FN(x):
    return (x - PUPPET_GRIPPER_JOINT_CLOSE) / (
        PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE
    )


def MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(x):
    return x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE


def PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(x):
    return x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE


def MASTER2PUPPET_JOINT_FN(x):
    return PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))


def MASTER_GRIPPER_VELOCITY_NORMALIZE_FN(x):
    return x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)


def PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(x):
    return x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)


def MASTER_POS2JOINT(x):
    return (
        MASTER_GRIPPER_POSITION_NORMALIZE_FN(x)
        * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
        + MASTER_GRIPPER_JOINT_CLOSE
    )


def MASTER_JOINT2POS(x):
    return MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(
        (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    )


def PUPPET_POS2JOINT(x):
    return (
        PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x)
        * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
        + PUPPET_GRIPPER_JOINT_CLOSE
    )


def PUPPET_JOINT2POS(x):
    return PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
        (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    )


MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2
