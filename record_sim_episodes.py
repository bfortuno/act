import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy
import ee_transforms

import IPython
e = IPython.embed


def save_task_space_episode(dataset_dir, episode_idx, episode, policy_actions, camera_names, episode_len):
    """Save one episode in canonical task-space form.

    /observations/qpos  (episode_len, 16)  achieved EE pose  [xyz, quat_wxyz, grip] x2
    /action             (episode_len, 16)  commanded EE target (scripted policy output)
    /observations/images/<cam>  (episode_len, 480, 640, 3) uint8
    No qvel. Rotation representation stored is raw quaternion; the dataset loader converts
    to quat / rpy / rot6d and to absolute / delta / relative at train time.
    """
    max_timesteps = episode_len
    qpos = np.stack([ee_transforms.canonicalize_quats(episode[t].observation['ee_pose'])
                     for t in range(max_timesteps)]).astype(np.float32)
    action = np.stack([ee_transforms.canonicalize_quats(np.asarray(policy_actions[t]))
                       for t in range(max_timesteps)]).astype(np.float32)
    data_dict = {'/observations/qpos': qpos, '/action': action}
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = np.stack(
            [episode[t].observation['images'][cam_name] for t in range(max_timesteps)])

    t0 = time.time()
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        root.attrs['task_space'] = True
        root.attrs['rot_repr'] = 'quat'
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                 chunks=(1, 480, 640, 3))
        obs.create_dataset('qpos', (max_timesteps, ee_transforms.CANONICAL_DIM))
        root.create_dataset('action', (max_timesteps, ee_transforms.CANONICAL_DIM))
        for name, array in data_dict.items():
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs\n')


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.

    With --task_space, skip the joint-space replay and instead record, straight from the
    EE rollout, the canonical 16-dim task-space vectors: /observations/qpos = achieved EE
    pose [xyz, quat_wxyz, grip] x2, /action = the scripted policy's commanded EE target.
    qvel is not recorded. See ee_transforms.py.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    task_space = args['task_space']
    inject_noise = False
    render_cam_name = 'angle'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    else:
        raise NotImplementedError

    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)
        policy_actions = []
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action = policy(ts)
            policy_actions.append(np.array(action))
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        if task_space:
            # record achieved EE pose + commanded EE target straight from the EE rollout;
            # no joint-space replay
            success.append(int(episode_max_reward == env.task.max_reward))
            save_task_space_episode(dataset_dir, episode_idx, episode, policy_actions,
                                    camera_names, episode_len)
            del env
            del episode
            del policy
            continue

        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0

        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--task_space', action='store_true',
                        help='record canonical task-space EE pose (16-dim) instead of joint-space (14-dim)')
    
    main(vars(parser.parse_args()))

