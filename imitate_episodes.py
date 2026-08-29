import argparse
import contextlib
import os
import pickle
from copy import deepcopy

import IPython
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import ee_transforms
from constants import CAMERA_HEIGHT, CAMERA_WIDTH, DT, PUPPET_GRIPPER_JOINT_OPEN
from policy import ACTPolicy, CNNMLPPolicy, DiffusionFlowPolicy
from sim_env import BOX_POSE
from utils import (  # robot functions  # helper functions
    compute_dict_mean,
    detach_dict,
    load_data,  # data functions
    sample_box_pose,
    sample_insertion_pose,
    set_seed,
)
from visualize_episodes import save_videos

e = IPython.embed


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]
    task_space = args["task_space"]
    action_repr = args["action_repr"]
    rot_repr = args["rot_repr"]
    if task_space and args.get("chunk_size") is None:
        raise ValueError("--chunk_size is required with --task_space (per-chunk-step action stats)")

    # get task parameters
    is_sim = task_name[:4] == "sim_"
    if is_sim:
        from constants import SIM_TASK_CONFIGS

        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS

        task_config = TASK_CONFIGS[task_name]
    dataset_dir = args.get("dataset_dir") or task_config["dataset_dir"]
    num_episodes = args.get("num_episodes") or task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = ee_transforms.state_dim(rot_repr) if task_space else 14
    action_dim = state_dim
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }
    elif policy_class == "DiffusionFlow":
        if args.get("chunk_size") is None:
            raise ValueError("--chunk_size is required with --policy_class DiffusionFlow")
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "hidden_dim": args["hidden_dim"] or 512,
            "dim_feedforward": args["dim_feedforward"] or 2048,
            "nheads": args["nheads"] or 8,
            "dit_layers": args["dit_layers"],
            "dit_arch": args["dit_arch"],
            "n_inference_steps": args["n_inference_steps"],
            "state_dropout_prob": args["state_dropout_prob"],
            "cam_dropout_prob": args["cam_dropout_prob"],
            "dropout": 0.1,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "camera_names": camera_names,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "task_space": task_space,
        "action_repr": action_repr,
        "rot_repr": rot_repr,
        "chunk_size": args["chunk_size"],
        "num_rollouts": args.get("num_rollouts"),
        "num_checkpoints": args.get("num_checkpoints", 5),
    }

    if is_eval:
        ckpt_names = ["policy_best.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            eval_fn = eval_bc_task_space if task_space else eval_bc
            success_rate, avg_return = eval_fn(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        task_space=task_space,
        action_repr=action_repr,
        rot_repr=rot_repr,
        chunk_size=args["chunk_size"],
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == "DiffusionFlow":
        policy = DiffusionFlowPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT" or policy_class == "CNNMLP" or policy_class == "DiffusionFlow":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    def pre_process(s_qpos):
        return (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]

    def post_process(a):
        return a * stats["action_std"] + stats["action_mean"]

    # load environment
    if real_robot:
        from aloha_scripts.real_env import make_real_env  # requires aloha
        from aloha_scripts.robot_utils import move_grippers  # requires aloha

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env

        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(
                env._physics.render(
                    height=CAMERA_HEIGHT, width=CAMERA_WIDTH, camera_id=onscreen_cam
                )
            )
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(
                        height=CAMERA_HEIGHT, width=CAMERA_WIDTH, camera_id=onscreen_cam
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config["policy_class"] in ("ACT", "DiffusionFlow"):
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right],
                [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                move_time=0.5,
            )  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])  # noqa: E711  numpy boolean-mask, drops None entries
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward == env_max_reward}"
        )

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f"video{rollout_id}.mp4"))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += (
            f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate * 100}%\n"
        )

    print(summary_str)

    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def _aggregate_abs_poses(cands, weights):
    """Temporal-aggregate a set of canonical 16-dim absolute EE poses.

    xyz + gripper are weighted-averaged; each arm's quaternion is averaged with
    scipy Rotation.mean (re-orthonormalized) to stay on SO(3).
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


def eval_bc_task_space(config, ckpt_name, save_episode=True):
    """Rollout eval for task-space (EE-pose) policies in ee_sim_env.

    The policy predicts a chunk in the trained (action_repr, rot_repr); each query it is
    denormalized (per-chunk-step stats for delta/relative), inverted back to canonical
    absolute EE poses using the current achieved EE pose as reference, and stepped
    straight into ee_sim_env (16-dim [xyz, quat_wxyz, grip] x2).
    """
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    action_repr = stats.get("action_repr", config["action_repr"])
    rot_repr = stats.get("rot_repr", config["rot_repr"])
    qpos_mean, qpos_std = stats["qpos_mean"], stats["qpos_std"]
    action_mean, action_std = stats["action_mean"], stats["action_std"]

    def pre_process(ee_pose):
        return (ee_transforms.transform_state(ee_pose, rot_repr) - qpos_mean) / qpos_std

    def denorm_chunk(norm_chunk):
        if action_mean.ndim == 2:
            idx = np.minimum(np.arange(norm_chunk.shape[0]), action_mean.shape[0] - 1)
            return norm_chunk * action_std[idx] + action_mean[idx]
        return norm_chunk * action_std + action_mean

    from ee_sim_env import make_ee_sim_env

    env = make_ee_sim_env(task_name)
    env_max_reward = env.task.max_reward

    num_queries = policy_config["num_queries"]
    query_frequency = 1 if temporal_agg else num_queries
    C = ee_transforms.CANONICAL_DIM

    num_rollouts = config.get("num_rollouts") or 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        ts = env.reset()
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(
                env._physics.render(
                    height=CAMERA_HEIGHT, width=CAMERA_WIDTH, camera_id=onscreen_cam
                )
            )
            plt.ion()

        if temporal_agg:
            all_time_actions = np.zeros([max_timesteps, max_timesteps + num_queries, C])
            all_time_filled = np.zeros([max_timesteps, max_timesteps + num_queries], dtype=bool)

        abs_chunk = None
        image_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                if onscreen_render:
                    image = env._physics.render(
                        height=CAMERA_HEIGHT, width=CAMERA_WIDTH, camera_id=onscreen_cam
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                obs = ts.observation
                image_list.append(obs["images"])
                ee_pose = np.array(obs["ee_pose"])
                qpos = torch.from_numpy(pre_process(ee_pose)).float().cuda().unsqueeze(0)
                curr_image = get_image(ts, camera_names)

                if t % query_frequency == 0:
                    ref = ee_pose
                    norm_actions = policy(qpos, curr_image)[0].cpu().numpy()  # (num_queries, D)
                    abs_chunk = ee_transforms.invert_action_chunk(
                        denorm_chunk(norm_actions), ref, action_repr, rot_repr
                    )  # (num_queries, 16)

                if temporal_agg:
                    all_time_actions[t, t : t + num_queries] = abs_chunk
                    all_time_filled[t, t : t + num_queries] = True
                    mask = all_time_filled[:, t]
                    cands = all_time_actions[mask, t]
                    k = 0.01
                    w = np.exp(-k * np.arange(len(cands)))
                    w = w / w.sum()
                    action16 = _aggregate_abs_poses(cands, w)
                else:
                    action16 = abs_chunk[t % query_frequency]

                ts = env.step(action16)
                rewards.append(ts.reward)
            plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])  # noqa: E711  numpy boolean-mask, drops None entries
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward == env_max_reward}"
        )

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f"video{rollout_id}.mp4"))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += (
            f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate * 100}%\n"
        )
    print(summary_str)

    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None


def prune_old_checkpoints(ckpt_dir, seed, keep):
    """Keep only the `keep` most recent periodic `policy_epoch_*_seed_{seed}.ckpt` files.
    keep <= 0 disables pruning (all checkpoints retained)."""
    if keep is None or keep <= 0:
        return
    prefix, suffix = "policy_epoch_", f"_seed_{seed}.ckpt"
    epochs = []
    for name in os.listdir(ckpt_dir):
        if name.startswith(prefix) and name.endswith(suffix):
            with contextlib.suppress(ValueError):
                epochs.append(int(name[len(prefix) : -len(suffix)]))
    for ep in sorted(epochs)[:-keep]:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(ckpt_dir, f"{prefix}{ep}{suffix}"))


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    num_checkpoints = config.get("num_checkpoints", 5)

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for data in val_dataloader:
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):  # noqa: B007  batch_idx used after loop
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            prune_old_checkpoints(ckpt_dir, seed, num_checkpoints)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves; a key may be present in only one of the two histories
    # (e.g. DiffusionFlow reports `sample_mse` during validation only)
    keys = list(train_history[0].keys())
    for k in validation_history[0]:
        if k not in keys:
            keys.append(k)
    for key in keys:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        if key in train_history[0]:
            train_values = [summary[key].item() for summary in train_history]
            plt.plot(
                np.linspace(0, num_epochs - 1, len(train_history)), train_values, label="train"
            )
        if key in validation_history[0]:
            val_values = [summary[key].item() for summary in validation_history]
            plt.plot(
                np.linspace(0, num_epochs - 1, len(validation_history)),
                val_values,
                label="validation",
            )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument(
        "--policy_class", action="store", type=str, help="policy_class, capitalize", required=True
    )
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        default=None,
        help="override task_config dataset_dir",
    )
    parser.add_argument(
        "--num_episodes",
        action="store",
        type=int,
        default=None,
        help="override task_config num_episodes",
    )
    parser.add_argument("--batch_size", action="store", type=int, help="batch_size", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False)
    parser.add_argument(
        "--dim_feedforward", action="store", type=int, help="dim_feedforward", required=False
    )
    parser.add_argument("--temporal_agg", action="store_true")

    # for DiffusionFlow (flow-matching DiT policy)
    parser.add_argument(
        "--nheads",
        action="store",
        type=int,
        default=None,
        help="attention heads (DiffusionFlow; default 8)",
    )
    parser.add_argument(
        "--dit_arch",
        action="store",
        type=str,
        default="cross_attn",
        choices=("cross_attn", "concat"),
        help="DiT conditioning: interleaved self/cross-attn vs concat-token self-attn",
    )
    parser.add_argument(
        "--dit_layers",
        action="store",
        type=int,
        default=8,
        help="number of DiT blocks (DiffusionFlow)",
    )
    parser.add_argument(
        "--n_inference_steps",
        action="store",
        type=int,
        default=4,
        help="Euler integration steps at sampling time (DiffusionFlow)",
    )
    parser.add_argument(
        "--state_dropout_prob",
        action="store",
        type=float,
        default=0.0,
        help="prob of zeroing the whole proprio vector during training (DiffusionFlow)",
    )
    parser.add_argument(
        "--cam_dropout_prob",
        action="store",
        type=float,
        default=0.0,
        help="per-camera prob of masking that view during training (DiffusionFlow)",
    )

    # task-space (end-effector) training
    parser.add_argument(
        "--task_space",
        action="store_true",
        help="train on canonical task-space EE-pose data (see ee_transforms.py)",
    )
    parser.add_argument(
        "--action_repr",
        action="store",
        type=str,
        default="absolute",
        choices=list(ee_transforms.ACTION_REPRS),
        help="UMI-style action representation (task-space only)",
    )
    parser.add_argument(
        "--rot_repr",
        action="store",
        type=str,
        default="quat",
        choices=list(ee_transforms.ROT_REPRS),
        help="rotation representation (task-space only)",
    )
    parser.add_argument(
        "--num_rollouts",
        action="store",
        type=int,
        default=None,
        help="number of eval rollouts (default 50)",
    )
    parser.add_argument(
        "--num_checkpoints",
        action="store",
        type=int,
        default=5,
        help="keep only the last N periodic checkpoints (0 = keep all)",
    )

    main(vars(parser.parse_args()))
