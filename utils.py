import os

import h5py
import IPython
import numpy as np
import torch
from torch.utils.data import DataLoader

import ee_transforms

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        task_space=False,
        action_repr="absolute",
        rot_repr="quat",
        chunk_size=None,
    ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.task_space = task_space
        self.action_repr = action_repr
        self.rot_repr = rot_repr
        self.chunk_size = chunk_size
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs["sim"]
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][
                    max(0, start_ts - 1) :
                ]  # hack, to make timesteps more aligned
                action_len = episode_len - max(
                    0, start_ts - 1
                )  # hack, to make timesteps more aligned

        self.is_sim = is_sim

        if self.task_space:
            # qpos / action stored canonically as [xyz, quat_wxyz, grip] x2 (16-dim).
            # Convert to the chosen rotation representation and action representation,
            # using the achieved EE pose at start_ts as the per-arm reference frame.
            ref = qpos
            qpos = ee_transforms.transform_state(qpos, self.rot_repr)
            action = ee_transforms.transform_action_chunk(
                action, ref, self.action_repr, self.rot_repr
            )
            state_dim = qpos.shape[-1]
            if self.action_repr != "absolute":
                # delta / relative only produce meaningful (and normalizable) values within
                # the prediction horizon; anything beyond is padded and masked out anyway.
                action_len = min(self.chunk_size, action_len)
            padded_action = np.zeros((episode_len, state_dim), dtype=np.float32)
            padded_action[: action.shape[0]] = action
        else:
            padded_action = np.zeros(original_action_shape, dtype=np.float32)
            padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0

        action_mean = self.norm_stats["action_mean"]
        action_std = self.norm_stats["action_std"]
        if action_mean.ndim == 2:
            # per-chunk-step stats (chunk_size, D); clamp beyond-horizon rows to the last.
            idx = np.minimum(np.arange(episode_len), action_mean.shape[0] - 1)
            action_mean = action_mean[idx]
            action_std = action_std[idx]
        action_data = (action_data - action_mean) / action_std
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def _clip_std(std, mask):
    """Clip std on normalizable channels; force mean0/std1 (pass-through) elsewhere."""
    std = np.array(std, dtype=np.float32)
    std[..., ~mask] = 1.0
    std[..., mask] = np.clip(std[..., mask], 1e-2, np.inf)
    return std


def _get_task_space_norm_stats(dataset_dir, num_episodes, action_repr, rot_repr, chunk_size):
    mask = ee_transforms.normalizable_mask(rot_repr)
    D = ee_transforms.state_dim(rot_repr)

    all_state = []  # transformed qpos, for state stats
    per_step = [[] for _ in range(chunk_size)]  # transformed action, grouped by horizon index
    flat_action = []  # transformed action, all steps pooled (absolute)

    canon_qpos = None
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]  # (T, 16) canonical
            action = root["/action"][()]  # (T, 16) canonical
        canon_qpos = qpos
        T = qpos.shape[0]
        all_state.append(ee_transforms.transform_state(qpos, rot_repr))

        # cap total windows so stat computation stays fast; every horizon index still
        # ends up with tens of thousands of samples.
        stride = max(1, (num_episodes * T) // 4000)
        for start_ts in range(0, T, stride):
            ref = qpos[start_ts]
            window = action[start_ts : start_ts + chunk_size]
            feat = ee_transforms.transform_action_chunk(window, ref, action_repr, rot_repr)
            if action_repr == "absolute":
                flat_action.append(feat)
            else:
                for j in range(feat.shape[0]):
                    per_step[j].append(feat[j])

    all_state = np.concatenate(all_state, axis=0)
    qpos_mean = all_state.mean(axis=0)
    qpos_std = all_state.std(axis=0)
    qpos_mean[~mask] = 0.0
    qpos_std = _clip_std(qpos_std, mask)

    if action_repr == "absolute":
        flat = np.concatenate(flat_action, axis=0)
        action_mean = flat.mean(axis=0)
        action_std = flat.std(axis=0)
        action_mean[~mask] = 0.0
        action_std = _clip_std(action_std, mask)
    else:
        action_mean = np.zeros((chunk_size, D), dtype=np.float32)
        action_std = np.ones((chunk_size, D), dtype=np.float32)
        for j in range(chunk_size):
            if not per_step[j]:
                continue
            step = np.stack(per_step[j], axis=0)
            action_mean[j] = step.mean(axis=0)
            action_std[j] = step.std(axis=0)
        action_mean[:, ~mask] = 0.0
        action_std = _clip_std(action_std, mask)

    return {
        "action_mean": action_mean.astype(np.float32),
        "action_std": action_std.astype(np.float32),
        "qpos_mean": qpos_mean.astype(np.float32),
        "qpos_std": qpos_std.astype(np.float32),
        "example_qpos": canon_qpos,
        "task_space": True,
        "action_repr": action_repr,
        "rot_repr": rot_repr,
        "chunk_size": chunk_size,
    }


def get_norm_stats(
    dataset_dir,
    num_episodes,
    task_space=False,
    action_repr="absolute",
    rot_repr="quat",
    chunk_size=None,
):
    if task_space:
        return _get_task_space_norm_stats(
            dataset_dir, num_episodes, action_repr, rot_repr, chunk_size
        )

    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            qvel = root["/observations/qvel"][()]
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    return stats


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    task_space=False,
    action_repr="absolute",
    rot_repr="quat",
    chunk_size=None,
):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(
        dataset_dir,
        num_episodes,
        task_space=task_space,
        action_repr=action_repr,
        rot_repr=rot_repr,
        chunk_size=chunk_size,
    )

    # construct dataset and dataloader
    ds_kwargs = dict(
        task_space=task_space, action_repr=action_repr, rot_repr=rot_repr, chunk_size=chunk_size
    )
    train_dataset = EpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats, **ds_kwargs
    )
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, **ds_kwargs)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
