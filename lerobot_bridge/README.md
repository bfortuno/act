# lerobot_bridge

Run **upstream LeRobot** ACT and Diffusion Policy on this repo's simulated
task-space datasets, as a parallel track to the in-repo `imitate_episodes.py`
stack. The root ACT project (Python 3.8 / torch 2.0) is never modified.

## Why a separate environment

LeRobot 0.6 (dataset **v3**) requires Python >= 3.12 and numpy >= 2, incompatible
with the root repo's pins. This directory is its own `uv` project with its own
`.venv`. Shared modules (`ee_transforms`, `utils`, `ee_sim_env`, `constants`) are
imported from the repo root via `_shared.py` (a `sys.path` insert) - single source
of truth, no vendored copies.

```bash
cd lerobot_bridge
uv sync                 # training deps (lerobot, torch, ...)
uv sync --group sim     # + current mujoco/dm-control for eval_sim.py
```

## Pipeline

### 1. Convert HDF5 -> LeRobot dataset v3

```bash
uv run python convert_dataset.py \
    --source-dir /path/to/sim_transfer_cube_scripted \
    --out-dir outputs/data/ts_cube --task-name sim_transfer_cube_scripted \
    --num-episodes 400 --verify
```

Stores canonical raw poses (16-dim task-space or 14-dim joint-space) + camera
video; a sidecar `meta/act_bridge.json` carries the task-space flag, source dir
and per-episode lengths. `--verify` reloads and checks round-trip fidelity.

### 1b. Or record directly into a LeRobot dataset (no HDF5)

```bash
MUJOCO_GL=egl uv run --group sim python record_dataset.py \
    --task-name sim_transfer_cube_scripted --out-dir outputs/data/ts_cube \
    --num-episodes 400 --seed 0   # --only-success to discard failed rollouts
```

Runs the scripted policy in `ee_sim_env` and streams each step into the dataset:
same stored quantities as `record_sim_episodes.py --task_space` + the converter
(task-space only), same sidecar with `source_dir: null` plus per-episode `success`.
Norm stats for these datasets are computed from the dataset's own parquet columns.
The rollout uses this project's mujoco (see the sim-fidelity caveat below), so the
episodes are not the ones the in-repo HDF5 recorder would produce.

### 2. Train (LeRobot ACT or Diffusion)

`EEReprDataset` applies the `(rot_repr, action_repr)` transform + per-chunk-step
z-score from `utils.get_norm_stats` on the fly; the policy is configured with
IDENTITY normalization (LeRobot's processor pipeline is bypassed). Stats are
cached to `meta/ee_repr_stats_<rot>_<act>_c<chunk>.npz`.

```bash
# ACT  - target regime: task-space rot6d / relative / chunk 32
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --dataset-root outputs/data/ts_cube --out-dir outputs/lerobot_act_ts_rot6d_rel \
    --policy act --rot-repr rot6d --action-repr relative --chunk-size 32 \
    --steps 100000 --batch-size 64 --lr 1e-5 --seed 0 --wandb

# Diffusion Policy (DDPM)
CUDA_VISIBLE_DEVICES=1 uv run python train.py \
    --dataset-root outputs/data/ts_cube --out-dir outputs/lerobot_dp_ts_rot6d_rel \
    --policy diffusion --rot-repr rot6d --action-repr relative --chunk-size 32 \
    --steps 100000 --batch-size 64 --lr 1e-4 --seed 0 --wandb
```

Other representations work too: `--rot-repr {quat,rpy,rot6d}`,
`--action-repr {absolute,delta,relative}`; joint-space datasets are handled as a
passthrough (state/action used as stored). Checkpoints (`best/`, `last/`) hold the
`save_pretrained` model + the stats npz + `run_config.json`.

### 3. Eval in `ee_sim_env`

```bash
MUJOCO_GL=egl uv run --group sim python eval_sim.py \
    --ckpt-dir outputs/lerobot_act_ts_rot6d_rel/best --num-rollouts 50   # --temporal-agg
```

Mirrors `imitate_episodes.py:eval_bc_task_space` (denorm -> `invert_action_chunk`
-> optional SO(3) temporal agg -> `env.step`). Writes `result_<tag>.txt`,
`eval_result.json`, `video*.mp4` into the checkpoint dir - same layout as the
in-repo baselines in `outputs/`.

**Sim-fidelity caveat:** the bridge uses a current mujoco/dm-control (`mujoco 3.3`,
no cp312 wheel exists for the root repo's `mujoco 2.3.7`). Contact dynamics may
differ slightly from the in-repo numbers. If a comparison looks off, evaluate the
same checkpoint through a two-process bridge (bridge serves policy actions; the
root py3.8 env runs `ee_sim_env` with mujoco 2.3.7) - not yet implemented.

## Development loop status

- [x] Loop 0 - skeleton + isolated env (`smoke_test.py`)
- [x] Loop 1 - HDF5 -> LeRobot v3 converter (`convert_dataset.py --verify`)
- [x] Loop 2 - representation wrapper + norm stats (`ee_repr_dataset.py`, `test_loop2.py`)
- [x] Loop 3 - thin trainer: LeRobot ACT (`train.py --policy act`)
- [x] Loop 4 - thin trainer: LeRobot Diffusion Policy (`train.py --policy diffusion`)
- [x] Loop 5 - direct eval in `ee_sim_env` (`eval_sim.py`)
- [ ] Loop 6 - real 400-episode runs on the 2-GPU machine + parity table vs `outputs/`

All loop exit tests pass on 2-episode fixtures (`_fixtures/ts`, `_fixtures/js`) and
an 8 GB GPU. Loop 6 is the real training run on the other machine.
