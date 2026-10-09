# ACT: Action Chunking with Transformers

### *New*: [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

#### Project Website: https://tonyzhaozh.github.io/aloha/

This repo contains the implementation of ACT, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate ACT in sim or real.
For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments — pick-and-handover demos are multimodal/human-like, see [README_scripted_policy.md](README_scripted_policy.md)
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

This repo uses [uv](https://docs.astral.sh/uv/) for dependency management.
Install uv, then from the repo root run:

    uv sync

This creates a `.venv` with all dependencies (including the local `detr` package) pinned in `uv.lock`.

### Example Usages

Run every script through `uv run` from the repo root (no manual venv activation needed), e.g.

    cd <path to act repo>
    uv run python record_sim_episodes.py ...

`uv run` transparently keeps the environment in sync with `pyproject.toml` / `uv.lock` before each call.

#### Running under WSL

The default MuJoCo GL backend (GLX/GLFW) needs a real X display and tends to abort
under WSL with `xcb ... Aborting`. Use `wsl_gl.sh` to pick a working setup once per
shell, then run the scripts normally:

    source wsl_gl.sh headless     # offscreen only (data gen, training, eval videos) - EGL + matplotlib Agg
    source wsl_gl.sh onscreen     # also show the live --onscreen_render window via WSLg X - EGL + matplotlib TkAgg
    source wsl_gl.sh status       # print the current settings and run a render self-test

    uv run python imitate_episodes.py ...

Or wrap a single command without sourcing:

    ./wsl_gl.sh headless uv run python record_sim_episodes.py --task_name sim_transfer_cube_scripted ...

Both modes render on the GPU via EGL (this repo only ever renders offscreen); the
`onscreen` mode additionally wires up `$DISPLAY` and an interactive matplotlib
backend for the live preview. To manage it yourself, just export `MUJOCO_GL`
(`egl`, or `glfw` if you have a working X display) and `MPLBACKEND` before running
and skip the script.

### Simulated experiments

We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generated 50 episodes of scripted data, run:

    uv run python record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

    uv run python visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To train ACT:

    # Transfer Cube task
    uv run python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.
