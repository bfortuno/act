# Pre-emptive, FPM-gated DAgger demo collection

Standard DAgger records a recovery only **after** the failure (cube dropped, empty grasp), so its
data teaches resets rather than care near the failure boundary. Here a Failure Prediction Model
(FPM) gates the learner online. When the FPM predicts that the chunk about to execute leads to an
irreversible failure, the learner is frozen and an oracle expert takes over **from that exact
borderline state**, before the failure happens. Only the expert part of the episode is used for
training.

```text
Standard DAgger:  [safe] -> [near-miss] -> [FAILURE] -> recovery demo (teaches resets)
Pre-emptive:      [safe] -> [near-miss] -> [p_hat > q_alpha: expert takes over] -> clean save
```

Scope: `sim_transfer_cube_scripted`, task-space learners (ACT / DiffusionPolicy / DiffusionFlow
trained with `--task_space`, normally UMI-style `--action_repr relative`). **Deterministic on
purpose:** the recovery expert uses the fixed scripted geometry (meet point `[0, 0.5, 0.25]`, the
same grasp/meet quaternions, linear waypoint interpolation, the scripted segment durations). It has
no via-modes, timing jitter, sway or random meet point. That way the only new behavior in the data
is the recovery itself.

| File                          | Role                                                                                                                                                 |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fpm.py`                      | FPM interface (`FPMContext`, `FailurePredictor`), ground-truth `TransferCubeFailureMonitor`, built-in FPMs, window labels, conformal calibration CLI |
| `recovery_expert.py`          | `TransferCubeRecoveryExpert.plan(...)`: deterministic replanning oracle from any mid-episode state                                                   |
| `collect_preemptive_demos.py` | rollout / gate / takeover / save loop                                                                                                                |
| `utils.py`                    | loader honours `/is_expert` (only expert steps sampled; chunks truncated at the end of the expert segment)                                           |

## Workflow

```bash
source wsl_gl.sh headless
BASE=data/sim_transfer_cube_ts          # deterministic task-space base demos (record_sim_episodes.py --task_space)
CKPT=outputs/<learner_run>              # learner trained on $BASE

# 1) observe: learner only, log FPM scores + ground-truth failure times (no interventions)
uv run python collect_preemptive_demos.py --ckpt_dir $CKPT --dataset_dir runs/observe \
    --fpm oracle --observe_only --num_rollouts 100

# 2) calibrate: split-conformal threshold, per-episode false-alarm rate <= alpha
uv run python fpm.py calibrate --scores runs/observe/fpm_scores.npz --alpha 0.1

# 3) collect: append pre-emptive recovery episodes to a COPY of the base dataset
cp -r $BASE data/sim_transfer_cube_preemptive
uv run python collect_preemptive_demos.py --ckpt_dir $CKPT \
    --dataset_dir data/sim_transfer_cube_preemptive \
    --fpm oracle --fpm_threshold <q from step 2> --num_rollouts 100 --save_video

# 4) retrain on base + recoveries (the collector prints the --num_episodes to use)
uv run python imitate_episodes.py --task_space --action_repr relative --rot_repr rot6d \
    --dataset_dir data/sim_transfer_cube_preemptive --num_episodes <N> ...
```

## FPMs

`FPMContext` gives f(o*t, L, A*{t:t+H}) its inputs:

- `observation`: images per camera and `ee_pose`.
- `language`: the task name.
- `rel_chunk`: what the policy emitted, de-normalized, UMI-relative to `ref_pose`, in its
  `rot_repr`.
- `norm_chunk`.
- `abs_chunk`: canonical absolute (H, 16).
- `pending_abs`: the part still to be executed from `t`.
- `chunk_t`.
- `physics`, `env`, `monitor`: privileged; only filled in for FPMs with `privileged=True`.

| `--fpm`           | What                                                                                                                                                                                                                   | Notes                                                                                                                                                 |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `oracle`          | `LookaheadOracleFPM`: simulates `pending_abs` (no rendering) from a physics snapshot, runs the failure monitor, restores state bit-exactly. Score = γ^k (γ=0.98), where k is the steps to the first predicted failure. | Privileged upper bound. Threshold = time-to-failure gate: `p > thr` ⇔ failure within `log(thr)/log(γ)` steps (0.5 → ~34 steps).                       |
| `acc`             | `ChunkConsistencyFPM`: disagreement between overlapping receding-horizon chunks, compared in absolute space.                                                                                                           | Training-free and black-box. Needs overlapping chunks: `--temporal_agg` or `--query_every < chunk_size`.                                              |
| `pkg.mod:factory` | Your learned FPM, e.g. a dual-stream DINOv2/SigLIP + TCN guardian.                                                                                                                                                     | `factory(**json.loads(--fpm_kwargs))` must return a callable `(FPMContext) -> float`. Optionally it can define `reset(env)` and `privileged = False`. |

Ground-truth failure events (`TransferCubeFailureMonitor`, latched, first wins). All are geometric
and must hold for 10 consecutive steps (0.2 s), so a one-step contact flicker while the fingers
slide on the cube never fires; `tau_fail` is the first step of that streak:

- `drop`: the cube was really lifted (more than 2 cm above its resting height) and is back at table
  height with no finger contact.
- `missed_grasp`: the cube was never lifted; the right gripper opened earlier and is now commanded
  closed, its commanded (mocap) height is more than 6 cm above the cube, and the cube is on the
  table without right-finger contact.
- `off_table`: the cube is more than 3 cm below its resting height.
- `missed_handover`: the cube is lifted and held by the right arm only, and the left gripper (which
  opened earlier) has closed past the cube width (achieved opening < 0.45; closed on the cube it is
  ~0.60) with no left-finger contact.
- `stalled_grasp`: the right fingers first touched the cube 75 steps ago, but it was never lifted
  and is still on the table (a fumbled grasp that would otherwise only end in a timeout). `tau_fail`
  is the first touch.

FPM labels follow the env's success criterion: a rollout the env counts as successful is never
labelled as failing, even if the monitor latched an event (the event stays in the attrs).

The expert decides who holds the cube from contact _history_ (touched within the last 15 steps) and
the cube's height, not from one step's contacts. Even a firm grasp can show zero contacts for a step
while the cube is in the air.

Workspace: with the deterministic scripted grasp, cubes at x > ~0.26 cannot be grasped (even by the
expert). Keep `sample_box_pose` at x ≤ 0.25, otherwise those rollouts are unrecoverable.

Each saved episode has `/monitor_trace` (T, 4): touch_left_arm, touch_right_arm, touch_table and
cube_z after every step, plus `/observations/env_state` (cube pose), so you can check any event
against what the detector saw.

Interventions are pre-emptive only: once an event has fired, the gate no longer triggers.

## Recovery expert

`plan()` reads the phase from privileged contacts:

- **pick**: retract up if low, hover above the _current_ cube pose, descend, close, carry, meet,
  hand over, retreat.
- **carry**: keep the grasp orientation, go to meet, hand over, retreat.
- **handover**: left closes and retreats, then right opens and retreats.
- **off_table**: unrecoverable.

The plan starts at the current mocap target and the last commanded gripper, so there is no jump at
the handoff. If it does not fit in the remaining steps, it is uniformly time-compressed. If that
would need more than a 2× speed-up, `plan()` returns `None`; the episode is logged as unrecoverable
and not saved.

## Saved episode format

The base task-space HDF5 format (`/observations/qpos`, `/action`, `/observations/images/*`;
canonical absolute poses), plus:

| key                                                         | shape                   | meaning                                                                                                                                          |
| ----------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/is_expert`                                                | (T,) bool               | training mask: the loader samples only these steps                                                                                               |
| `/fpm_score`                                                | (T,)                    | p̂ at each check, NaN elsewhere                                                                                                                   |
| `/fpm_label`                                                | (T,) int8               | 1 in [τ−H, τ], −1 (ignore) in (τ−2H, τ−H), 0 otherwise or for successful rollouts                                                                |
| `/query_mask`, `/proposed_rel_chunk`, `/proposed_abs_chunk` | (T,), (T,H,D), (T,H,16) | learner proposals at query steps: (o*t, A*{t:t+H}) pairs for training an FPM                                                                     |
| attrs                                                       |                         | `intervened`, `interventions` (json: t0, phase, time_scale, p), `success`, `tau_fail`, `fail_event`, `learner_ckpt`, `fpm`, `fpm_threshold`, ... |

`--observe_only` runs save every complete rollout, including failures: this is the FPM training set.
Collection runs save, by default, only episodes that were intervened and then succeeded (`--save`).

## Caveats

- Episodes must keep the task's fixed `episode_len` (the dataloader stacks full-length tensors). The
  learner prefix is stored but masked out by `/is_expert`.
- `--handback_after N` gives control back to the learner after N expert steps, allowing several
  interventions per episode (`--max_interventions`). The loader truncates chunks at the end of each
  expert segment, so learner actions never become labels.
- `lerobot_bridge/convert_dataset.py` does not carry `/is_expert` yet. Converting these episodes to
  LeRobot format would train on the learner prefix too.
