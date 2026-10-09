# Scripted policy: multimodal, human-like pick-and-handover demos

`scripted_policy.py` generates the expert demonstrations used to train ACT /
DiffusionPolicy / DiffusionFlow. Originally it produced near-deterministic,
piecewise-linear trajectories (only the cube's x/y start position was randomized);
every episode handed the cube off at the same point, along the same straight-line
path, at the same speed. A unimodal, linear dataset gives a unimodal regressor (ACT)
no real disadvantage against a multimodal policy (DiffusionPolicy/DiffusionFlow) and
nothing for either to actually learn path diversity from.

This change makes `sim_transfer_cube_scripted` demos (a) kinematically human-like
(smooth, eased reaches instead of straight lines with instant stops) and (b)
genuinely multimodal (same functional bottlenecks — grasp, handover, release — reached
by different discrete paths and different handover points episode to episode).

| | |
|---|---|
| Motion primitive | `scripted_policy.py` — `BasePolicy._build_motion` / `_query_motion` |
| Demo randomization | `scripted_policy.py` — `PickAndTransferPolicy.generate_trajectory` |
| Cube yaw randomization | `utils.py` — `sample_box_pose` |
| Scope | `sim_transfer_cube_scripted` only; `InsertionPolicy` inherits the shared motion primitive but its own waypoints are untouched; `ee_sim_env.py` / `sim_env.py` / `record_sim_episodes.py` unchanged |

---

## Motion primitive (`BasePolicy`)

The original `interpolate()` blended two adjacent waypoints linearly in both position
and (unnormalized) quaternion — every reach was a straight line with a velocity
discontinuity at each waypoint.

`_build_motion` now fits, over the full waypoint list for one arm:
- **Position**: a Catmull-Rom cubic Hermite spline (`scipy.interpolate.CubicHermiteSpline`,
  finite-difference tangents, zero tangent at the true first/last waypoint so the arm
  doesn't overshoot past its start/hold pose).
- **Orientation**: `scipy.spatial.transform.Slerp` over the waypoint quaternions
  (converted wxyz → xyzw, same convention as `ee_transforms.py`).
- **Gripper**: linear blend using the same warped time fraction as position/orientation
  (not its own spline), so it can never leave `[0, 1]`.

**Two-level minimum-jerk time-warp.** A naive per-adjacent-pair warp would decelerate to
near-zero velocity at *every* waypoint, including inserted via-points — reintroducing
stop-start motion. Instead, each waypoint is tagged `"kind": "bottleneck"` (originally
authored) or `"via"` (inserted, see below); the quintic warp `s = 6u^5 - 15u^4 + 10u^3`
is applied only across bottleneck-to-bottleneck intervals, using the *outer* interval's
progress to compute `u`. Via-points are interior to one warped interval, so they're
passed through at full speed — only true bottlenecks (grasp, meet, release) get the
eased stop a real reach would have. Because `s(0)=0, s(1)=1` exactly, every bottleneck
still reproduces its authored `xyz`/`quat`/`gripper` at its exact `t` — this keeps the
change a drop-in: same `episode_len`, same waypoint dict schema
(`{"t","xyz","quat","gripper","kind"}`, `"kind"` defaults to `"bottleneck"` so
`InsertionPolicy`'s unmodified waypoint lists are unaffected).

## Demo randomization (`PickAndTransferPolicy`)

- **Handover point**: `meet_xyz` was a fixed `[0, 0.5, 0.25]`; now sampled per episode
  (`x ∈ [-0.05, 0.05]`, `y ∈ [0.45, 0.55]`, `z ∈ [0.22, 0.28]`).
- **Cube yaw**: `sample_box_pose()` now samples `yaw ∈ [-π/4, π/4]` (the cube is a
  symmetric 0.02³ box, so ±45° covers its full unique rotational symmetry class without
  aliasing). The grasp-approach quaternion is derived from the box's actual yaw
  (read back out of `env_state`, not re-sampled independently) instead of using a fixed
  approach angle.
- **Discrete via-point "funnel" diversity**: one mode per episode —
  `direct` / `arc_high` / `arc_left` / `arc_right` (fixed 3D offsets) — inserted as one
  extra via-point on each of 6 transit segments (right arm: sleep→approach-cube,
  grasp→approach-meet, release→retreat; left arm: sleep→approach-meet,
  approach-meet→move-to-meet, receive→retreat). Grasp/meet/release/receive waypoints
  are never modified. One mode per episode (not per segment) so episodes form separable
  behavioral clusters instead of washing out into noise.
- **Timing jitter**: `r ~ U(0.85, 1.15)` scales waypoint `t`, anchored at the shared
  handover instant `t=310` (present in both arms' trajectories) so both arms stay
  synchronized at the handoff regardless of `r`; everything after is rescaled to still
  land exactly on `t=400`.

Order of operations in `generate_trajectory`: compute bottleneck waypoints (randomized
`meet_xyz` + yaw-derived grasp quat) → insert via-points → apply timing jitter → return.
`__call__` builds the spline/Slerp motion once at `step_count==0` and queries it per step.

## Verification

- 50-seed stress test of trajectory construction: strictly increasing `t`, exact episode
  end at `t=400`, exact reproduction of every bottleneck waypoint, finite/bounded outputs.
- Full MuJoCo rollouts: `sim_transfer_cube_scripted` 8/8 and `sim_insertion_scripted` 6/6
  episodes reach `max_reward == 4` (insertion was the regression risk flagged for the
  shared motion-primitive change, since it's precision-sensitive — unaffected).
- `record_sim_episodes.py --task_name sim_transfer_cube_scripted --num_episodes 10`:
  10/10 successful, with visibly varying episode returns (463–692) confirming the
  randomization is actually taking effect, not just the pre-existing cube x/y jitter.

## Explicitly out of scope (deferred)

- Role swap (which arm picks vs. receives).
- Condition/precondition-triggered phase transitions (state machine replacing fixed
  timesteps) — the funnel-in-space approach here keeps fixed timesteps but varies the
  path and handover point between them.
