"""Pre-emptive, FPM-gated DAgger demo collection (sim_transfer_cube, task-space learners).

Loop per rollout:
  learner (trained ckpt, UMI-relative chunks) acts in ee_sim_env
  -> every --fpm_every steps the FPM scores the chunk still to be executed
  -> p_hat > --fpm_threshold: freeze learner, the deterministic replanning oracle
     (recovery_expert.py) takes over from that exact state and finishes the episode
  -> save the episode in the canonical task-space HDF5 format + /is_expert mask, so only
     expert steps are trained on (utils.EpisodicDataset honours the mask).

--observe_only never intervenes: it logs FPM scores + ground-truth tau_fail/labels for
conformal calibration (fpm.py calibrate) and for training a learned FPM.

See README_preemptive_dagger.md.
"""

import argparse
import glob
import inspect
import json
import os
import pickle
import shutil
import subprocess
import sys

import h5py
import numpy as np
import torch

import ee_transforms
from constants import DT, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from fpm import FPMContext, TransferCubeFailureMonitor, make_fpm, window_labels
from imitate_episodes import _aggregate_abs_poses, get_image, make_policy
from record_sim_episodes import save_task_space_episode
from recovery_expert import TransferCubeRecoveryExpert
from utils import set_seed

# --------------------------------------------------------------------------------------
# learner
# --------------------------------------------------------------------------------------


class Learner:
    """Task-space learner restored from a training ckpt_dir; mirrors
    imitate_episodes.eval_bc_task_space (query -> denorm -> invert to absolute)."""

    def __init__(self, ckpt_dir, ckpt_name="policy_best.ckpt", num_inference_steps=None):
        with open(os.path.join(ckpt_dir, "run_config.json")) as f:
            rc = json.load(f)
        if not rc.get("task_space"):
            raise ValueError("only task-space (EE) learners are supported")
        self.policy_class = rc["policy_class"]
        pc = dict(rc["policy_config"])
        if num_inference_steps is not None:
            pc["n_inference_steps"] = num_inference_steps
        self.policy_config = pc
        # detr/main.py's builders call parser.parse_args() on sys.argv (ACT quirk) and then
        # override every field from policy_config; hand them a minimal valid argv.
        argv = sys.argv
        sys.argv = [
            argv[0],
            "--policy_class",
            self.policy_class,
            "--task_name",
            "sim",
            "--seed",
            "0",
            "--num_epochs",
            "1",
            "--ckpt_dir",
            ckpt_dir,
        ]
        try:
            self.policy = make_policy(self.policy_class, pc)
        finally:
            sys.argv = argv
        status = self.policy.load_state_dict(
            torch.load(os.path.join(ckpt_dir, ckpt_name)), strict=False
        )
        if status.missing_keys or status.unexpected_keys:
            print(
                f"[learner] load: {len(status.missing_keys)} missing, "
                f"{len(status.unexpected_keys)} unexpected keys"
            )
        self.policy.cuda().eval()
        with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "rb") as f:
            stats = pickle.load(f)
        self.action_repr = stats.get("action_repr", rc["action_repr"])
        self.rot_repr = stats.get("rot_repr", rc["rot_repr"])
        self.stats = stats
        self.num_queries = pc["num_queries"]
        self.camera_names = pc["camera_names"]
        self.ckpt = os.path.join(ckpt_dir, ckpt_name)

    def _denorm(self, norm_chunk):
        m, s = self.stats["action_mean"], self.stats["action_std"]
        if m.ndim == 2:
            idx = np.minimum(np.arange(norm_chunk.shape[0]), m.shape[0] - 1)
            return norm_chunk * s[idx] + m[idx]
        return norm_chunk * s + m

    @torch.inference_mode()
    def query(self, ts):
        """-> (ref_pose, norm_chunk, rel_chunk, abs_chunk)"""
        ee_pose = np.array(ts.observation["ee_pose"])
        state = ee_transforms.transform_state(ee_pose, self.rot_repr)
        state = (state - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        qpos = torch.from_numpy(state).float().cuda().unsqueeze(0)
        norm = self.policy(qpos, get_image(ts, self.camera_names))[0].cpu().numpy()
        rel = self._denorm(norm)
        abs_chunk = ee_transforms.invert_action_chunk(rel, ee_pose, self.action_repr, self.rot_repr)
        return ee_pose, norm, rel, abs_chunk


class Executor:
    """Turns learner queries into per-step actions (chunking or temporal aggregation).
    Counter-based so it can be restarted after a handback."""

    def __init__(self, learner, T, temporal_agg, query_every):
        self.L, self.T, self.agg = learner, T, temporal_agg
        H = learner.num_queries
        self.query_every = query_every or (1 if temporal_agg else H)
        self.H = H
        self.restart()

    def restart(self):
        self.since_query = None
        self.chunk = None  # dict(t, ref, norm, rel, abs)
        if self.agg:
            self.buf = np.zeros([self.T, self.T + self.H, ee_transforms.CANONICAL_DIM])
            self.filled = np.zeros([self.T, self.T + self.H], dtype=bool)

    def step(self, t, ts):
        """Returns (action16, queried_now)."""
        queried = False
        if self.since_query is None or self.since_query >= self.query_every:
            ref, norm, rel, ab = self.L.query(ts)
            self.chunk = {"t": t, "ref": ref, "norm": norm, "rel": rel, "abs": ab}
            self.since_query = 0
            queried = True
            if self.agg:
                self.buf[t, t : t + self.H] = ab
                self.filled[t, t : t + self.H] = True
        off = t - self.chunk["t"]
        if self.agg:
            mask = self.filled[:, t]
            cands = self.buf[mask, t]
            w = np.exp(-0.01 * np.arange(len(cands)))
            action = _aggregate_abs_poses(cands, w / w.sum())
        else:
            action = self.chunk["abs"][min(off, self.H - 1)]
        self.since_query += 1
        return action, queried

    def pending(self, t):
        off = t - self.chunk["t"]
        return self.chunk["abs"][min(off, self.H - 1) :]


# --------------------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------------------


def rollout(env, learner, fpm, args, T, task_name):
    ex = Executor(learner, T, args.temporal_agg, args.query_every)
    ts = env.reset()
    fpm.reset(env)
    monitor = TransferCubeFailureMonitor(record=True, episode_len=T)
    monitor.reset(env.physics)

    H, D = learner.num_queries, None
    episode, actions = [ts], []
    is_expert = np.zeros(T, dtype=bool)
    fpm_score = np.full(T, np.nan, dtype=np.float32)
    query_mask = np.zeros(T, dtype=bool)
    prop_rel, prop_abs = None, np.full((T, H, ee_transforms.CANONICAL_DIM), np.nan, np.float32)
    interventions = []
    predictions = []  # every check where the FPM predicted a failure (p > 0)
    expert, expert_steps, last_action = None, 0, None
    unrecoverable = False
    r_max = 0

    for t in range(T):
        if expert is None:
            action, queried = ex.step(t, ts)
            if queried:
                query_mask[t] = True
                c = ex.chunk
                if prop_rel is None:
                    D = c["rel"].shape[-1]
                    prop_rel = np.full((T, H, D), np.nan, np.float32)
                prop_rel[t], prop_abs[t] = c["rel"], c["abs"]
            if t >= args.fpm_warmup and (t - args.fpm_warmup) % args.fpm_every == 0:
                c = ex.chunk
                ctx = FPMContext(
                    t=t,
                    observation=ts.observation,
                    language=task_name,
                    chunk_t=c["t"],
                    ref_pose=c["ref"],
                    norm_chunk=c["norm"],
                    rel_chunk=c["rel"],
                    abs_chunk=c["abs"],
                    pending_abs=ex.pending(t),
                    action_repr=learner.action_repr,
                    rot_repr=learner.rot_repr,
                )
                if fpm.privileged:
                    ctx.physics, ctx.env, ctx.monitor = env.physics, env, monitor
                p = float(fpm(ctx))
                fpm_score[t] = p
                pred = getattr(fpm, "last_prediction", None) if p > 0 else None
                if pred is not None:
                    predictions.append({"t": t, "p": round(p, 4), **pred})
                if (
                    not args.observe_only
                    and p > args.fpm_threshold
                    and len(interventions) < args.max_interventions
                    and monitor.event is None  # pre-emptive only: not after the failure
                ):
                    expert = TransferCubeRecoveryExpert.plan(
                        env.physics, ts.observation, last_action, t, T, monitor=monitor
                    )
                    if expert is None:
                        unrecoverable = True
                        interventions.append({"t0": t, "p": p, "phase": "unrecoverable"})
                        break
                    interventions.append({**expert.describe(), "p": p, "predicted": pred})
                    expert_steps = 0
        if expert is not None:
            action = expert()
            is_expert[t] = True
            expert_steps += 1

        actions.append(np.asarray(action))
        ts = env.step(action)
        episode.append(ts)
        last_action = action
        r_max = max(r_max, ts.reward)
        monitor.update(env.physics, action, t)

        # optional handback: after N expert steps, let the learner try again
        if (
            expert is not None
            and args.handback_after
            and expert_steps >= args.handback_after
            and len(interventions) < args.max_interventions
        ):
            expert = None
            ex.restart()

    success = (not unrecoverable) and r_max == env.task.max_reward
    tau = monitor.t_event
    return {
        "episode": episode,
        "actions": actions,
        "is_expert": is_expert,
        "fpm_score": fpm_score,
        "query_mask": query_mask,
        "prop_rel": prop_rel,
        "prop_abs": prop_abs,
        "interventions": interventions,
        "predictions": predictions,
        "unrecoverable": unrecoverable,
        "success": bool(success),
        "r_max": int(r_max),
        "tau_fail": tau,
        "fail_event": monitor.event,
        "monitor_trace": np.array(monitor.trace, dtype=np.float32),
        "complete": len(actions) == T,
    }


def should_save(res, mode, observe_only):
    if not res["complete"]:
        return False
    if observe_only:
        return True
    intervened = len(res["interventions"]) > 0
    if mode == "all":
        return True
    if mode == "intervened":
        return intervened
    return intervened and res["success"]  # intervened_success


def save_episode(res, dataset_dir, idx, camera_names, T, meta, label_horizon):
    save_task_space_episode(dataset_dir, idx, res["episode"], res["actions"], camera_names, T)
    path = os.path.join(dataset_dir, f"episode_{idx}.hdf5")
    tau = res["tau_fail"]
    # a failed rollout without an identified event is labelled as failing at the timeout
    # labels follow the env's success criterion: a rollout the env counts as successful
    # is never labelled as failing (the monitor event, if any, stays in the attrs)
    timeout_tau = T - 1 if tau is None else tau  # failure without an event: the timeout
    tau_label = None if res["success"] else timeout_tau
    with h5py.File(path, "a") as root:
        root.create_dataset("is_expert", data=res["is_expert"])
        root.create_dataset("fpm_score", data=res["fpm_score"])
        root.create_dataset("fpm_label", data=window_labels(T, tau_label, label_horizon))
        root.create_dataset("query_mask", data=res["query_mask"])
        # debugging / replay: what the failure monitor saw after each step
        # columns: touch_left_arm, touch_right_arm, touch_table, cube_z
        root.create_dataset("monitor_trace", data=res["monitor_trace"][:T])
        root.create_dataset(
            "observations/env_state",
            data=np.stack([res["episode"][t].observation["env_state"] for t in range(T)]),
        )
        if res["prop_rel"] is not None:
            root.create_dataset("proposed_rel_chunk", data=res["prop_rel"], compression="gzip")
        root.create_dataset("proposed_abs_chunk", data=res["prop_abs"], compression="gzip")
        root.attrs["preemptive_dagger"] = True
        root.attrs["intervened"] = len(res["interventions"]) > 0
        root.attrs["interventions"] = json.dumps(res["interventions"])
        root.attrs["success"] = res["success"]
        root.attrs["tau_fail"] = -1 if tau is None else int(tau)
        root.attrs["tau_fail_is_timeout"] = tau is None and not res["success"]
        root.attrs["fail_event"] = res["fail_event"] or ""
        for k, v in meta.items():
            root.attrs[k] = v


def save_marked_video(res, path, cams=("top",)):
    from visualize_episodes import save_videos

    frames = []
    for t in range(len(res["actions"])):
        d = {}
        for c in cams:
            img = res["episode"][t].observation["images"][c].copy()
            if res["is_expert"][t]:
                img[:6], img[-6:], img[:, :6], img[:, -6:] = 255, 255, 255, 255
                img[:6, :, 1:], img[-6:, :, 1:], img[:, :6, 1:], img[:, -6:, 1:] = 0, 0, 0, 0
            d[c] = img
        frames.append(d)
    save_videos(frames, DT, video_path=path)
    to_h264(path)


def to_h264(mp4_path):
    """cv2 writes MPEG-4 Part 2 ('mp4v'), which VS Code's media preview and browsers
    cannot decode. If ffmpeg is available, re-encode in place to H.264 (yuv420p,
    faststart) so the .mp4 plays everywhere."""
    if shutil.which("ffmpeg") is None:
        return
    tmp = mp4_path + ".h264.mp4"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        mp4_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        tmp,
    ]
    if subprocess.run(cmd).returncode == 0:
        os.replace(tmp, mp4_path)
    else:
        print(f"[collect] ffmpeg could not re-encode {mp4_path} to H.264; left as mp4v")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def make_env(task_name, camera_height=None, camera_width=None):
    """Build ee_sim_env exactly like imitate_episodes.py --eval does (no size args), so the
    learner sees images at the resolution it was trained on. Some checkouts render a fixed
    size and take no camera kwargs; others accept them. Explicit --camera_* flags are only
    forwarded when given, and fail loudly if this env cannot honour them."""
    if camera_height is None and camera_width is None:
        return make_ee_sim_env(task_name)
    params = inspect.signature(make_ee_sim_env).parameters
    if "camera_height" not in params or "camera_width" not in params:
        raise SystemExit(
            "[collect] this checkout's make_ee_sim_env renders a fixed size and takes no "
            "camera_height/camera_width; drop --camera_height/--camera_width"
        )
    kw = {}
    if camera_height is not None:
        kw["camera_height"] = camera_height
    if camera_width is not None:
        kw["camera_width"] = camera_width
    return make_ee_sim_env(task_name, **kw)


def check_output_dir(dataset_dir, observe_only):
    """Never mix learner-only observe rollouts with training data: --observe_only must
    write to a dir holding only observe rollouts, and collection must not append to one."""
    for path in sorted(glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))):
        with h5py.File(path, "r") as root:
            is_observe = bool(root.attrs.get("observe_only", False))
        if observe_only and not is_observe:
            raise SystemExit(
                f"[collect] {dataset_dir} already holds training episodes ({os.path.basename(path)}). "
                "--observe_only saves learner-only rollouts (no expert labels); write them to a "
                "separate dir, e.g. --dataset_dir runs/observe"
            )
        if not observe_only and is_observe:
            raise SystemExit(
                f"[collect] {dataset_dir} contains --observe_only rollouts "
                f"({os.path.basename(path)}); collect into a copy of the base dataset instead"
            )


def main(args):
    set_seed(args.seed)
    task_name = args.task_name
    if "sim_transfer_cube" not in task_name:
        raise NotImplementedError("recovery expert / failure monitor are transfer-cube only")
    cfg = SIM_TASK_CONFIGS[task_name]
    T = cfg["episode_len"]
    camera_names = cfg["camera_names"]

    learner = Learner(args.ckpt_dir, args.ckpt_name, args.n_inference_steps)
    fpm = make_fpm(args.fpm, json.loads(args.fpm_kwargs) if args.fpm_kwargs else None)
    label_horizon = args.label_horizon or learner.num_queries

    os.makedirs(args.dataset_dir, exist_ok=True)
    check_output_dir(args.dataset_dir, args.observe_only)
    if args.start_episode_idx is None:
        idx = len(glob.glob(os.path.join(args.dataset_dir, "episode_*.hdf5")))
    else:
        idx = args.start_episode_idx
    print(f"[collect] writing from episode_{idx} into {args.dataset_dir}")

    meta = {
        "learner_ckpt": learner.ckpt,
        "learner_policy_class": learner.policy_class,
        "action_repr": learner.action_repr,
        "rot_repr": learner.rot_repr,
        "fpm": fpm.name,
        "fpm_threshold": float(args.fpm_threshold),
        "observe_only": bool(args.observe_only),
        "label_horizon": int(label_horizon),
    }

    env = make_env(task_name, args.camera_height, args.camera_width)
    summary, all_scores, all_success, all_tau = [], [], [], []
    n_saved = 0
    for r in range(args.num_rollouts):
        res = rollout(env, learner, fpm, args, T, task_name)
        saved_as = None
        if should_save(res, args.save, args.observe_only):
            save_episode(res, args.dataset_dir, idx, camera_names, T, meta, label_horizon)
            if args.save_video:
                save_marked_video(res, os.path.join(args.dataset_dir, f"episode_{idx}.mp4"))
            saved_as, idx, n_saved = idx, idx + 1, n_saved + 1
        row = {
            "rollout": r,
            "success": res["success"],
            "r_max": res["r_max"],
            "interventions": res["interventions"],
            "predictions": res["predictions"],
            "unrecoverable": res["unrecoverable"],
            "fail_event": res["fail_event"],
            "tau_fail": res["tau_fail"],
            "max_fpm": float(np.nanmax(res["fpm_score"]))
            if np.isfinite(res["fpm_score"]).any()
            else None,
            "saved_as": saved_as,
        }
        summary.append(row)
        all_scores.append(res["fpm_score"])
        all_success.append(res["success"])
        all_tau.append(-1 if res["tau_fail"] is None else res["tau_fail"])
        print(
            f"[rollout {r}] success={res['success']} r_max={res['r_max']} "
            f"event={res['fail_event']}@{res['tau_fail']} max_p={row['max_fpm']} "
            f"interventions={res['interventions']} saved_as={saved_as}"
            + (f"\n    first prediction: {res['predictions'][0]}" if res["predictions"] else "")
        )

    tag = "observe" if args.observe_only else "collect"
    with open(os.path.join(args.dataset_dir, f"{tag}_summary.json"), "w") as f:
        json.dump({"args": vars(args), "meta": meta, "rollouts": summary}, f, indent=2, default=str)
    np.savez(
        os.path.join(
            args.dataset_dir, "fpm_scores.npz" if args.observe_only else "fpm_scores_collect.npz"
        ),
        scores=np.stack(all_scores).astype(np.float32),  # (R, T), NaN = not evaluated
        success=np.array(all_success),
        tau_fail=np.array(all_tau),
    )

    n = len(summary)
    n_int = sum(bool(s["interventions"]) for s in summary)
    print(
        f"\n[collect] rollouts={n} success={sum(all_success)}/{n} intervened={n_int} "
        f"saved={n_saved} -> next index {idx} (train with --num_episodes {idx})"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--task_name", default="sim_transfer_cube_scripted")
    p.add_argument("--ckpt_dir", required=True, help="trained task-space learner (run_config.json)")
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument(
        "--dataset_dir", required=True, help="output dir (append to a COPY of the base dataset)"
    )
    p.add_argument("--num_rollouts", type=int, default=50)
    p.add_argument(
        "--start_episode_idx", type=int, default=None, help="default: #existing episodes"
    )
    p.add_argument("--seed", type=int, default=0)
    # learner execution
    p.add_argument("--temporal_agg", action="store_true")
    p.add_argument(
        "--query_every",
        type=int,
        default=None,
        help="re-plan stride (default: chunk size, or 1 with --temporal_agg)",
    )
    p.add_argument(
        "--n_inference_steps", type=int, default=None, help="override diffusion/flow sampling steps"
    )
    # FPM gate
    p.add_argument("--fpm", default="oracle", help="oracle | acc | pkg.module:factory")
    p.add_argument("--fpm_kwargs", default=None, help="JSON kwargs for the FPM")
    p.add_argument(
        "--fpm_threshold",
        type=float,
        default=0.5,
        help="trigger if p_hat > thr (use fpm.py calibrate)",
    )
    p.add_argument("--fpm_every", type=int, default=5, help="FPM check stride (steps)")
    p.add_argument("--fpm_warmup", type=int, default=0, help="no FPM checks before this step")
    p.add_argument(
        "--observe_only",
        action="store_true",
        help="never intervene; log scores/labels for calibration + FPM training",
    )
    p.add_argument(
        "--label_horizon",
        type=int,
        default=None,
        help="H for /fpm_label windows (default: chunk size)",
    )
    # intervention
    p.add_argument("--max_interventions", type=int, default=1)
    p.add_argument(
        "--handback_after",
        type=int,
        default=0,
        help="return control to the learner after N expert steps (0: expert finishes)",
    )
    # output
    p.add_argument(
        "--save", choices=["intervened_success", "intervened", "all"], default="intervened_success"
    )
    p.add_argument(
        "--save_video",
        action="store_true",
        help="mp4 per saved episode; expert frames get a red border",
    )
    p.add_argument(
        "--camera_height", type=int, default=None, help="default: env default (= what --eval uses)"
    )
    p.add_argument(
        "--camera_width", type=int, default=None, help="default: env default (= what --eval uses)"
    )
    main(p.parse_args())
