"""Merge episode folders into one training dataset (episode_0..N-1), by symlink.

    uv run python merge_datasets.py --out data/ts_preemptive --src data/ts runs/collect_fpm
    uv run python merge_datasets.py --out data/ts_reactive --src data/ts runs/collect_reactive:120

Each --src is a folder of episode_<i>.hdf5 files, optionally ``:N`` to take only the first
N (used to give every comparison arm the same number of extra episodes). Episodes are
symlinked (absolute paths) by default -- base datasets are ~100 GB -- or copied with
--copy. Refuses anything the training loader must not see or cannot stack:

- --observe_only rollouts (learner-only, no expert labels)
- episodes with /is_expert but no expert step
- episodes whose length, camera set or image size differ from the first source

Writes <out>/merge_manifest.json (source of every merged episode) and prints the
--num_episodes to train with.
"""

import argparse
import glob
import json
import os
import re
import shutil

import h5py


def _episodes(folder):
    eps = []
    for p in glob.glob(os.path.join(folder, "episode_*.hdf5")):
        m = re.fullmatch(r"episode_(\d+)\.hdf5", os.path.basename(p))
        if m:
            eps.append((int(m.group(1)), p))
    eps.sort()
    idx = [i for i, _ in eps]
    if idx != list(range(len(idx))):
        raise SystemExit(f"{folder}: episode indices are not contiguous from 0 ({idx[:5]}...)")
    return [p for _, p in eps]


def _signature(path):
    with h5py.File(path, "r") as r:
        if r.attrs.get("observe_only", False):
            raise SystemExit(f"{path}: --observe_only rollout (no expert labels); not mergeable")
        if "is_expert" in r and not r["is_expert"][()].any():
            raise SystemExit(f"{path}: /is_expert has no expert step; not mergeable")
        imgs = r["observations/images"]
        return {
            "T": int(r["action"].shape[0]),
            "action_dim": int(r["action"].shape[1]),
            "task_space": bool(r.attrs.get("task_space", False)),
            "cams": {k: tuple(imgs[k].shape[1:]) for k in sorted(imgs)},
            "preemptive": bool(r.attrs.get("preemptive_dagger", False)),
        }


def main(args):
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} exists and is not empty")
    plan, ref = [], None
    for spec in args.src:
        folder, _, n = spec.partition(":")
        eps = _episodes(folder)
        if n:
            if int(n) > len(eps):
                raise SystemExit(f"{folder}: asked for {n} episodes, only {len(eps)} exist")
            eps = eps[: int(n)]
        for p in eps:
            sig = _signature(p)
            core = {k: sig[k] for k in ("T", "action_dim", "task_space", "cams")}
            if ref is None:
                ref = core
            elif core != ref:
                raise SystemExit(f"{p}: incompatible with the first source\n  {core}\n  vs {ref}")
            plan.append((p, sig["preemptive"]))
        print(f"  {folder}: {len(eps)} episodes")

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for i, (src, pre) in enumerate(plan):
        dst = os.path.join(args.out, f"episode_{i}.hdf5")
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)
        manifest.append({"episode": i, "source": os.path.abspath(src), "preemptive_dagger": pre})
    with open(os.path.join(args.out, "merge_manifest.json"), "w") as f:
        json.dump({"sources": args.src, "episodes": manifest}, f, indent=1)
    n_extra = sum(m["preemptive_dagger"] for m in manifest)
    print(
        f"[merge] {len(plan)} episodes -> {args.out} ({n_extra} with expert masks); "
        f"train with --dataset_dir {args.out} --num_episodes {len(plan)}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True)
    p.add_argument("--src", nargs="+", required=True, help="episode folders, optionally folder:N")
    p.add_argument("--copy", action="store_true", help="copy instead of symlink")
    main(p.parse_args())
