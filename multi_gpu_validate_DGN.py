"""Launch validate_DGN.py across multiple GPUs.

Mirrors multi_gpu_inference_DGN.py: enumerates the flat scene list produced by
validate_DGN.find_grasp_npys, splits it into contiguous shards, and spawns one
subprocess per GPU with CUDA_VISIBLE_DEVICES set so each shard sees a single
device. validate_DGN.py's --overwrite / per-scene marker JSONs make repeated
runs idempotent: shards skip scenes that already have a result marker.

Each child writes stdout/stderr to <out_dir>/<bodex_hand>/runinfo/validate_<ts>/
<shard_idx>_gpu<gid>.log.

Example:
  python multi_gpu_validate_DGN.py -g 0 1 2 3 \\
      --bodex-hand sim_xhand/fc_right \\
      --dgn-root /home/yulin/workspace/DexCom/assets/misc/DGN_2k_origin

  # smoke test on one GPU
  python multi_gpu_validate_DGN.py -g 0 \\
      --object-glob 'core_bottle_1a7ba1f4*' \\
      --bodex-hand sim_xhand/fc_right
"""

import argparse
import datetime
import multiprocessing
import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from validate_DGN import find_grasp_npys


def worker(gpu_id, shard_idx, scene_start, scene_end, args, output_path):
    """Run one shard on a single GPU."""
    log_name = f"{args.log_name_base}_shard{shard_idx}"
    cmd = [
        "python", "validate_DGN.py",
        "--out-dir", args.out_dir,
        "--bodex-hand", args.bodex_hand,
        "--scene-kind", args.scene_kind,
        "--object-glob", args.object_glob,
        "--obj-start", str(scene_start),
        "--obj-end", str(scene_end),
        "--log-name", log_name,
        "--gpu", "0",  # CUDA_VISIBLE_DEVICES already pins the physical device
    ]
    if args.hand_name:
        cmd.extend(["--hand-name", args.hand_name])
    if args.dgn_root:
        cmd.extend(["--dgn-root", args.dgn_root])
    if args.results_dir:
        cmd.extend(["--results-dir", args.results_dir])
    if args.overwrite:
        cmd.append("--overwrite")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(
        f"[shard {shard_idx}] gpu={gpu_id}  scenes=[{scene_start},{scene_end})  log={output_path}",
        flush=True,
    )
    with open(output_path, "w") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n# CUDA_VISIBLE_DEVICES={gpu_id}\n\n")
        f.flush()
        rc = subprocess.call(cmd, cwd=ROOT_DIR, env=env, stdout=f, stderr=f)
    print(f"[shard {shard_idx}] gpu={gpu_id} done (rc={rc})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-g", "--gpu", nargs="+", required=True, help="gpu id list, e.g. -g 0 1 2 3")
    p.add_argument("--out-dir", default=os.path.join(ROOT_DIR, "dro_bodex_output"))
    p.add_argument("--bodex-hand", default="sim_xhand/fc_right")
    p.add_argument("--scene-kind", default="*")
    p.add_argument("--object-glob", default="*")
    p.add_argument("--hand-name", default=None)
    p.add_argument("--dgn-root", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-validate scenes whose result marker already exists.")
    p.add_argument("--obj-start", type=int, default=None,
                   help="Inclusive start index into the flat scene list.")
    p.add_argument("--obj-end", type=int, default=None,
                   help="Exclusive end index into the flat scene list.")
    p.add_argument("--log-name-base", default=None,
                   help="Per-shard log name prefix (default: validate_DGN_<bodex_hand>).")
    args = p.parse_args()

    graspdata_root = os.path.join(args.out_dir, args.bodex_hand, "graspdata")
    if not os.path.isdir(graspdata_root):
        raise SystemExit(f"graspdata root not found: {graspdata_root}")

    entries = find_grasp_npys(graspdata_root, args.object_glob, args.scene_kind)
    n_total = len(entries)
    if n_total == 0:
        raise SystemExit(
            f"no *_grasp.npy under {graspdata_root} "
            f"(object_glob={args.object_glob!r}, scene_kind={args.scene_kind!r})"
        )

    window_start = 0 if args.obj_start is None else args.obj_start
    window_end = n_total if args.obj_end is None else args.obj_end
    if window_start < 0:
        window_start = max(n_total + window_start, 0)
    if window_end < 0:
        window_end = max(n_total + window_end, 0)
    window_start = max(0, min(window_start, n_total))
    window_end = max(window_start, min(window_end, n_total))
    n_scenes = window_end - window_start
    if n_scenes == 0:
        raise SystemExit(
            f"empty scene window [{window_start}, {window_end}) of {n_total}"
        )

    n_gpus = len(args.gpu)
    base, rem = divmod(n_scenes, n_gpus)
    counts = [base + (1 if i < rem else 0) for i in range(n_gpus)]
    starts = [window_start + sum(counts[:i]) for i in range(n_gpus)]
    ends = [starts[i] + counts[i] for i in range(n_gpus)]

    if args.log_name_base is None:
        args.log_name_base = f"validate_DGN_{args.bodex_hand.replace('/', '_')}"

    runinfo_dir = os.path.join(
        args.out_dir, args.bodex_hand, "runinfo",
        "validate_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    )
    os.makedirs(runinfo_dir, exist_ok=True)

    window_msg = (
        f" (window [{window_start},{window_end}) of {n_total})"
        if (window_start, window_end) != (0, n_total) else ""
    )
    print(f"[main] {n_scenes} scenes across {n_gpus} GPUs{window_msg}; runinfo -> {runinfo_dir}")

    procs = []
    for i, gpu_id in enumerate(args.gpu):
        if counts[i] == 0:
            print(f"[shard {i}] gpu={gpu_id} no scenes assigned, skipping")
            continue
        log_path = os.path.join(runinfo_dir, f"{i}_gpu{gpu_id}.log")
        proc = multiprocessing.Process(
            target=worker,
            args=(gpu_id, i, starts[i], ends[i], args, log_path),
        )
        proc.start()
        procs.append(proc)
        print(f"[main] spawned pid={proc.pid} for shard {i} (gpu {gpu_id})")

    for proc in procs:
        proc.join()

    print(f"[main] all shards finished. logs at {runinfo_dir}")


if __name__ == "__main__":
    main()
