"""Launch inference_DGN_bodex_seed.py across multiple GPUs.

Mimics BODex/example_grasp/multi_gpu.py: enumerates BODex object directories,
splits them into contiguous shards, and spawns one subprocess per GPU with
CUDA_VISIBLE_DEVICES set so each shard sees a single device.

Each child writes stdout/stderr to <out_dir>/runinfo/<shard_idx>_output.txt so
runs can be diagnosed after the fact.

Example:
  python multi_gpu_inference_DGN.py -g 0 1 2 3 \\
      --bodex-hand sim_xhand/fc_left \\
      --bodex-object-set DGN \\
      --bodex-scene-kind tabletop_ur10e \\
      --batch-size 20 --split-batch-size 20

  # one-object smoke test on a single GPU
  python multi_gpu_inference_DGN.py -g 0 \
      --bodex-glob 'core_bottle_1a7ba1f4c892e2da30711cdbdbc73924' \
      --batch-size 20 --split-batch-size 20
"""

import argparse
import datetime
import multiprocessing
import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from inference_DGN_bodex_seed import (
    BODEX_OUTPUT_ROOT,
    list_bodex_object_dirs,
)


def worker(gpu_id, shard_idx, obj_start, obj_end, args, output_path):
    """Run one shard on a single GPU. All Hydra knobs are passed via +key=value."""
    cmd = [
        "python", "inference_DGN_bodex_seed.py",
        f"+bodex_hand={args.bodex_hand}",
        f"+bodex_input_root={args.bodex_input_root}",
        f"+bodex_scene_kind={args.bodex_scene_kind}",
        f"+bodex_glob={args.bodex_glob}",
        f"+out_dir={args.out_dir}",
        f"+pregrasp_factor={args.pregrasp_factor}",
        f"+squeeze_factor={args.squeeze_factor}",
        f"+obj_start={obj_start}",
        f"+obj_end={obj_end}",
        f"dataset.batch_size={args.batch_size}",
        f"split_batch_size={args.split_batch_size}",
        # Pin Hydra+DRO to GPU 0 inside the subprocess; CUDA_VISIBLE_DEVICES
        # already isolates the physical device, so cfg.gpu=0 maps to gpu_id.
        "gpu=0",
    ]
    if args.bodex_object_set:
        cmd.append(f"+bodex_object_set={args.bodex_object_set}")
    if args.hand_name:
        cmd.append(f"+hand_name={args.hand_name}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(
        f"[shard {shard_idx}] gpu={gpu_id}  objs=[{obj_start},{obj_end})  log={output_path}",
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
    p.add_argument("--bodex-hand", default="sim_xhand/fc_left")
    p.add_argument("--bodex-input-root", default=BODEX_OUTPUT_ROOT)
    p.add_argument("--bodex-object-set", default="DGN")
    p.add_argument("--bodex-scene-kind", default="tabletop_ur10e")
    p.add_argument("--bodex-glob", default="*", help="object_id directory glob")
    p.add_argument("--hand-name", default=None, help="DRO hand_name (auto-inferred if omitted)")
    p.add_argument("--out-dir", default=os.path.join(ROOT_DIR, "dro_bodex_output"))
    p.add_argument("--batch-size", type=int, default=20,
                   help="dataset.batch_size; should equal n_seeds for 1-to-1 BODex correspondence")
    p.add_argument("--split-batch-size", type=int, default=20)
    p.add_argument("--pregrasp-factor", type=float, default=0.9)
    p.add_argument("--squeeze-factor", type=float, default=1.05)
    p.add_argument(
        "--obj-start", type=int, default=None,
        help="Index into the sorted object list to start at (inclusive). "
             "Combined with --obj-end for small-batch testing.",
    )
    p.add_argument(
        "--obj-end", type=int, default=None,
        help="Index into the sorted object list to stop at (exclusive).",
    )
    args = p.parse_args()

    hand_root = os.path.join(args.bodex_input_root, args.bodex_hand)
    if not os.path.isdir(hand_root):
        raise SystemExit(f"bodex hand path not found: {hand_root}")

    obj_dirs = list_bodex_object_dirs(hand_root, args.bodex_object_set, args.bodex_glob)
    n_objs_total = len(obj_dirs)
    if n_objs_total == 0:
        raise SystemExit(
            f"no objects matched under {hand_root} (set='{args.bodex_object_set}', glob='{args.bodex_glob}')"
        )

    # Restrict to the user-requested [obj_start, obj_end) window before sharding,
    # so --obj-start/--obj-end describe a global range and shards split it evenly.
    window_start = 0 if args.obj_start is None else args.obj_start
    window_end = n_objs_total if args.obj_end is None else args.obj_end
    if window_start < 0:
        window_start = max(n_objs_total + window_start, 0)
    if window_end < 0:
        window_end = max(n_objs_total + window_end, 0)
    window_start = max(0, min(window_start, n_objs_total))
    window_end = max(window_start, min(window_end, n_objs_total))
    n_objs = window_end - window_start
    if n_objs == 0:
        raise SystemExit(
            f"empty object window [{window_start}, {window_end}) of {n_objs_total} matched objects"
        )

    n_gpus = len(args.gpu)

    # Even split within the window, first (n_objs % n_gpus) shards get one extra.
    base, rem = divmod(n_objs, n_gpus)
    counts = [base + (1 if i < rem else 0) for i in range(n_gpus)]
    starts = [window_start + sum(counts[:i]) for i in range(n_gpus)]
    ends = [starts[i] + counts[i] for i in range(n_gpus)]

    runinfo_dir = os.path.join(
        args.out_dir, "runinfo",
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    )
    os.makedirs(runinfo_dir, exist_ok=True)
    if window_start != 0 or window_end != n_objs_total:
        window_msg = f" (window [{window_start},{window_end}) of {n_objs_total})"
    else:
        window_msg = ""
    print(f"[main] {n_objs} objects across {n_gpus} GPUs{window_msg}; runinfo -> {runinfo_dir}")

    procs = []
    for i, gpu_id in enumerate(args.gpu):
        if counts[i] == 0:
            print(f"[shard {i}] gpu={gpu_id} no objects assigned, skipping")
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
