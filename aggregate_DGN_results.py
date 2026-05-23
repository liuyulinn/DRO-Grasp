"""Aggregate per-scene validation markers written by validate_DGN.py.

Walks <results-dir>/<object_id>/<scene_kind>/<scale_pose>_result.json
(each marker is {"succ": int, "total": int, "object_id":..., "scene_kind":...,
"scale_pose": "scale006_pose000_0", ...}) and prints a DexGraspBench-style
summary. Markers with succ < 0 are counted toward "formatted" but not
"evaluated" (Isaac subprocess failures, missing scenes, etc.).

Default --results-dir is <out-dir>/<bodex-hand>/validation/, matching
validate_DGN.py's default output location.

Example:
  python aggregate_DGN_results.py --bodex-hand sim_xhand/fc_right
  python aggregate_DGN_results.py --results-dir /path/to/validation --write
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

_SCALE_RE = re.compile(r"scale(\d+)")
# Matches validate_DGN.py per-scene log lines like:
#   [xhand/core_bottle_xxx/tabletop_ur10e/scale006_pose000_0] 14/20
# 4 path components are required so per-object and TOTAL lines are ignored.
_LOG_RE = re.compile(
    r"^\[(?P<hand>[^/\]]+)/(?P<obj>[^/\]]+)/(?P<kind>[^/\]]+)/(?P<sp>[^/\]]+)\]\s+(?P<succ>-?\d+)/(?P<total>\d+)\b"
)


def parse_scale(scale_pose: str) -> str:
    """Extract 'NNN' from a scale_pose like 'scale006_pose000_0'.
    Returns '???' if the pattern doesn't match (kept in stats so nothing is dropped)."""
    m = _SCALE_RE.match(scale_pose)
    return m.group(1) if m else "???"


def load_log(log_path: str):
    """Parse a validate_DGN.py-style log into marker-shaped dicts.
    De-duplicates by (object_id, scene_kind, scale_pose), keeping the LAST line
    (so resumed runs that re-validated a scene override earlier entries)."""
    by_key = {}
    with open(log_path) as f:
        for line in f:
            m = _LOG_RE.match(line.strip())
            if not m:
                continue
            key = (m["obj"], m["kind"], m["sp"])
            by_key[key] = {
                "hand_name": m["hand"],
                "object_id": m["obj"],
                "scene_kind": m["kind"],
                "scale_pose": m["sp"],
                "succ": int(m["succ"]),
                "total": int(m["total"]),
            }
    return list(by_key.values())


def load_markers(results_dir: str):
    paths = sorted(glob.glob(os.path.join(results_dir, "*", "*", "*_result.json")))
    out = []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"[warn] skipping unreadable marker {p}: {e}", file=sys.stderr)
            continue
        d.setdefault("object_id", p.split(os.sep)[-3])
        d.setdefault("scene_kind", p.split(os.sep)[-2])
        d.setdefault("scale_pose", os.path.basename(p).replace("_result.json", ""))
        out.append(d)
    return out


def aggregate(markers):
    # Per-object: succ_total, eval_total, formatted_total
    per_obj = defaultdict(lambda: [0, 0, 0])  # [succ, evaluated, formatted]
    # Per-scale: succ, evaluated, formatted, set of objects touched
    per_scale = defaultdict(lambda: [0, 0, 0, set()])
    # (obj, scale) pairs seen / evaluated
    pairs_seen = set()
    pairs_evaluated = set()

    formatted = 0
    evaluated = 0
    successful = 0

    for d in markers:
        succ = int(d["succ"])
        total = int(d["total"])
        obj = d["object_id"]
        scale = parse_scale(d["scale_pose"])

        formatted += total
        per_obj[obj][2] += total
        per_scale[scale][2] += total
        per_scale[scale][3].add(obj)
        pairs_seen.add((obj, scale))

        if succ >= 0:
            evaluated += total
            successful += succ
            per_obj[obj][0] += succ
            per_obj[obj][1] += total
            per_scale[scale][0] += succ
            per_scale[scale][1] += total
            pairs_evaluated.add((obj, scale))

    objs_evaluated = {obj for obj, v in per_obj.items() if v[1] > 0}
    micro_sr = (successful / evaluated) if evaluated else float("nan")
    per_obj_sr = [
        per_obj[obj][0] / per_obj[obj][1]
        for obj in objs_evaluated
    ]
    macro_sr = (sum(per_obj_sr) / len(per_obj_sr)) if per_obj_sr else float("nan")

    scales_sorted = sorted(per_scale.keys())
    scale_rows = []
    for s in scales_sorted:
        succ, ev, fmt, objs = per_scale[s]
        # per-object avg within this scale
        obj_to_succ = defaultdict(lambda: [0, 0])
        for d in markers:
            if parse_scale(d["scale_pose"]) != s:
                continue
            if int(d["succ"]) < 0:
                continue
            obj_to_succ[d["object_id"]][0] += int(d["succ"])
            obj_to_succ[d["object_id"]][1] += int(d["total"])
        obj_avg = (
            sum(v[0] / v[1] for v in obj_to_succ.values() if v[1] > 0)
            / max(1, sum(1 for v in obj_to_succ.values() if v[1] > 0))
        ) if obj_to_succ else float("nan")
        micro = (succ / ev) if ev else float("nan")
        scale_rows.append({
            "scale": s,
            "succ": succ,
            "evaluated": ev,
            "formatted": fmt,
            "n_objects": len(objs),
            "obj_avg": obj_avg,
            "micro": micro,
        })

    return {
        "objects": len(per_obj),
        "objects_evaluated": len(objs_evaluated),
        "pairs_seen": len(pairs_seen),
        "pairs_evaluated": len(pairs_evaluated),
        "formatted_grasps": formatted,
        "evaluated_grasps": evaluated,
        "successful_grasps": successful,
        "eval_coverage": (evaluated / formatted) if formatted else float("nan"),
        "micro_sr": micro_sr,
        "macro_sr": macro_sr,
        "scales": scale_rows,
    }


def format_summary(summary, summary_path: str) -> str:
    lines = []
    lines.append(f"Wrote {summary_path}")
    lines.append(f"  objects:                  {summary['objects']} (evaluated: {summary['objects_evaluated']})")
    lines.append(f"  (obj,scale) pairs:        {summary['pairs_seen']} (evaluated: {summary['pairs_evaluated']})")
    lines.append(f"  formatted grasps:         {summary['formatted_grasps']}")
    lines.append(f"  evaluated grasps:         {summary['evaluated_grasps']}")
    lines.append(f"  successful grasps:        {summary['successful_grasps']}")
    lines.append(f"  eval coverage:            {summary['eval_coverage']:.4f}")
    lines.append(f"  micro SR (succ/eval):     {summary['micro_sr']:.4f}")
    lines.append(f"  macro SR (mean per obj):  {summary['macro_sr']:.4f}")
    for row in summary["scales"]:
        lines.append(
            f"  scale {row['scale']}: obj_avg={row['obj_avg']:.4f}  "
            f"micro={row['micro']:.4f}  "
            f"({row['succ']}/{row['evaluated']} over {row['n_objects']} objs)"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=None,
                   help="Parse a validate_DGN.py .log file instead of scanning "
                        "marker JSONs. Useful for runs done before per-scene "
                        "markers were introduced.")
    p.add_argument("--results-dir", default=None,
                   help="Directory of per-scene marker JSONs. Defaults to "
                        "<out-dir>/<bodex-hand>/validation/.")
    p.add_argument("--out-dir", default=os.path.join(ROOT_DIR, "dro_bodex_output"))
    p.add_argument("--bodex-hand", default="sim_xhand/fc_right")
    p.add_argument("--write", action="store_true",
                   help="Write eval_summary.json next to the input (results "
                        "dir or log file).")
    p.add_argument("--summary-path", default=None,
                   help="Override summary JSON path (implies --write).")
    args = p.parse_args()

    if args.log:
        if not os.path.isfile(args.log):
            raise SystemExit(f"log not found: {args.log}")
        markers = load_log(args.log)
        if not markers:
            raise SystemExit(f"no parsable scene lines in {args.log}")
        default_summary_dir = os.path.dirname(os.path.abspath(args.log))
    else:
        results_dir = args.results_dir or os.path.join(
            args.out_dir, args.bodex_hand, "validation"
        )
        if not os.path.isdir(results_dir):
            raise SystemExit(f"results dir not found: {results_dir}")
        markers = load_markers(results_dir)
        if not markers:
            raise SystemExit(f"no *_result.json under {results_dir}")
        default_summary_dir = results_dir

    summary = aggregate(markers)

    summary_path = args.summary_path or os.path.join(default_summary_dir, "eval_summary.json")
    if args.write or args.summary_path:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(format_summary(summary, summary_path if (args.write or args.summary_path) else "(not written; pass --write)"))


if __name__ == "__main__":
    main()
