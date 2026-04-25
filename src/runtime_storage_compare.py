## Draft 2 - 4/22/26
#!/usr/bin/env python3
"""
runtime_storage_compare.py

Purpose
-------
Build one runtime/storage comparison table for baseline vs adaptive runs.

This is meant to be an engineering-efficiency supplement, not the main
paper-metric comparison. It compares:
- runtime (optional rerun of train/precompute/inject)
- peak memory (best-effort, process tree)
- artifact sizes
- subgroup counts
- optional active-group stats for adaptive runs

Examples
--------
# Use existing artifacts only (storage/counts only; runtime columns stay empty)
python -m src.runtime_storage_compare ^
  --baseline-checkpoint adult_model_10 ^
  --adaptive-checkpoint adult_model_adaptive_10 ^
  --output-dir report-metrics/runtime_compare_nt10

# Rerun inject only to measure runtime/memory
python -m src.runtime_storage_compare ^
  --baseline-checkpoint adult_model_10 ^
  --adaptive-checkpoint adult_model_adaptive_10 ^
  --rerun-stage inject ^
  --n-targets 10 ^
  --noise 0.50 ^
  --output-dir report-metrics/runtime_compare_nt10

# Full rerun (train + precompute + inject)
python -m src.runtime_storage_compare ^
  --baseline-checkpoint adult_model_10 ^
  --adaptive-checkpoint adult_model_adaptive_10 ^
  --rerun-stage all ^
  --n-targets 50 ^
  --noise 0.50 ^
  --output-dir report-metrics/runtime_compare_nt50
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import psutil
except Exception:
    psutil = None


def safe_pickle_load(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def file_size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size if path.exists() else 0
    except Exception:
        return 0


def sum_file_sizes(paths: Iterable[Path]) -> int:
    return int(sum(file_size_bytes(p) for p in paths))


def format_mb(n_bytes: Optional[int]) -> Optional[float]:
    if n_bytes is None:
        return None
    return float(n_bytes) / (1024.0 * 1024.0)


def list_matching_files(root: Path, pattern: str) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob(pattern) if p.is_file()])


def parse_support_bucket(name: str) -> Optional[Tuple[float, float]]:
    m = re.search(r"support-([0-9.]+)-([0-9.]+)-", name)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def summarize_support_buckets(files: List[Path]) -> Tuple[int, Dict[str, int]]:
    counts: Dict[str, int] = {}
    for f in files:
        bucket = parse_support_bucket(f.name)
        if bucket is None:
            continue
        key = f"{bucket[0]:.4f}-{bucket[1]:.4f}"
        counts[key] = counts.get(key, 0) + 1
    return len(counts), counts


def estimate_subgroup_count(models_ckpt: Path, checkpoint: str) -> Optional[int]:
    obj = safe_pickle_load(models_ckpt / f"matches-{checkpoint}.pkl")
    if obj is None:
        return None
    try:
        return int(len(obj["matches_train"].fi))
    except Exception:
        return None


def estimate_active_group_stats(models_ckpt: Path, checkpoint: str) -> Optional[Dict[str, float]]:
    """
    Looks for 'active_group_counts' inside saved sup-wise payloads.
    This is usually meaningful for adaptive runs only.
    """
    supwise_dir = models_ckpt / "sup-wise"
    files = list_matching_files(supwise_dir, f"{checkpoint}-*.pkl")

    counts: List[int] = []
    for p in files:
        obj = safe_pickle_load(p)
        if isinstance(obj, dict) and "active_group_counts" in obj:
            try:
                counts.extend(int(x) for x in obj["active_group_counts"])
            except Exception:
                pass

    if not counts:
        return None

    arr = np.asarray(counts, dtype=float)
    return {
        "avg_active_groups": float(np.mean(arr)),
        "min_active_groups": int(np.min(arr)),
        "max_active_groups": int(np.max(arr)),
    }


def collect_artifact_stats(models_ckpt: Path, checkpoint: str) -> Dict[str, object]:
    model_path = models_ckpt / f"{checkpoint}.pkl"
    dataset_path = models_ckpt / f"{checkpoint}.dataset.pkl"
    matches_path = models_ckpt / f"matches-{checkpoint}.pkl"

    supwise_files = list_matching_files(models_ckpt / "sup-wise", f"{checkpoint}-*.pkl")
    drift_dirs = [p for p in models_ckpt.glob(f"{checkpoint}-accuracy-noise-*") if p.is_dir()]
    drift_files: List[Path] = []
    for d in drift_dirs:
        drift_files.extend(list_matching_files(d, "*.pkl"))

    model_bytes = file_size_bytes(model_path)
    dataset_bytes = file_size_bytes(dataset_path)
    matches_bytes = file_size_bytes(matches_path)
    supwise_bytes = sum_file_sizes(supwise_files)
    drift_eval_bytes = sum_file_sizes(drift_files)

    support_bucket_count, support_bucket_summary = summarize_support_buckets(supwise_files)

    return {
        "model_bytes": model_bytes,
        "dataset_bytes": dataset_bytes,
        "matches_bytes": matches_bytes,
        "supwise_bytes": supwise_bytes,
        "drift_eval_bytes": drift_eval_bytes,
        "total_artifact_bytes": model_bytes + dataset_bytes + matches_bytes + supwise_bytes + drift_eval_bytes,
        "model_mb": format_mb(model_bytes),
        "dataset_mb": format_mb(dataset_bytes),
        "matches_mb": format_mb(matches_bytes),
        "supwise_mb": format_mb(supwise_bytes),
        "drift_eval_mb": format_mb(drift_eval_bytes),
        "total_artifact_mb": format_mb(model_bytes + dataset_bytes + matches_bytes + supwise_bytes + drift_eval_bytes),
        "supwise_file_count": len(supwise_files),
        "drift_eval_file_count": len(drift_files),
        "support_bucket_count": support_bucket_count,
        "support_bucket_summary": json.dumps(support_bucket_summary, sort_keys=True),
    }


def process_tree_rss_bytes(proc: "psutil.Process") -> int:
    """
    Best-effort peak RSS across parent + children.
    """
    total = 0
    try:
        procs = [proc] + proc.children(recursive=True)
    except Exception:
        procs = [proc]

    for p in procs:
        try:
            total += p.memory_info().rss
        except Exception:
            pass
    return total


def run_stage(cmd: List[str], cwd: Path) -> Tuple[int, float, Optional[float]]:
    """
    Returns:
    - return code
    - runtime seconds
    - peak memory MB (process tree, if psutil is available)
    """
    start = time.perf_counter()
    peak_rss = 0

    proc = subprocess.Popen(cmd, cwd=str(cwd))
    ps_proc = psutil.Process(proc.pid) if psutil is not None else None

    while True:
        ret = proc.poll()

        if ps_proc is not None:
            try:
                peak_rss = max(peak_rss, process_tree_rss_bytes(ps_proc))
            except Exception:
                pass

        if ret is not None:
            break

        time.sleep(0.2)

    runtime_sec = time.perf_counter() - start
    peak_mb = (peak_rss / (1024.0 * 1024.0)) if peak_rss > 0 else None
    return proc.returncode, runtime_sec, peak_mb


def build_stage_commands(
    python_exe: str,
    implementation: str,
    checkpoint: str,
    minsup: float,
    n_proc: int,
    noise: float,
    n_targets: int,
    light_output: bool,
    adaptive_top_k: int,
    adaptive_interval: int,
    adaptive_score: str,
    adaptive_min_groups: int,
    win_size: int,
    adaptive_mode: str,
    stable_score_threshold: float,
    stable_rounds: int,
    refresh_interval: int,
    refresh_top_k: Optional[int],
    refresh_threshold: Optional[float],
) -> Dict[str, List[str]]:
    train_cmd = [
        python_exe,
        "-m",
        "src.adult.models",
        "--checkpoint",
        checkpoint,
    ]

    precompute_cmd = [
        python_exe,
        "-m",
        "src.adult.precompute",
        "--checkpoint",
        checkpoint,
        "--minsup",
        str(minsup),
        "--n-proc",
        str(n_proc),
    ]

    if implementation == "baseline":
        inject_cmd = [
            python_exe,
            "-m",
            "src.adult.inject",
            "--checkpoint",
            checkpoint,
            "--frac-noise",
            str(noise),
            "--n-targets",
            str(n_targets),
            "--n-proc",
            str(n_proc),
            "--resume",
        ]
    else:
        inject_cmd = [
            python_exe,
            "-m",
            "src.adult.inject_adaptive",
            "--checkpoint",
            checkpoint,
            "--frac-noise",
            str(noise),
            "--n-targets",
            str(n_targets),
            "--adaptive-enable",
            "--adaptive-mode",
            str(adaptive_mode),
            "--adaptive-interval",
            str(adaptive_interval),
            "--adaptive-score",
            str(adaptive_score),
            "--adaptive-min-groups",
            str(adaptive_min_groups),
            "--win-size",
            str(win_size),
            "--stable-score-threshold",
            str(stable_score_threshold),
            "--stable-rounds",
            str(stable_rounds),
            "--refresh-interval",
            str(refresh_interval),
            "--resume",
        ]
        if adaptive_top_k is not None:
            inject_cmd.extend(["--adaptive-top-k", str(adaptive_top_k)])
        if refresh_top_k is not None:
            inject_cmd.extend(["--refresh-top-k", str(refresh_top_k)])
        if refresh_threshold is not None:
            inject_cmd.extend(["--refresh-threshold", str(refresh_threshold)])

    if light_output:
        inject_cmd.append("--light-output")

    return {
        "train": train_cmd,
        "precompute": precompute_cmd,
        "inject": inject_cmd,
    }


def safe_ratio(num, den) -> Optional[float]:
    if num is None or den is None:
        return None
    if pd.isna(num) or pd.isna(den) or den == 0:
        return None
    return float(num) / float(den)


def main():
    parser = argparse.ArgumentParser(description="Build one runtime/storage comparison table.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--models-ckpt", type=Path, default=Path("models-ckpt"))
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--adaptive-checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("report-metrics/runtime_storage_compare"))

    parser.add_argument(
        "--rerun-stage",
        choices=["none", "inject", "all"],
        default="none",
        help="none = only inspect existing artifacts; inject = rerun inject only; all = rerun train/precompute/inject",
    )

    parser.add_argument("--minsup", type=float, default=0.05)
    parser.add_argument("--n-proc", type=int, default=2)
    parser.add_argument("--n-targets", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.50)
    parser.add_argument("--win-size", type=int, default=5)
    parser.add_argument("--light-output", action="store_true")

    parser.add_argument("--adaptive-top-k", type=int, default=500)
    parser.add_argument("--adaptive-interval", type=int, default=5)
    parser.add_argument("--adaptive-score", type=str, default="abs_tstat")
    parser.add_argument("--adaptive-min-groups", type=int, default=100)
    parser.add_argument("--adaptive-mode", choices=["active_only", "stability", "refresh"], default="active_only")
    parser.add_argument("--stable-score-threshold", type=float, default=0.0)
    parser.add_argument("--stable-rounds", type=int, default=2)
    parser.add_argument("--refresh-interval", type=int, default=3)
    parser.add_argument("--refresh-top-k", type=int, default=None)
    parser.add_argument("--refresh-threshold", type=float, default=None)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    impls = [
        ("baseline", args.baseline_checkpoint),
        ("adaptive", args.adaptive_checkpoint),
    ]

    for implementation, checkpoint in impls:
        stage_runtime = {
            "train_runtime_sec": None,
            "precompute_runtime_sec": None,
            "inject_runtime_sec": None,
            "total_runtime_sec": None,
        }
        stage_peak = {
            "train_peak_mb": None,
            "precompute_peak_mb": None,
            "inject_peak_mb": None,
            "peak_memory_mb": None,
        }
        stage_rc = {
            "train_returncode": None,
            "precompute_returncode": None,
            "inject_returncode": None,
        }

        commands = build_stage_commands(
            python_exe=args.python_exe,
            implementation=implementation,
            checkpoint=checkpoint,
            minsup=args.minsup,
            n_proc=args.n_proc,
            noise=args.noise,
            n_targets=args.n_targets,
            light_output=args.light_output,
            adaptive_top_k=args.adaptive_top_k,
            adaptive_interval=args.adaptive_interval,
            adaptive_score=args.adaptive_score,
            adaptive_min_groups=args.adaptive_min_groups,
            win_size=args.win_size,
            adaptive_mode=args.adaptive_mode,
            stable_score_threshold=args.stable_score_threshold,
            stable_rounds=args.stable_rounds,
            refresh_interval=args.refresh_interval,
            refresh_top_k=args.refresh_top_k,
            refresh_threshold=args.refresh_threshold,
        )


        stages_to_run: List[str] = []
        if args.rerun_stage == "inject":
            stages_to_run = ["inject"]
        elif args.rerun_stage == "all":
            stages_to_run = ["train", "precompute", "inject"]

        for stage in stages_to_run:
            cmd = commands[stage]
            rc, runtime_sec, peak_mb = run_stage(cmd, args.project_root)

            stage_runtime[f"{stage}_runtime_sec"] = runtime_sec
            stage_peak[f"{stage}_peak_mb"] = peak_mb
            stage_rc[f"{stage}_returncode"] = rc

            if rc != 0:
                print(f"[WARN] {implementation}:{stage} failed with return code {rc}")

        runtime_vals = [v for v in stage_runtime.values() if isinstance(v, (int, float))]
        peak_vals = [v for v in stage_peak.values() if isinstance(v, (int, float))]

        if runtime_vals:
            stage_runtime["total_runtime_sec"] = float(np.sum(runtime_vals))
        if peak_vals:
            stage_peak["peak_memory_mb"] = float(np.max(peak_vals))

        subgroup_count = estimate_subgroup_count(args.models_ckpt, checkpoint)
        active_stats = estimate_active_group_stats(args.models_ckpt, checkpoint)
        effective_monitored = subgroup_count
        if active_stats is not None:
            effective_monitored = active_stats["avg_active_groups"]

        artifact_stats = collect_artifact_stats(args.models_ckpt, checkpoint)

        row: Dict[str, object] = {
            "implementation": implementation,
            "checkpoint": checkpoint,
            "rerun_stage": args.rerun_stage,
            "n_targets": args.n_targets,
            "noise": args.noise,
            "minsup": args.minsup,
            "n_proc": args.n_proc,
            "win_size": args.win_size,
            "light_output": args.light_output,
            "full_subgroup_count": subgroup_count,
            "effective_monitored_subgroups": effective_monitored,
            **stage_runtime,
            **stage_peak,
            **stage_rc,
            **artifact_stats,
        }

        if active_stats is not None:
            row.update(active_stats)

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) == 2:
        base = df[df["implementation"] == "baseline"].iloc[0]
        adapt = df[df["implementation"] == "adaptive"].iloc[0]

        subgroup_ratio = safe_ratio(
            adapt.get("effective_monitored_subgroups"),
            base.get("effective_monitored_subgroups"),
        )
        runtime_ratio = safe_ratio(
            adapt.get("total_runtime_sec"),
            base.get("total_runtime_sec"),
        )
        storage_ratio = safe_ratio(
            adapt.get("total_artifact_bytes"),
            base.get("total_artifact_bytes"),
        )

        df.loc[df["implementation"] == "adaptive", "subgroup_ratio_vs_baseline"] = subgroup_ratio
        df.loc[df["implementation"] == "adaptive", "runtime_ratio_vs_baseline"] = runtime_ratio
        df.loc[df["implementation"] == "adaptive", "storage_ratio_vs_baseline"] = storage_ratio

        df.loc[df["implementation"] == "adaptive", "subgroup_reduction_vs_baseline"] = (
            1.0 - subgroup_ratio if subgroup_ratio is not None else None
        )
        df.loc[df["implementation"] == "adaptive", "runtime_reduction_vs_baseline"] = (
            1.0 - runtime_ratio if runtime_ratio is not None else None
        )
        df.loc[df["implementation"] == "adaptive", "storage_reduction_vs_baseline"] = (
            1.0 - storage_ratio if storage_ratio is not None else None
        )

    out_csv = args.output_dir / "runtime_storage_compare.csv"
    out_md = args.output_dir / "runtime_storage_compare.md"

    df.to_csv(out_csv, index=False)

    # Small pretty markdown table for report use
    md_cols = [
        "implementation",
        "checkpoint",
        "total_runtime_sec",
        "peak_memory_mb",
        "full_subgroup_count",
        "effective_monitored_subgroups",
        "total_artifact_mb",
        "supwise_file_count",
        "support_bucket_count",
    ]
    md_df = df[[c for c in md_cols if c in df.columns]].copy()
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_df.to_markdown(index=False))

    print("\n=== Runtime / Storage Compare ===")
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved MD : {out_md}")


if __name__ == "__main__":
    main()

### Draft 1
# #!/usr/bin/env python3
# """
# runtime_storage_compare.py
#
# Purpose
# -------
# Build one runtime/storage comparison table for baseline vs adaptive runs.
#
# This is meant to be an engineering-efficiency supplement, not the main
# paper-metric comparison. It compares:
# - runtime (optional rerun of train/precompute/inject)
# - peak memory (best-effort, process tree)
# - artifact sizes
# - subgroup counts
# - optional active-group stats for adaptive runs
#
# Examples
# --------
# # Use existing artifacts only (storage/counts only; runtime columns stay empty)
# python -m src.runtime_storage_compare ^
#   --baseline-checkpoint adult_model_10 ^
#   --adaptive-checkpoint adult_model_adaptive_10 ^
#   --output-dir report-metrics/runtime_compare_nt10
#
# # Rerun inject only to measure runtime/memory
# python -m src.runtime_storage_compare ^
#   --baseline-checkpoint adult_model_10 ^
#   --adaptive-checkpoint adult_model_adaptive_10 ^
#   --rerun-stage inject ^
#   --n-targets 10 ^
#   --noise 0.50 ^
#   --output-dir report-metrics/runtime_compare_nt10
#
# # Full rerun (train + precompute + inject)
# python -m src.runtime_storage_compare ^
#   --baseline-checkpoint adult_model_10 ^
#   --adaptive-checkpoint adult_model_adaptive_10 ^
#   --rerun-stage all ^
#   --n-targets 50 ^
#   --noise 0.50 ^
#   --output-dir report-metrics/runtime_compare_nt50
# """
#
# from __future__ import annotations
#
# import argparse
# import json
# import pickle
# import re
# import subprocess
# import sys
# import time
# from pathlib import Path
# from typing import Dict, Iterable, List, Optional, Tuple
#
# import numpy as np
# import pandas as pd
#
# try:
#     import psutil
# except Exception:
#     psutil = None
#
#
# def safe_pickle_load(path: Path):
#     try:
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     except Exception:
#         return None
#
#
# def file_size_bytes(path: Path) -> int:
#     try:
#         return path.stat().st_size if path.exists() else 0
#     except Exception:
#         return 0
#
#
# def sum_file_sizes(paths: Iterable[Path]) -> int:
#     return int(sum(file_size_bytes(p) for p in paths))
#
#
# def format_mb(n_bytes: Optional[int]) -> Optional[float]:
#     if n_bytes is None:
#         return None
#     return float(n_bytes) / (1024.0 * 1024.0)
#
#
# def list_matching_files(root: Path, pattern: str) -> List[Path]:
#     if not root.exists():
#         return []
#     return sorted([p for p in root.rglob(pattern) if p.is_file()])
#
#
# def parse_support_bucket(name: str) -> Optional[Tuple[float, float]]:
#     m = re.search(r"support-([0-9.]+)-([0-9.]+)-", name)
#     if not m:
#         return None
#     return float(m.group(1)), float(m.group(2))
#
#
# def summarize_support_buckets(files: List[Path]) -> Tuple[int, Dict[str, int]]:
#     counts: Dict[str, int] = {}
#     for f in files:
#         bucket = parse_support_bucket(f.name)
#         if bucket is None:
#             continue
#         key = f"{bucket[0]:.4f}-{bucket[1]:.4f}"
#         counts[key] = counts.get(key, 0) + 1
#     return len(counts), counts
#
#
# def estimate_subgroup_count(models_ckpt: Path, checkpoint: str) -> Optional[int]:
#     obj = safe_pickle_load(models_ckpt / f"matches-{checkpoint}.pkl")
#     if obj is None:
#         return None
#     try:
#         return int(len(obj["matches_train"].fi))
#     except Exception:
#         return None
#
#
# def estimate_active_group_stats(models_ckpt: Path, checkpoint: str) -> Optional[Dict[str, float]]:
#     """
#     Looks for 'active_group_counts' inside saved sup-wise payloads.
#     This is usually meaningful for adaptive runs only.
#     """
#     supwise_dir = models_ckpt / "sup-wise"
#     files = list_matching_files(supwise_dir, f"{checkpoint}-*.pkl")
#
#     counts: List[int] = []
#     for p in files:
#         obj = safe_pickle_load(p)
#         if isinstance(obj, dict) and "active_group_counts" in obj:
#             try:
#                 counts.extend(int(x) for x in obj["active_group_counts"])
#             except Exception:
#                 pass
#
#     if not counts:
#         return None
#
#     arr = np.asarray(counts, dtype=float)
#     return {
#         "avg_active_groups": float(np.mean(arr)),
#         "min_active_groups": int(np.min(arr)),
#         "max_active_groups": int(np.max(arr)),
#     }
#
#
# def collect_artifact_stats(models_ckpt: Path, checkpoint: str) -> Dict[str, object]:
#     model_path = models_ckpt / f"{checkpoint}.pkl"
#     dataset_path = models_ckpt / f"{checkpoint}.dataset.pkl"
#     matches_path = models_ckpt / f"matches-{checkpoint}.pkl"
#
#     supwise_files = list_matching_files(models_ckpt / "sup-wise", f"{checkpoint}-*.pkl")
#     drift_dirs = [p for p in models_ckpt.glob(f"{checkpoint}-accuracy-noise-*") if p.is_dir()]
#     drift_files: List[Path] = []
#     for d in drift_dirs:
#         drift_files.extend(list_matching_files(d, "*.pkl"))
#
#     model_bytes = file_size_bytes(model_path)
#     dataset_bytes = file_size_bytes(dataset_path)
#     matches_bytes = file_size_bytes(matches_path)
#     supwise_bytes = sum_file_sizes(supwise_files)
#     drift_eval_bytes = sum_file_sizes(drift_files)
#
#     support_bucket_count, support_bucket_summary = summarize_support_buckets(supwise_files)
#
#     return {
#         "model_bytes": model_bytes,
#         "dataset_bytes": dataset_bytes,
#         "matches_bytes": matches_bytes,
#         "supwise_bytes": supwise_bytes,
#         "drift_eval_bytes": drift_eval_bytes,
#         "total_artifact_bytes": model_bytes + dataset_bytes + matches_bytes + supwise_bytes + drift_eval_bytes,
#         "model_mb": format_mb(model_bytes),
#         "dataset_mb": format_mb(dataset_bytes),
#         "matches_mb": format_mb(matches_bytes),
#         "supwise_mb": format_mb(supwise_bytes),
#         "drift_eval_mb": format_mb(drift_eval_bytes),
#         "total_artifact_mb": format_mb(model_bytes + dataset_bytes + matches_bytes + supwise_bytes + drift_eval_bytes),
#         "supwise_file_count": len(supwise_files),
#         "drift_eval_file_count": len(drift_files),
#         "support_bucket_count": support_bucket_count,
#         "support_bucket_summary": json.dumps(support_bucket_summary, sort_keys=True),
#     }
#
#
# def process_tree_rss_bytes(proc: "psutil.Process") -> int:
#     """
#     Best-effort peak RSS across parent + children.
#     """
#     total = 0
#     try:
#         procs = [proc] + proc.children(recursive=True)
#     except Exception:
#         procs = [proc]
#
#     for p in procs:
#         try:
#             total += p.memory_info().rss
#         except Exception:
#             pass
#     return total
#
#
# def run_stage(cmd: List[str], cwd: Path) -> Tuple[int, float, Optional[float]]:
#     """
#     Returns:
#     - return code
#     - runtime seconds
#     - peak memory MB (process tree, if psutil is available)
#     """
#     start = time.perf_counter()
#     peak_rss = 0
#
#     proc = subprocess.Popen(cmd, cwd=str(cwd))
#     ps_proc = psutil.Process(proc.pid) if psutil is not None else None
#
#     while True:
#         ret = proc.poll()
#
#         if ps_proc is not None:
#             try:
#                 peak_rss = max(peak_rss, process_tree_rss_bytes(ps_proc))
#             except Exception:
#                 pass
#
#         if ret is not None:
#             break
#
#         time.sleep(0.2)
#
#     runtime_sec = time.perf_counter() - start
#     peak_mb = (peak_rss / (1024.0 * 1024.0)) if peak_rss > 0 else None
#     return proc.returncode, runtime_sec, peak_mb
#
#
# def build_stage_commands(
#     python_exe: str,
#     implementation: str,
#     checkpoint: str,
#     minsup: float,
#     n_proc: int,
#     noise: float,
#     n_targets: int,
#     light_output: bool,
#     adaptive_top_k: int,
#     adaptive_interval: int,
#     adaptive_score: str,
#     adaptive_min_groups: int,
#     win_size: int,
# ) -> Dict[str, List[str]]:
#     train_cmd = [
#         python_exe,
#         "-m",
#         "src.adult.models",
#         "--checkpoint",
#         checkpoint,
#     ]
#
#     precompute_cmd = [
#         python_exe,
#         "-m",
#         "src.adult.precompute",
#         "--checkpoint",
#         checkpoint,
#         "--minsup",
#         str(minsup),
#         "--n-proc",
#         str(n_proc),
#     ]
#
#     if implementation == "baseline":
#         inject_cmd = [
#             python_exe,
#             "-m",
#             "src.adult.inject",
#             "--checkpoint",
#             checkpoint,
#             "--frac-noise",
#             str(noise),
#             "--n-targets",
#             str(n_targets),
#             "--n-proc",
#             str(n_proc),
#             "--resume",
#         ]
#     else:
#         inject_cmd = [
#             python_exe,
#             "-m",
#             "src.adult.inject_adaptive",
#             "--checkpoint",
#             checkpoint,
#             "--frac-noise",
#             str(noise),
#             "--n-targets",
#             str(n_targets),
#             "--adaptive-enable",
#             "--adaptive-top-k",
#             str(adaptive_top_k),
#             "--adaptive-interval",
#             str(adaptive_interval),
#             "--adaptive-score",
#             str(adaptive_score),
#             "--adaptive-min-groups",
#             str(adaptive_min_groups),
#             "--win-size",
#             str(win_size),
#             "--resume",
#         ]
#
#     if light_output:
#         inject_cmd.append("--light-output")
#
#     return {
#         "train": train_cmd,
#         "precompute": precompute_cmd,
#         "inject": inject_cmd,
#     }
#
#
# def safe_ratio(num, den) -> Optional[float]:
#     if num is None or den is None:
#         return None
#     if pd.isna(num) or pd.isna(den) or den == 0:
#         return None
#     return float(num) / float(den)
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Build one runtime/storage comparison table.")
#     parser.add_argument("--project-root", type=Path, default=Path("."))
#     parser.add_argument("--python-exe", type=str, default=sys.executable)
#     parser.add_argument("--models-ckpt", type=Path, default=Path("models-ckpt"))
#     parser.add_argument("--baseline-checkpoint", required=True)
#     parser.add_argument("--adaptive-checkpoint", required=True)
#     parser.add_argument("--output-dir", type=Path, default=Path("report-metrics/runtime_storage_compare"))
#
#     parser.add_argument(
#         "--rerun-stage",
#         choices=["none", "inject", "all"],
#         default="none",
#         help="none = only inspect existing artifacts; inject = rerun inject only; all = rerun train/precompute/inject",
#     )
#
#     parser.add_argument("--minsup", type=float, default=0.05)
#     parser.add_argument("--n-proc", type=int, default=2)
#     parser.add_argument("--n-targets", type=int, default=50)
#     parser.add_argument("--noise", type=float, default=0.50)
#     parser.add_argument("--win-size", type=int, default=5)
#     parser.add_argument("--light-output", action="store_true")
#
#     parser.add_argument("--adaptive-top-k", type=int, default=500)
#     parser.add_argument("--adaptive-interval", type=int, default=5)
#     parser.add_argument("--adaptive-score", type=str, default="abs_tstat")
#     parser.add_argument("--adaptive-min-groups", type=int, default=100)
#
#     args = parser.parse_args()
#     args.output_dir.mkdir(parents=True, exist_ok=True)
#
#     rows: List[Dict[str, object]] = []
#
#     impls = [
#         ("baseline", args.baseline_checkpoint),
#         ("adaptive", args.adaptive_checkpoint),
#     ]
#
#     for implementation, checkpoint in impls:
#         stage_runtime = {
#             "train_runtime_sec": None,
#             "precompute_runtime_sec": None,
#             "inject_runtime_sec": None,
#             "total_runtime_sec": None,
#         }
#         stage_peak = {
#             "train_peak_mb": None,
#             "precompute_peak_mb": None,
#             "inject_peak_mb": None,
#             "peak_memory_mb": None,
#         }
#         stage_rc = {
#             "train_returncode": None,
#             "precompute_returncode": None,
#             "inject_returncode": None,
#         }
#
#         commands = build_stage_commands(
#             python_exe=args.python_exe,
#             implementation=implementation,
#             checkpoint=checkpoint,
#             minsup=args.minsup,
#             n_proc=args.n_proc,
#             noise=args.noise,
#             n_targets=args.n_targets,
#             light_output=args.light_output,
#             adaptive_top_k=args.adaptive_top_k,
#             adaptive_interval=args.adaptive_interval,
#             adaptive_score=args.adaptive_score,
#             adaptive_min_groups=args.adaptive_min_groups,
#             win_size=args.win_size,
#         )
#
#         stages_to_run: List[str] = []
#         if args.rerun_stage == "inject":
#             stages_to_run = ["inject"]
#         elif args.rerun_stage == "all":
#             stages_to_run = ["train", "precompute", "inject"]
#
#         for stage in stages_to_run:
#             cmd = commands[stage]
#             rc, runtime_sec, peak_mb = run_stage(cmd, args.project_root)
#
#             stage_runtime[f"{stage}_runtime_sec"] = runtime_sec
#             stage_peak[f"{stage}_peak_mb"] = peak_mb
#             stage_rc[f"{stage}_returncode"] = rc
#
#             if rc != 0:
#                 print(f"[WARN] {implementation}:{stage} failed with return code {rc}")
#
#         runtime_vals = [v for v in stage_runtime.values() if isinstance(v, (int, float))]
#         peak_vals = [v for v in stage_peak.values() if isinstance(v, (int, float))]
#
#         if runtime_vals:
#             stage_runtime["total_runtime_sec"] = float(np.sum(runtime_vals))
#         if peak_vals:
#             stage_peak["peak_memory_mb"] = float(np.max(peak_vals))
#
#         subgroup_count = estimate_subgroup_count(args.models_ckpt, checkpoint)
#         active_stats = estimate_active_group_stats(args.models_ckpt, checkpoint)
#         effective_monitored = subgroup_count
#         if active_stats is not None:
#             effective_monitored = active_stats["avg_active_groups"]
#
#         artifact_stats = collect_artifact_stats(args.models_ckpt, checkpoint)
#
#         row: Dict[str, object] = {
#             "implementation": implementation,
#             "checkpoint": checkpoint,
#             "rerun_stage": args.rerun_stage,
#             "n_targets": args.n_targets,
#             "noise": args.noise,
#             "minsup": args.minsup,
#             "n_proc": args.n_proc,
#             "win_size": args.win_size,
#             "light_output": args.light_output,
#             "full_subgroup_count": subgroup_count,
#             "effective_monitored_subgroups": effective_monitored,
#             **stage_runtime,
#             **stage_peak,
#             **stage_rc,
#             **artifact_stats,
#         }
#
#         if active_stats is not None:
#             row.update(active_stats)
#
#         rows.append(row)
#
#     df = pd.DataFrame(rows)
#
#     if len(df) == 2:
#         base = df[df["implementation"] == "baseline"].iloc[0]
#         adapt = df[df["implementation"] == "adaptive"].iloc[0]
#
#         subgroup_ratio = safe_ratio(
#             adapt.get("effective_monitored_subgroups"),
#             base.get("effective_monitored_subgroups"),
#         )
#         runtime_ratio = safe_ratio(
#             adapt.get("total_runtime_sec"),
#             base.get("total_runtime_sec"),
#         )
#         storage_ratio = safe_ratio(
#             adapt.get("total_artifact_bytes"),
#             base.get("total_artifact_bytes"),
#         )
#
#         df.loc[df["implementation"] == "adaptive", "subgroup_ratio_vs_baseline"] = subgroup_ratio
#         df.loc[df["implementation"] == "adaptive", "runtime_ratio_vs_baseline"] = runtime_ratio
#         df.loc[df["implementation"] == "adaptive", "storage_ratio_vs_baseline"] = storage_ratio
#
#         df.loc[df["implementation"] == "adaptive", "subgroup_reduction_vs_baseline"] = (
#             1.0 - subgroup_ratio if subgroup_ratio is not None else None
#         )
#         df.loc[df["implementation"] == "adaptive", "runtime_reduction_vs_baseline"] = (
#             1.0 - runtime_ratio if runtime_ratio is not None else None
#         )
#         df.loc[df["implementation"] == "adaptive", "storage_reduction_vs_baseline"] = (
#             1.0 - storage_ratio if storage_ratio is not None else None
#         )
#
#     out_csv = args.output_dir / "runtime_storage_compare.csv"
#     out_md = args.output_dir / "runtime_storage_compare.md"
#
#     df.to_csv(out_csv, index=False)
#
#     # Small pretty markdown table for report use
#     md_cols = [
#         "implementation",
#         "checkpoint",
#         "total_runtime_sec",
#         "peak_memory_mb",
#         "full_subgroup_count",
#         "effective_monitored_subgroups",
#         "total_artifact_mb",
#         "supwise_file_count",
#         "support_bucket_count",
#     ]
#     md_df = df[[c for c in md_cols if c in df.columns]].copy()
#     with open(out_md, "w", encoding="utf-8") as f:
#         f.write(md_df.to_markdown(index=False))
#
#     print("\n=== Runtime / Storage Compare ===")
#     print(df.to_string(index=False))
#     print(f"\nSaved CSV: {out_csv}")
#     print(f"Saved MD : {out_md}")
#
#
# if __name__ == "__main__":
#     main()