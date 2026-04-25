#!/usr/bin/env python3
"""
benchmark_compare.py

Compare baseline vs adaptive DriftInspector runs using:
- existing artifacts in models-ckpt/
- existing summary CSVs in report-metrics/
- optional stage reruns for timing/memory measurement

Examples:
python benchmark_compare.py --baseline-checkpoint adult_model --adaptive-checkpoint adult_model_adaptive
python benchmark_compare.py --project-root . --python-exe .venv/Scripts/python.exe --baseline-checkpoint adult_model --adaptive-checkpoint adult_model_adaptive --rerun
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    import psutil
except Exception:
    psutil = None


def file_size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size if path.exists() else 0
    except Exception:
        return 0


def list_files(path: Path, pattern: str = "*") -> List[Path]:
    if not path.exists():
        return []
    return [p for p in path.rglob(pattern) if p.is_file()]


def safe_pickle_load(path: Path):
    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def parse_support_bucket(fname: str) -> Optional[Tuple[float, float]]:
    import re
    m = re.search(r"support-([0-9.]+)-([0-9.]+)-", fname)
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
    path = models_ckpt / f"matches-{checkpoint}.pkl"
    obj = safe_pickle_load(path)
    if obj is None:
        return None
    try:
        return len(obj["matches_train"].fi)
    except Exception:
        return None


def estimate_active_group_stats(models_ckpt: Path, checkpoint: str):
    """
    For adaptive runs, infer effective monitored subgroup counts from saved payloads.
    For baseline runs, this usually returns None.
    """
    supwise_files = list_files(models_ckpt / "sup-wise", f"{checkpoint}-*.pkl")
    counts = []

    for p in supwise_files:
        obj = safe_pickle_load(p)
        if isinstance(obj, dict) and "active_group_counts" in obj:
            try:
                counts.extend(int(x) for x in obj["active_group_counts"])
            except Exception:
                pass

    if not counts:
        return None

    return {
        "avg_active_groups": float(np.mean(counts)),
        "min_active_groups": int(np.min(counts)),
        "max_active_groups": int(np.max(counts)),
    }


def collect_artifacts(models_ckpt: Path, checkpoint: str) -> Dict[str, int]:
    model_path = models_ckpt / f"{checkpoint}.pkl"
    dataset_path = models_ckpt / f"{checkpoint}.dataset.pkl"
    matches_path = models_ckpt / f"matches-{checkpoint}.pkl"
    supwise_dir = models_ckpt / "sup-wise"

    supwise_files = list_files(supwise_dir, f"{checkpoint}-*.pkl")

    drift_dirs = [p for p in models_ckpt.glob(f"{checkpoint}-accuracy-noise-*") if p.is_dir()]
    drift_files = []
    for d in drift_dirs:
        drift_files.extend(list_files(d, "*.pkl"))

    model_bytes = file_size_bytes(model_path)
    dataset_bytes = file_size_bytes(dataset_path)
    matches_bytes = file_size_bytes(matches_path)
    supwise_bytes = sum(file_size_bytes(p) for p in supwise_files)
    drift_eval_bytes = sum(file_size_bytes(p) for p in drift_files)

    return {
        "model_bytes": model_bytes,
        "dataset_bytes": dataset_bytes,
        "matches_bytes": matches_bytes,
        "supwise_bytes": supwise_bytes,
        "drift_eval_bytes": drift_eval_bytes,
        "total_artifact_bytes": model_bytes + dataset_bytes + matches_bytes + supwise_bytes + drift_eval_bytes,
        "supwise_file_count": len(supwise_files),
        "drift_eval_file_count": len(drift_files),
    }


def run_stage(cmd: List[str], cwd: Path) -> Tuple[int, float, Optional[float]]:
    """
    Run a single subprocess stage and measure:
    - return code
    - wall-clock runtime
    - peak RSS memory, if psutil is available
    """
    start = time.perf_counter()
    peak_mb = None

    proc = subprocess.Popen(cmd, cwd=str(cwd))
    ps_proc = psutil.Process(proc.pid) if psutil is not None else None
    peak_rss = 0

    while True:
        ret = proc.poll()
        if ps_proc is not None:
            try:
                peak_rss = max(peak_rss, ps_proc.memory_info().rss)
            except Exception:
                pass
        if ret is not None:
            break
        time.sleep(0.1)

    runtime = time.perf_counter() - start
    if peak_rss > 0:
        peak_mb = peak_rss / (1024 * 1024)

    return proc.returncode, runtime, peak_mb


def load_scalar_metric(csv_path: Path, implementation: str, metric_col: str, method: Optional[str] = None) -> Optional[float]:
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    try:
        if method is not None and "method" in df.columns:
            row = df[(df["implementation"] == implementation) & (df["method"] == method)]
        else:
            row = df[df["implementation"] == implementation]

        if row.empty or metric_col not in row.columns:
            return None

        return float(row.iloc[0][metric_col])
    except Exception:
        return None


def safe_ratio(num, den):
    if den is None or pd.isna(den) or den == 0 or num is None or pd.isna(num):
        return None
    return float(num / den)


def append_progress_row(csv_path: Path, row: Dict):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_progress(txt_path: Path, msg: str):
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs adaptive efficiency.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--python-exe", type=Path, default=Path("python"))
    parser.add_argument("--models-ckpt", type=Path, default=Path("models-ckpt"))
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--adaptive-checkpoint", required=True)

    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--adaptive-inject-script", type=str, default="src/adult/inject_adaptive.py")
    parser.add_argument("--adaptive-top-k", type=int, default=200)
    parser.add_argument("--adaptive-interval", type=int, default=5)
    parser.add_argument("--adaptive-score", type=str, default="abs_tstat")
    parser.add_argument("--adaptive-min-groups", type=int, default=100)
    parser.add_argument("--adaptive-mode", choices=["active_only", "stability", "refresh"], default="active_only")
    parser.add_argument("--stable-score-threshold", type=float, default=0.0)
    parser.add_argument("--stable-rounds", type=int, default=2)
    parser.add_argument("--refresh-interval", type=int, default=3)
    parser.add_argument("--refresh-top-k", type=int, default=None)
    parser.add_argument("--refresh-threshold", type=float, default=None)

    parser.add_argument("--n-targets", type=int, default=10)
    parser.add_argument("--minsup", type=float, default=0.05)
    parser.add_argument("--noise", type=float, default=0.50)
    parser.add_argument("--metrics-dir", type=Path, default=Path("report-metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("report-metrics"))
    parser.add_argument("--light-output", action="store_true")
    parser.add_argument("--n-proc", type=int, default=2)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_csv = args.output_dir / "benchmark_compare_progress.csv"
    progress_txt = args.output_dir / "benchmark_compare_progress.txt"

    rows = []

    impls = [
        ("baseline", args.baseline_checkpoint),
        ("adaptive", args.adaptive_checkpoint),
    ]

    for implementation, checkpoint in tqdm(impls, desc="Implementations"):
        stage_times = {
            "train_runtime_sec": None,
            "precompute_runtime_sec": None,
            "inject_runtime_sec": None,
        }
        stage_mem = {
            "train_peak_mb": None,
            "precompute_peak_mb": None,
            "inject_peak_mb": None,
        }

        if args.rerun:
            if implementation == "baseline":
                inject_cmd = [
                    str(args.python_exe), "-m", "src.adult.inject",
                    f"--checkpoint={checkpoint}",
                    f"--frac-noise={args.noise}",
                    f"--n-targets={args.n_targets}",
                    f"--n-proc={args.n_proc}",
                ]
                if args.light_output:
                    inject_cmd.append("--light-output")
            else:
                inject_cmd = [
                    str(args.python_exe), args.adaptive_inject_script,
                    "--checkpoint", checkpoint,
                    "--frac-noise", str(args.noise),
                    "--n-targets", str(args.n_targets),
                    "--adaptive-enable",
                    "--adaptive-mode", str(args.adaptive_mode),
                    "--adaptive-interval", str(args.adaptive_interval),
                    "--adaptive-score", str(args.adaptive_score),
                    "--adaptive-min-groups", str(args.adaptive_min_groups),
                    "--win-size", "5",
                    "--stable-score-threshold", str(args.stable_score_threshold),
                    "--stable-rounds", str(args.stable_rounds),
                    "--refresh-interval", str(args.refresh_interval),
                ]
                if args.adaptive_top_k is not None:
                    inject_cmd.extend(["--adaptive-top-k", str(args.adaptive_top_k)])
                if args.refresh_top_k is not None:
                    inject_cmd.extend(["--refresh-top-k", str(args.refresh_top_k)])
                if args.refresh_threshold is not None:
                    inject_cmd.extend(["--refresh-threshold", str(args.refresh_threshold)])
                if args.light_output:
                    inject_cmd.append("--light-output")

            commands = {
                "train": [
                    str(args.python_exe), "-m", "src.adult.models",
                    f"--checkpoint={checkpoint}",
                ],
                "precompute": [
                    str(args.python_exe), "-m", "src.adult.precompute",
                    f"--checkpoint={checkpoint}",
                    f"--minsup={args.minsup}",
                    f"--n-proc={args.n_proc}",
                ],
                "inject": inject_cmd,
            }

            for stage, cmd in tqdm(commands.items(), desc=f"Stages:{implementation}", leave=False):
                log_progress(progress_txt, f"[RUN] {implementation} {stage}: {' '.join(cmd)}")
                rc, runtime, peak_mb = run_stage(cmd, args.project_root)
                if rc != 0:
                    print(f"[WARN] Stage {stage} failed for {checkpoint} with return code {rc}")
                    log_progress(progress_txt, f"[WARN] {implementation} {stage} failed rc={rc}")

                stage_times[f"{stage}_runtime_sec"] = runtime
                stage_mem[f"{stage}_peak_mb"] = peak_mb

                append_progress_row(progress_csv, {
                    "implementation": implementation,
                    "checkpoint": checkpoint,
                    "stage": stage,
                    "returncode": rc,
                    "runtime_sec": runtime,
                    "peak_mb": peak_mb,
                    "command": " ".join(cmd),
                })

        artifacts = collect_artifacts(args.models_ckpt, checkpoint)
        subgroup_count = estimate_subgroup_count(args.models_ckpt, checkpoint)
        active_stats = estimate_active_group_stats(args.models_ckpt, checkpoint)

        effective_group_count = subgroup_count
        if active_stats is not None:
            effective_group_count = active_stats["avg_active_groups"]

        supwise_files = list_files(args.models_ckpt / "sup-wise", f"{checkpoint}-*.pkl")
        support_bucket_count, support_bucket_summary = summarize_support_buckets(supwise_files)

        ranking_csv = args.metrics_dir / "ranking_summary_overall.csv"
        detection_csv = args.metrics_dir / "detection_summary_overall.csv"

        ranking_ndcg = load_scalar_metric(ranking_csv, implementation, "nDCG", method="tstat")
        detection_f1 = load_scalar_metric(detection_csv, implementation, "F1")

        row = {
            "implementation": implementation,
            "checkpoint": checkpoint,
            "total_runtime_sec": (
                np.nansum([v for v in stage_times.values() if v is not None])
                if any(v is not None for v in stage_times.values())
                else None
            ),
            "peak_memory_mb": (
                np.nanmax([v for v in stage_mem.values() if v is not None])
                if any(v is not None for v in stage_mem.values())
                else None
            ),
            "full_subgroup_count": subgroup_count,
            "effective_monitored_subgroups": effective_group_count,
            "support_bucket_count": support_bucket_count,
            "support_bucket_summary": json.dumps(support_bucket_summary),
            "ranking_ndcg_tstat": ranking_ndcg,
            "detection_f1": detection_f1,
            **stage_times,
            **stage_mem,
            **artifacts,
        }

        if active_stats is not None:
            row.update(active_stats)

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) == 2:
        base = df[df["implementation"] == "baseline"].iloc[0]
        adapt = df[df["implementation"] == "adaptive"].iloc[0]

        subgroup_ratio = safe_ratio(adapt["effective_monitored_subgroups"], base["effective_monitored_subgroups"])
        runtime_ratio = safe_ratio(adapt["total_runtime_sec"], base["total_runtime_sec"])

        df.loc[df["implementation"] == "adaptive", "subgroup_reduction_ratio"] = (
            1.0 - subgroup_ratio if subgroup_ratio is not None else None
        )
        df.loc[df["implementation"] == "adaptive", "runtime_reduction_ratio"] = (
            1.0 - runtime_ratio if runtime_ratio is not None else None
        )
        df.loc[df["implementation"] == "adaptive", "ranking_retention_ratio"] = safe_ratio(
            adapt["ranking_ndcg_tstat"], base["ranking_ndcg_tstat"]
        )
        df.loc[df["implementation"] == "adaptive", "detection_retention_ratio"] = safe_ratio(
            adapt["detection_f1"], base["detection_f1"]
        )

    out_csv = args.output_dir / "benchmark_compare.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Benchmark Compare ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()



### Updated 4/21
# #!/usr/bin/env python3
# """
# benchmark_compare.py
#
# Compare baseline vs adaptive DriftInspector runs using:
# - existing artifacts in models-ckpt/
# - existing summary CSVs in report-metrics/
# - optional stage reruns for timing/memory measurement
#
# Examples:
# python benchmark_compare.py --baseline-checkpoint adult_model --adaptive-checkpoint adult_model_adaptive
# python benchmark_compare.py --project-root . --python-exe .venv/Scripts/python.exe --baseline-checkpoint adult_model --adaptive-checkpoint adult_model_adaptive --rerun
# """
#
# from __future__ import annotations
#
# import argparse
# import csv
# import json
# import subprocess
# import time
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple
#
# import numpy as np
# import pandas as pd
# from tqdm.auto import tqdm
#
# try:
#     import psutil
# except Exception:
#     psutil = None
#
#
# def file_size_bytes(path: Path) -> int:
#     try:
#         return path.stat().st_size if path.exists() else 0
#     except Exception:
#         return 0
#
#
# def list_files(path: Path, pattern: str = "*") -> List[Path]:
#     if not path.exists():
#         return []
#     return [p for p in path.rglob(pattern) if p.is_file()]
#
#
# def safe_pickle_load(path: Path):
#     try:
#         import pickle
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     except Exception:
#         return None
#
#
# def parse_support_bucket(fname: str) -> Optional[Tuple[float, float]]:
#     import re
#     m = re.search(r"support-([0-9.]+)-([0-9.]+)-", fname)
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
#     path = models_ckpt / f"matches-{checkpoint}.pkl"
#     obj = safe_pickle_load(path)
#     if obj is None:
#         return None
#     try:
#         return len(obj["matches_train"].fi)
#     except Exception:
#         return None
#
#
# def estimate_active_group_stats(models_ckpt: Path, checkpoint: str):
#     """
#     For adaptive runs, infer effective monitored subgroup counts from saved payloads.
#     For baseline runs, this usually returns None.
#     """
#     supwise_files = list_files(models_ckpt / "sup-wise", f"{checkpoint}-*.pkl")
#     counts = []
#
#     for p in supwise_files:
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
#     return {
#         "avg_active_groups": float(np.mean(counts)),
#         "min_active_groups": int(np.min(counts)),
#         "max_active_groups": int(np.max(counts)),
#     }
#
#
# def collect_artifacts(models_ckpt: Path, checkpoint: str) -> Dict[str, int]:
#     model_path = models_ckpt / f"{checkpoint}.pkl"
#     dataset_path = models_ckpt / f"{checkpoint}.dataset.pkl"
#     matches_path = models_ckpt / f"matches-{checkpoint}.pkl"
#     supwise_dir = models_ckpt / "sup-wise"
#
#     supwise_files = list_files(supwise_dir, f"{checkpoint}-*.pkl")
#
#     drift_dirs = [p for p in models_ckpt.glob(f"{checkpoint}-accuracy-noise-*") if p.is_dir()]
#     drift_files = []
#     for d in drift_dirs:
#         drift_files.extend(list_files(d, "*.pkl"))
#
#     model_bytes = file_size_bytes(model_path)
#     dataset_bytes = file_size_bytes(dataset_path)
#     matches_bytes = file_size_bytes(matches_path)
#     supwise_bytes = sum(file_size_bytes(p) for p in supwise_files)
#     drift_eval_bytes = sum(file_size_bytes(p) for p in drift_files)
#
#     return {
#         "model_bytes": model_bytes,
#         "dataset_bytes": dataset_bytes,
#         "matches_bytes": matches_bytes,
#         "supwise_bytes": supwise_bytes,
#         "drift_eval_bytes": drift_eval_bytes,
#         "total_artifact_bytes": model_bytes + dataset_bytes + matches_bytes + supwise_bytes + drift_eval_bytes,
#         "supwise_file_count": len(supwise_files),
#         "drift_eval_file_count": len(drift_files),
#     }
#
#
# def run_stage(cmd: List[str], cwd: Path) -> Tuple[int, float, Optional[float]]:
#     """
#     Run a single subprocess stage and measure:
#     - return code
#     - wall-clock runtime
#     - peak RSS memory, if psutil is available
#     """
#     start = time.perf_counter()
#     peak_mb = None
#
#     proc = subprocess.Popen(cmd, cwd=str(cwd))
#     ps_proc = psutil.Process(proc.pid) if psutil is not None else None
#     peak_rss = 0
#
#     while True:
#         ret = proc.poll()
#         if ps_proc is not None:
#             try:
#                 peak_rss = max(peak_rss, ps_proc.memory_info().rss)
#             except Exception:
#                 pass
#         if ret is not None:
#             break
#         time.sleep(0.1)
#
#     runtime = time.perf_counter() - start
#     if peak_rss > 0:
#         peak_mb = peak_rss / (1024 * 1024)
#
#     return proc.returncode, runtime, peak_mb
#
#
# def load_scalar_metric(csv_path: Path, implementation: str, metric_col: str, method: Optional[str] = None) -> Optional[float]:
#     if not csv_path.exists():
#         return None
#
#     df = pd.read_csv(csv_path)
#     try:
#         if method is not None and "method" in df.columns:
#             row = df[(df["implementation"] == implementation) & (df["method"] == method)]
#         else:
#             row = df[df["implementation"] == implementation]
#
#         if row.empty or metric_col not in row.columns:
#             return None
#
#         return float(row.iloc[0][metric_col])
#     except Exception:
#         return None
#
#
# def safe_ratio(num, den):
#     if den is None or pd.isna(den) or den == 0 or num is None or pd.isna(num):
#         return None
#     return float(num / den)
#
#
# def append_progress_row(csv_path: Path, row: Dict):
#     write_header = not csv_path.exists()
#     with open(csv_path, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(row.keys()))
#         if write_header:
#             writer.writeheader()
#         writer.writerow(row)
#
#
# def log_progress(txt_path: Path, msg: str):
#     with open(txt_path, "a", encoding="utf-8") as f:
#         f.write(msg + "\n")
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Compare baseline vs adaptive efficiency.")
#     parser.add_argument("--project-root", type=Path, default=Path("."))
#     parser.add_argument("--python-exe", type=Path, default=Path("python"))
#     parser.add_argument("--models-ckpt", type=Path, default=Path("models-ckpt"))
#     parser.add_argument("--baseline-checkpoint", required=True)
#     parser.add_argument("--adaptive-checkpoint", required=True)
#
#     parser.add_argument("--rerun", action="store_true")
#     parser.add_argument("--adaptive-inject-script", type=str, default="src/adult/inject_adaptive.py")
#     parser.add_argument("--adaptive-top-k", type=int, default=500)
#     parser.add_argument("--adaptive-interval", type=int, default=5)
#     parser.add_argument("--adaptive-score", type=str, default="abs_tstat")
#     parser.add_argument("--adaptive-min-groups", type=int, default=100)
#
#     parser.add_argument("--n-targets", type=int, default=10)
#     parser.add_argument("--minsup", type=float, default=0.05)
#     parser.add_argument("--noise", type=float, default=0.50)
#     parser.add_argument("--metrics-dir", type=Path, default=Path("report-metrics"))
#     parser.add_argument("--output-dir", type=Path, default=Path("report-metrics"))
#     parser.add_argument("--light-output", action="store_true")
#     parser.add_argument("--n-proc", type=int, default=2)
#
#     args = parser.parse_args()
#
#     args.output_dir.mkdir(parents=True, exist_ok=True)
#     progress_csv = args.output_dir / "benchmark_compare_progress.csv"
#     progress_txt = args.output_dir / "benchmark_compare_progress.txt"
#
#     rows = []
#
#     impls = [
#         ("baseline", args.baseline_checkpoint),
#         ("adaptive", args.adaptive_checkpoint),
#     ]
#
#     for implementation, checkpoint in tqdm(impls, desc="Implementations"):
#         stage_times = {
#             "train_runtime_sec": None,
#             "precompute_runtime_sec": None,
#             "inject_runtime_sec": None,
#         }
#         stage_mem = {
#             "train_peak_mb": None,
#             "precompute_peak_mb": None,
#             "inject_peak_mb": None,
#         }
#
#         if args.rerun:
#             if implementation == "baseline":
#                 inject_cmd = [
#                     str(args.python_exe), "-m", "src.adult.inject",
#                     f"--checkpoint={checkpoint}",
#                     f"--frac-noise={args.noise}",
#                     f"--n-targets={args.n_targets}",
#                     f"--n-proc={args.n_proc}",
#                 ]
#                 if args.light_output:
#                     inject_cmd.append("--light-output")
#             else:
#                 inject_cmd = [
#                     str(args.python_exe), args.adaptive_inject_script,
#                     "--checkpoint", checkpoint,
#                     "--frac-noise", str(args.noise),
#                     "--n-targets", str(args.n_targets),
#                     "--adaptive-enable",
#                     "--adaptive-top-k", str(args.adaptive_top_k),
#                     "--adaptive-interval", str(args.adaptive_interval),
#                     "--adaptive-score", str(args.adaptive_score),
#                     "--adaptive-min-groups", str(args.adaptive_min_groups),
#                     "--win-size", "5",
#                 ]
#                 if args.light_output:
#                     inject_cmd.append("--light-output")
#
#             commands = {
#                 "train": [
#                     str(args.python_exe), "-m", "src.adult.models",
#                     f"--checkpoint={checkpoint}",
#                 ],
#                 "precompute": [
#                     str(args.python_exe), "-m", "src.adult.precompute",
#                     f"--checkpoint={checkpoint}",
#                     f"--minsup={args.minsup}",
#                     f"--n-proc={args.n_proc}",
#                 ],
#                 "inject": inject_cmd,
#             }
#
#             for stage, cmd in tqdm(commands.items(), desc=f"Stages:{implementation}", leave=False):
#                 log_progress(progress_txt, f"[RUN] {implementation} {stage}: {' '.join(cmd)}")
#                 rc, runtime, peak_mb = run_stage(cmd, args.project_root)
#                 if rc != 0:
#                     print(f"[WARN] Stage {stage} failed for {checkpoint} with return code {rc}")
#                     log_progress(progress_txt, f"[WARN] {implementation} {stage} failed rc={rc}")
#
#                 stage_times[f"{stage}_runtime_sec"] = runtime
#                 stage_mem[f"{stage}_peak_mb"] = peak_mb
#
#                 append_progress_row(progress_csv, {
#                     "implementation": implementation,
#                     "checkpoint": checkpoint,
#                     "stage": stage,
#                     "returncode": rc,
#                     "runtime_sec": runtime,
#                     "peak_mb": peak_mb,
#                     "command": " ".join(cmd),
#                 })
#
#         artifacts = collect_artifacts(args.models_ckpt, checkpoint)
#         subgroup_count = estimate_subgroup_count(args.models_ckpt, checkpoint)
#         active_stats = estimate_active_group_stats(args.models_ckpt, checkpoint)
#
#         effective_group_count = subgroup_count
#         if active_stats is not None:
#             effective_group_count = active_stats["avg_active_groups"]
#
#         supwise_files = list_files(args.models_ckpt / "sup-wise", f"{checkpoint}-*.pkl")
#         support_bucket_count, support_bucket_summary = summarize_support_buckets(supwise_files)
#
#         ranking_csv = args.metrics_dir / "ranking_summary_overall.csv"
#         detection_csv = args.metrics_dir / "detection_summary_overall.csv"
#
#         ranking_ndcg = load_scalar_metric(ranking_csv, implementation, "nDCG", method="tstat")
#         detection_f1 = load_scalar_metric(detection_csv, implementation, "F1")
#
#         row = {
#             "implementation": implementation,
#             "checkpoint": checkpoint,
#             "total_runtime_sec": (
#                 np.nansum([v for v in stage_times.values() if v is not None])
#                 if any(v is not None for v in stage_times.values())
#                 else None
#             ),
#             "peak_memory_mb": (
#                 np.nanmax([v for v in stage_mem.values() if v is not None])
#                 if any(v is not None for v in stage_mem.values())
#                 else None
#             ),
#             "full_subgroup_count": subgroup_count,
#             "effective_monitored_subgroups": effective_group_count,
#             "support_bucket_count": support_bucket_count,
#             "support_bucket_summary": json.dumps(support_bucket_summary),
#             "ranking_ndcg_tstat": ranking_ndcg,
#             "detection_f1": detection_f1,
#             **stage_times,
#             **stage_mem,
#             **artifacts,
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
#         subgroup_ratio = safe_ratio(adapt["effective_monitored_subgroups"], base["effective_monitored_subgroups"])
#         runtime_ratio = safe_ratio(adapt["total_runtime_sec"], base["total_runtime_sec"])
#
#         df.loc[df["implementation"] == "adaptive", "subgroup_reduction_ratio"] = (
#             1.0 - subgroup_ratio if subgroup_ratio is not None else None
#         )
#         df.loc[df["implementation"] == "adaptive", "runtime_reduction_ratio"] = (
#             1.0 - runtime_ratio if runtime_ratio is not None else None
#         )
#         df.loc[df["implementation"] == "adaptive", "ranking_retention_ratio"] = safe_ratio(
#             adapt["ranking_ndcg_tstat"], base["ranking_ndcg_tstat"]
#         )
#         df.loc[df["implementation"] == "adaptive", "detection_retention_ratio"] = safe_ratio(
#             adapt["detection_f1"], base["detection_f1"]
#         )
#
#     out_csv = args.output_dir / "benchmark_compare.csv"
#     df.to_csv(out_csv, index=False)
#
#     print("\n=== Benchmark Compare ===")
#     print(df.to_string(index=False))
#     print(f"\nSaved: {out_csv}")
#
#
# if __name__ == "__main__":
#     main()