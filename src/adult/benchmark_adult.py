#!/usr/bin/env python3
"""
benchmark_adult.py

Benchmark and evaluation harness for comparing two DriftInspector implementations
(e.g. baseline vs adjusted) on the Adult dataset.

Designed for project layout like:
    src/adult/models.py
    src/adult/precompute.py
    src/adult/inject.py
    src/adult/drift-eval.py
    src/adult/ranking.py
    src/adult/config.py
    data/
    models-ckpt/

This script:
- creates reproducible Adult subsets
- runs pipeline stages via subprocess
- records runtime, CPU time, memory, artifact sizes, file counts
- summarizes support bucket coverage in sup-wise outputs
- saves CSV, plots, and terminal summary tables

Notes:
- This is a benchmark/evaluation harness, not a strict unit test.
- It is resilient to partial failures and records them in a manifest.
- It assumes your baseline and adjusted implementations are reachable either
  through different checkpoints or different command templates.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import psutil
except Exception:
    psutil = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class StageResult:
    implementation: str
    subset_name: str
    subset_rows_train: int
    subset_rows_test: int
    repetition: int
    stage: str
    checkpoint: str
    command: str
    returncode: int
    runtime_sec: float
    cpu_time_sec: Optional[float]
    peak_memory_mb: Optional[float]
    stdout_log: str
    stderr_log: str
    success: bool
    error: Optional[str]
    artifact_paths: List[str]
    artifact_bytes: int
    file_count: int
    subgroup_count: Optional[int]
    batch_count: Optional[int]
    support_bucket_count: Optional[int]
    support_bucket_summary: Optional[Dict[str, int]]
    ranking_metrics: Optional[Dict[str, float]]


# ----------------------------
# Utility helpers
# ----------------------------

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text_tail(path: Path, n_lines: int = 50) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""


def file_size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size if path.exists() else 0
    except Exception:
        return 0


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            total += file_size_bytes(p)
    return total


def count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob(pattern) if p.is_file())


def list_files(path: Path, pattern: str = "*") -> List[Path]:
    if not path.exists():
        return []
    return [p for p in path.rglob(pattern) if p.is_file()]


def parse_support_bucket(fname: str) -> Optional[Tuple[float, float]]:
    m = re.search(r"support-([0-9.]+)-([0-9.]+)-", fname)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def summarize_support_buckets(files: List[Path]) -> Tuple[int, Dict[str, int]]:
    counts: Counter[str] = Counter()
    for f in files:
        bucket = parse_support_bucket(f.name)
        if bucket is not None:
            key = f"{bucket[0]:.4f}-{bucket[1]:.4f}"
            counts[key] += 1
    return len(counts), dict(counts)


def try_pickle_load(path: Path) -> Any:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def estimate_subgroup_count_from_matches(matches_path: Path) -> Optional[int]:
    obj = try_pickle_load(matches_path)
    if obj is None:
        return None
    try:
        matches_train = obj["matches_train"]
        fi = matches_train.fi
        return len(fi)
    except Exception:
        try:
            return len(obj["matches_train"].fi)
        except Exception:
            return None


def estimate_batch_count_from_dataset(dataset_path: Path) -> Optional[int]:
    obj = try_pickle_load(dataset_path)
    if obj is None:
        return None
    try:
        return len(obj["test_chunks"])
    except Exception:
        return None


def read_ranking_metrics_from_stdout(stdout_path: Path) -> Optional[Dict[str, float]]:
    """
    Try to parse metrics from ranking.py stdout if it prints a latex table or plain numbers.
    This is intentionally permissive and non-fatal.
    """
    if not stdout_path.exists():
        return None

    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    metrics: Dict[str, float] = {}

    patterns = {
        "nDCG": r"nDCG[^0-9\-]*([0-9]*\.[0-9]+)",
        "nDCG@10": r"nDCG@10[^0-9\-]*([0-9]*\.[0-9]+)",
        "nDCG@100": r"nDCG@100[^0-9\-]*([0-9]*\.[0-9]+)",
        "Pearson": r"Pearson[^0-9\-]*([\-]?[0-9]*\.[0-9]+)",
        "Spearman": r"Spearman[^0-9\-]*([\-]?[0-9]*\.[0-9]+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except Exception:
                pass

    return metrics or None


def mean_std(values: List[float]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def format_bytes_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def maybe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


# ----------------------------
# Subset generation
# ----------------------------

def make_adult_subset(
    source_data_dir: Path,
    subset_dir: Path,
    train_rows: int,
    test_rows: int,
    seed: int,
) -> Tuple[Path, Path]:
    """
    Create a local subset data directory containing adult.data and adult.test.

    Keeps the original first header line of adult.test if present.
    Uses deterministic row sampling after loading full files.
    """
    safe_mkdir(subset_dir)

    adult_data_src = source_data_dir / "adult.data"
    adult_test_src = source_data_dir / "adult.test"

    if not adult_data_src.exists() or not adult_test_src.exists():
        raise FileNotFoundError(
            f"Missing dataset files in {source_data_dir}. Expected adult.data and adult.test"
        )

    rng = np.random.default_rng(seed)

    # adult.data: plain data rows
    data_lines = adult_data_src.read_text(encoding="utf-8", errors="replace").splitlines()
    data_lines = [x for x in data_lines if x.strip()]
    train_rows = min(train_rows, len(data_lines))
    train_idx = np.sort(rng.choice(len(data_lines), size=train_rows, replace=False))
    subset_train_lines = [data_lines[i] for i in train_idx]

    # adult.test: often first line is a metadata/header/comment line
    test_lines = adult_test_src.read_text(encoding="utf-8", errors="replace").splitlines()
    test_lines = [x for x in test_lines if x.strip()]
    test_header = ""
    test_body = test_lines
    if test_lines and test_lines[0].startswith("|"):
        test_header = test_lines[0]
        test_body = test_lines[1:]

    test_rows = min(test_rows, len(test_body))
    test_idx = np.sort(rng.choice(len(test_body), size=test_rows, replace=False))
    subset_test_lines = [test_body[i] for i in test_idx]

    subset_train_path = subset_dir / "adult.data"
    subset_test_path = subset_dir / "adult.test"

    subset_train_path.write_text("\n".join(subset_train_lines) + "\n", encoding="utf-8")
    if test_header:
        subset_test_path.write_text(
            test_header + "\n" + "\n".join(subset_test_lines) + "\n",
            encoding="utf-8",
        )
    else:
        subset_test_path.write_text("\n".join(subset_test_lines) + "\n", encoding="utf-8")

    return subset_train_path, subset_test_path


# ----------------------------
# Subprocess execution
# ----------------------------

def run_command_with_measurement(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> Tuple[int, float, Optional[float], Optional[float], Optional[str]]:
    """
    Run subprocess and measure:
    - wall clock runtime
    - process CPU time if psutil available
    - peak RSS if psutil available, else tracemalloc fallback for parent only is not useful

    Returns:
        (returncode, runtime_sec, cpu_time_sec, peak_memory_mb, error_string)
    """
    safe_mkdir(stdout_path.parent)
    safe_mkdir(stderr_path.parent)

    start = time.perf_counter()
    error = None
    cpu_time_sec = None
    peak_memory_mb = None

    with open(stdout_path, "w", encoding="utf-8", errors="replace") as fout, \
         open(stderr_path, "w", encoding="utf-8", errors="replace") as ferr:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=fout,
                stderr=ferr,
                text=True,
            )
        except Exception as e:
            return -1, 0.0, None, None, str(e)

        peak_rss = 0
        ps_proc = None

        if psutil is not None:
            try:
                ps_proc = psutil.Process(proc.pid)
            except Exception:
                ps_proc = None

        while True:
            ret = proc.poll()
            if ps_proc is not None:
                try:
                    rss = ps_proc.memory_info().rss
                    peak_rss = max(peak_rss, rss)
                except Exception:
                    pass
            if ret is not None:
                break
            time.sleep(0.1)

        end = time.perf_counter()
        runtime_sec = end - start

        if ps_proc is not None:
            try:
                t = ps_proc.cpu_times()
                cpu_time_sec = float(t.user + t.system)
            except Exception:
                cpu_time_sec = None
            peak_memory_mb = format_bytes_mb(peak_rss) if peak_rss > 0 else None

        return proc.returncode, runtime_sec, cpu_time_sec, peak_memory_mb, error


# ----------------------------
# Artifact discovery
# ----------------------------

def find_stage_artifacts(
    ckpt_dir: Path,
    checkpoint: str,
    stage: str,
    noise: float,
) -> List[Path]:
    artifacts: List[Path] = []

    model_path = ckpt_dir / f"{checkpoint}.pkl"
    dataset_path = ckpt_dir / f"{checkpoint}.dataset.pkl"
    matches_path = ckpt_dir / f"matches-{checkpoint}.pkl"
    supwise_dir = ckpt_dir / "sup-wise"
    drift_eval_dir = ckpt_dir / f"{checkpoint}-accuracy-noise-{noise:.2f}"

    if stage == "train":
        for p in [model_path, dataset_path]:
            if p.exists():
                artifacts.append(p)

    elif stage == "precompute":
        if matches_path.exists():
            artifacts.append(matches_path)

    elif stage == "inject":
        if supwise_dir.exists():
            artifacts.extend(list_files(supwise_dir, f"{checkpoint}-noise-{noise:.2f}-*.pkl"))

    elif stage == "drift_eval":
        if drift_eval_dir.exists():
            artifacts.extend(list_files(drift_eval_dir, "*.pkl"))

    elif stage == "ranking":
        # ranking often emits stdout/logs rather than new project files
        # keep empty here; handled externally if needed
        pass

    elif stage == "full":
        for p in [model_path, dataset_path, matches_path]:
            if p.exists():
                artifacts.append(p)
        if supwise_dir.exists():
            artifacts.extend(list_files(supwise_dir, f"{checkpoint}-noise-{noise:.2f}-*.pkl"))
        if drift_eval_dir.exists():
            artifacts.extend(list_files(drift_eval_dir, "*.pkl"))

    return artifacts


def collect_stage_metadata(
    project_root: Path,
    ckpt_dir: Path,
    checkpoint: str,
    stage: str,
    noise: float,
    ranking_stdout_path: Optional[Path] = None,
) -> Tuple[int, int, Optional[int], Optional[int], Optional[int], Optional[Dict[str, int]], Optional[Dict[str, float]], List[str]]:
    artifacts = find_stage_artifacts(ckpt_dir, checkpoint, stage, noise)
    artifact_bytes = sum(file_size_bytes(p) for p in artifacts)
    file_count = len(artifacts)

    matches_path = ckpt_dir / f"matches-{checkpoint}.pkl"
    dataset_path = ckpt_dir / f"{checkpoint}.dataset.pkl"

    subgroup_count = estimate_subgroup_count_from_matches(matches_path)
    batch_count = estimate_batch_count_from_dataset(dataset_path)

    support_bucket_count = None
    support_bucket_summary = None
    if stage in {"inject", "full"}:
        support_bucket_count, support_bucket_summary = summarize_support_buckets(
            [p for p in artifacts if "support-" in p.name]
        )

    ranking_metrics = None
    if ranking_stdout_path is not None and ranking_stdout_path.exists():
        ranking_metrics = read_ranking_metrics_from_stdout(ranking_stdout_path)

    artifact_paths = [maybe_rel(p, project_root) for p in artifacts]
    return (
        artifact_bytes,
        file_count,
        subgroup_count,
        batch_count,
        support_bucket_count,
        support_bucket_summary,
        ranking_metrics,
        artifact_paths,
    )


# ----------------------------
# Commands
# ----------------------------

def build_stage_command(
    python_exe: Path,
    implementation_mode: str,
    stage: str,
    checkpoint: str,
    noise: float,
    n_targets: int,
    minsup: float,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """
    Build subprocess command.

    implementation_mode:
      - "baseline"
      - "adjusted"

    By default this uses the same commands but a different checkpoint.
    If your adjusted implementation lives in different modules, customize this function.
    """
    extra_args = extra_args or []

    if stage == "train":
        cmd = [str(python_exe), "-m", "src.adult.models", f"--checkpoint={checkpoint}"]

    elif stage == "precompute":
        cmd = [
            str(python_exe), "-m", "src.adult.precompute",
            f"--checkpoint={checkpoint}",
            f"--minsup={minsup}",
        ]

    elif stage == "inject":
        cmd = [
            str(python_exe), "-m", "src.adult.inject",
            f"--checkpoint={checkpoint}",
            f"--frac-noise={noise}",
            f"--n-targets={n_targets}",
        ]

    elif stage == "drift_eval":
        cmd = [
            str(python_exe), "src/adult/drift-eval.py",
            f"--checkpoint={checkpoint}",
            f"--frac-noise={noise}",
            f"--n-targets={n_targets}",
        ]

    elif stage == "ranking":
        cmd = [str(python_exe), "src/adult/ranking.py"]

    elif stage == "full":
        raise ValueError("Use expanded stage loop for 'full' instead of a single command.")
    else:
        raise ValueError(f"Unknown stage: {stage}")

    cmd.extend(extra_args)
    return cmd


# ----------------------------
# Summary tables
# ----------------------------

def summarize_results(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Produce terminal-friendly summary tables.
    """
    out: Dict[str, pd.DataFrame] = {}

    # Runtime by subset size
    rt = (
        df.groupby(["implementation", "subset_name"])["runtime_sec"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out["runtime_by_subset"] = rt

    # Memory by subset size
    if "peak_memory_mb" in df.columns:
        mem = (
            df.groupby(["implementation", "subset_name"])["peak_memory_mb"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        out["memory_by_subset"] = mem

    # Subgroup count by subset size
    if "subgroup_count" in df.columns:
        sg = (
            df.groupby(["implementation", "subset_name"])["subgroup_count"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        out["subgroup_count_by_subset"] = sg

    # Artifact size by subset size
    art = (
        df.groupby(["implementation", "subset_name"])["artifact_bytes"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    art["artifact_mb_mean"] = art["mean"] / (1024 * 1024)
    art["artifact_mb_std"] = art["std"] / (1024 * 1024)
    out["artifact_size_by_subset"] = art

    return out


def print_table(title: str, table: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    if table.empty:
        print("(empty)")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(table.to_string(index=False))


# ----------------------------
# Plotting
# ----------------------------

def save_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, g in df.groupby(group_col):
        g = g.sort_values(x_col)
        ax.plot(g[x_col], g[y_col], marker="o", label=str(label))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Main benchmark flow
# ----------------------------

def benchmark_one_stage(
    project_root: Path,
    python_exe: Path,
    implementation: str,
    checkpoint: str,
    subset_name: str,
    subset_rows_train: int,
    subset_rows_test: int,
    repetition: int,
    stage: str,
    noise: float,
    n_targets: int,
    minsup: float,
    logs_dir: Path,
    ranking_stdout_path: Optional[Path] = None,
) -> StageResult:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    # Use project data/config path as-is. To benchmark subsets fairly, we overwrite
    # the data directory files directly before stage execution.
    cmd = build_stage_command(
        python_exe=python_exe,
        implementation_mode=implementation,
        stage=stage,
        checkpoint=checkpoint,
        noise=noise,
        n_targets=n_targets,
        minsup=minsup,
    )

    stage_log_dir = logs_dir / implementation / subset_name / f"rep_{repetition}"
    safe_mkdir(stage_log_dir)

    stdout_path = stage_log_dir / f"{stage}.stdout.txt"
    stderr_path = stage_log_dir / f"{stage}.stderr.txt"

    returncode, runtime_sec, cpu_time_sec, peak_memory_mb, error = run_command_with_measurement(
        cmd=cmd,
        cwd=project_root,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    ckpt_dir = project_root / "models-ckpt"
    artifact_bytes, file_count, subgroup_count, batch_count, support_bucket_count, support_bucket_summary, ranking_metrics, artifact_paths = collect_stage_metadata(
        project_root=project_root,
        ckpt_dir=ckpt_dir,
        checkpoint=checkpoint,
        stage=stage,
        noise=noise,
        ranking_stdout_path=stdout_path if stage == "ranking" else None,
    )

    success = returncode == 0
    if not success and error is None:
        error = read_text_tail(stderr_path, 50) or f"Stage {stage} failed with return code {returncode}"

    return StageResult(
        implementation=implementation,
        subset_name=subset_name,
        subset_rows_train=subset_rows_train,
        subset_rows_test=subset_rows_test,
        repetition=repetition,
        stage=stage,
        checkpoint=checkpoint,
        command=" ".join(cmd),
        returncode=returncode,
        runtime_sec=runtime_sec,
        cpu_time_sec=cpu_time_sec,
        peak_memory_mb=peak_memory_mb,
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
        success=success,
        error=error,
        artifact_paths=artifact_paths,
        artifact_bytes=artifact_bytes,
        file_count=file_count,
        subgroup_count=subgroup_count,
        batch_count=batch_count,
        support_bucket_count=support_bucket_count,
        support_bucket_summary=support_bucket_summary,
        ranking_metrics=ranking_metrics,
    )


def copy_subset_into_project_data(subset_dir: Path, project_data_dir: Path) -> None:
    """
    Overwrite project data/adult.data and adult.test with the chosen subset.
    """
    safe_mkdir(project_data_dir)
    shutil.copy2(subset_dir / "adult.data", project_data_dir / "adult.data")
    shutil.copy2(subset_dir / "adult.test", project_data_dir / "adult.test")


def remove_checkpoint_artifacts(project_root: Path, checkpoint: str) -> None:
    ckpt_dir = project_root / "models-ckpt"
    paths = [
        ckpt_dir / f"{checkpoint}.pkl",
        ckpt_dir / f"{checkpoint}.dataset.pkl",
        ckpt_dir / f"matches-{checkpoint}.pkl",
        ckpt_dir / f"{checkpoint}-accuracy-noise-0.00",
        ckpt_dir / f"{checkpoint}-accuracy-noise-0.50",
        ckpt_dir / f"{checkpoint}-accuracy-noise-1.00",
    ]
    for p in paths:
        if p.is_file():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)

    # Remove sup-wise files matching this checkpoint
    supwise_dir = ckpt_dir / "sup-wise"
    if supwise_dir.exists():
        for p in supwise_dir.glob(f"{checkpoint}-*.pkl"):
            p.unlink(missing_ok=True)


def flatten_results(results: List[StageResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in results:
        row = asdict(r)
        if r.ranking_metrics:
            for k, v in r.ranking_metrics.items():
                row[f"ranking_{k}"] = v
        row.pop("support_bucket_summary", None)
        row.pop("artifact_paths", None)
        row.pop("ranking_metrics", None)
        rows.append(row)
    return pd.DataFrame(rows)


def empirical_scaling_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rough empirical scaling summary from subset size vs runtime/memory/subgroups.
    """
    rows = []
    grouped = df.groupby(["implementation", "subset_name"], as_index=False).agg(
        subset_rows_train=("subset_rows_train", "first"),
        subset_rows_test=("subset_rows_test", "first"),
        runtime_sec=("runtime_sec", "mean"),
        peak_memory_mb=("peak_memory_mb", "mean"),
        subgroup_count=("subgroup_count", "mean"),
        artifact_bytes=("artifact_bytes", "mean"),
    )

    for impl, g in grouped.groupby("implementation"):
        g = g.sort_values("subset_rows_train")
        if len(g) >= 2:
            x = g["subset_rows_train"].to_numpy(dtype=float)
            rt = g["runtime_sec"].to_numpy(dtype=float)
            mem = np.nan_to_num(g["peak_memory_mb"].to_numpy(dtype=float), nan=0.0)
            sg = np.nan_to_num(g["subgroup_count"].to_numpy(dtype=float), nan=0.0)

            def slope(a: np.ndarray, b: np.ndarray) -> float:
                if len(a) < 2 or np.any(a <= 0) or np.any(b <= 0):
                    return float("nan")
                # log-log slope for practical growth trend
                return float(np.polyfit(np.log(a), np.log(b), 1)[0])

            rows.append({
                "implementation": impl,
                "runtime_growth_exponent": slope(x, rt),
                "memory_growth_exponent": slope(x, mem) if np.any(mem > 0) else np.nan,
                "subgroup_growth_exponent": slope(x, sg) if np.any(sg > 0) else np.nan,
            })

    return pd.DataFrame(rows)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs adjusted DriftInspector on Adult.")
    parser.add_argument("--project-root", type=str, default=".", help="Project root containing src/, data/, models-ckpt/")
    parser.add_argument("--python-exe", type=str, default=sys.executable, help="Python executable to use for subprocess runs.")
    parser.add_argument("--baseline-checkpoint", type=str, default="adult_model", help="Checkpoint name for baseline.")
    parser.add_argument("--adjusted-checkpoint", type=str, default="adult_model_adjusted", help="Checkpoint name for adjusted implementation.")
    parser.add_argument("--implementations", type=str, default="baseline,adjusted", help="Comma-separated: baseline,adjusted")
    parser.add_argument("--stages", type=str, default="train,precompute,inject,drift_eval,ranking", help="Comma-separated stages or 'full'.")
    parser.add_argument("--subset-spec", type=str, default="very_small:200:200,small:1000:1000,medium:5000:5000,large:10000:10000",
                        help="Comma-separated subset definitions name:train_rows:test_rows")
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repeated runs per implementation/subset/stage.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset creation.")
    parser.add_argument("--minsup", type=float, default=0.05, help="Min support for precompute stage.")
    parser.add_argument("--n-targets", type=int, default=10, help="n-targets for inject/drift-eval stages.")
    parser.add_argument("--noise-pos", type=float, default=0.50, help="Positive drift noise level.")
    parser.add_argument("--noise-neg", type=float, default=0.00, help="Negative/no-drift noise level.")
    parser.add_argument("--run-negative", action="store_true", help="Also run negative/no-drift inject + drift-eval.")
    parser.add_argument("--output-dir", type=str, default="benchmark-results", help="Output directory for benchmark artifacts.")
    parser.add_argument("--plots", action="store_true", help="Generate plots if matplotlib is available.")
    parser.add_argument("--cleanup-before-run", action="store_true", help="Delete existing checkpoint artifacts before each repetition.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    python_exe = Path(args.python_exe).resolve()
    output_dir = Path(args.output_dir).resolve()

    safe_mkdir(output_dir)
    logs_dir = output_dir / "logs"
    safe_mkdir(logs_dir)

    implementations = [x.strip() for x in args.implementations.split(",") if x.strip()]
    requested_stages = [x.strip() for x in args.stages.split(",") if x.strip()]

    # If full requested, expand
    if requested_stages == ["full"]:
        requested_stages = ["train", "precompute", "inject", "drift_eval", "ranking"]

    subset_defs = []
    for token in args.subset_spec.split(","):
        name, tr, te = token.split(":")
        subset_defs.append((name, int(tr), int(te)))

    data_dir = project_root / "data"
    source_data_dir = data_dir  # use current project data as source full dataset
    subsets_root = output_dir / "subsets"
    safe_mkdir(subsets_root)

    # Build subset files once
    subset_paths: Dict[str, Path] = {}
    for idx, (subset_name, tr, te) in enumerate(subset_defs):
        subset_dir = subsets_root / subset_name
        make_adult_subset(
            source_data_dir=source_data_dir,
            subset_dir=subset_dir,
            train_rows=tr,
            test_rows=te,
            seed=args.seed + idx,
        )
        subset_paths[subset_name] = subset_dir

    # Implementation config
    impl_to_checkpoint = {
        "baseline": args.baseline_checkpoint,
        "adjusted": args.adjusted_checkpoint,
    }

    all_results: List[StageResult] = []
    manifest_rows: List[Dict[str, Any]] = []

    for implementation in implementations:
        checkpoint = impl_to_checkpoint[implementation]

        for subset_name, tr, te in subset_defs:
            subset_dir = subset_paths[subset_name]

            for repetition in range(1, args.repetitions + 1):
                # Ensure both implementations run on identical subset data
                copy_subset_into_project_data(subset_dir, data_dir)

                if args.cleanup_before_run:
                    remove_checkpoint_artifacts(project_root, checkpoint)

                # Positive pipeline
                stage_sequence = requested_stages[:]
                for stage in stage_sequence:
                    noise_for_stage = args.noise_pos if stage in {"inject", "drift_eval"} else args.noise_pos

                    result = benchmark_one_stage(
                        project_root=project_root,
                        python_exe=python_exe,
                        implementation=implementation,
                        checkpoint=checkpoint,
                        subset_name=subset_name,
                        subset_rows_train=tr,
                        subset_rows_test=te,
                        repetition=repetition,
                        stage=stage,
                        noise=noise_for_stage,
                        n_targets=args.n_targets,
                        minsup=args.minsup,
                        logs_dir=logs_dir,
                    )
                    all_results.append(result)
                    manifest_rows.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "implementation": implementation,
                        "subset_name": subset_name,
                        "subset_rows_train": tr,
                        "subset_rows_test": te,
                        "repetition": repetition,
                        "stage": stage,
                        "checkpoint": checkpoint,
                        "command": result.command,
                        "returncode": result.returncode,
                        "runtime_sec": result.runtime_sec,
                        "peak_memory_mb": result.peak_memory_mb,
                        "artifact_paths": result.artifact_paths,
                        "success": result.success,
                        "error": result.error,
                    })

                # Optional negative/no-drift evaluation
                if args.run_negative:
                    for stage in requested_stages:
                        if stage not in {"inject", "drift_eval"}:
                            continue
                        result = benchmark_one_stage(
                            project_root=project_root,
                            python_exe=python_exe,
                            implementation=f"{implementation}_neg",
                            checkpoint=checkpoint,
                            subset_name=subset_name,
                            subset_rows_train=tr,
                            subset_rows_test=te,
                            repetition=repetition,
                            stage=stage,
                            noise=args.noise_neg,
                            n_targets=args.n_targets,
                            minsup=args.minsup,
                            logs_dir=logs_dir,
                        )
                        all_results.append(result)
                        manifest_rows.append({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "implementation": f"{implementation}_neg",
                            "subset_name": subset_name,
                            "subset_rows_train": tr,
                            "subset_rows_test": te,
                            "repetition": repetition,
                            "stage": stage,
                            "checkpoint": checkpoint,
                            "command": result.command,
                            "returncode": result.returncode,
                            "runtime_sec": result.runtime_sec,
                            "peak_memory_mb": result.peak_memory_mb,
                            "artifact_paths": result.artifact_paths,
                            "success": result.success,
                            "error": result.error,
                        })

    # Save manifest

    # Save flat results
    results_df = flatten_results(all_results)
    results_csv = output_dir / "benchmark_results.csv"
    results_df.to_csv(results_csv, index=False)

    # Summary tables
    success_df = results_df[results_df["success"] == True].copy()
    summary_tables = summarize_results(success_df)

    for name, table in summary_tables.items():
        print_table(name, table)
        table.to_csv(output_dir / f"{name}.csv", index=False)

    scaling_df = empirical_scaling_table(success_df)
    print_table("empirical_scaling", scaling_df)
    scaling_df.to_csv(output_dir / "empirical_scaling.csv", index=False)

    # Optional ranking summary table
    ranking_cols = [c for c in success_df.columns if c.startswith("ranking_")]
    if ranking_cols:
        ranking_summary = (
            success_df.groupby(["implementation", "subset_name"])[ranking_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        print_table("ranking_quality_by_subset", ranking_summary)
        ranking_summary.to_csv(output_dir / "ranking_quality_by_subset.csv", index=False)

    # Optional plots
    if args.plots and plt is not None:
        plot_df = (
            success_df.groupby(["implementation", "subset_name", "subset_rows_train"], as_index=False)
            .agg(
                runtime_sec=("runtime_sec", "mean"),
                peak_memory_mb=("peak_memory_mb", "mean"),
                subgroup_count=("subgroup_count", "mean"),
                artifact_mb=("artifact_bytes", lambda x: np.mean(x) / (1024 * 1024)),
            )
        )

        save_plot(
            plot_df, "subset_rows_train", "runtime_sec", "implementation",
            output_dir / "runtime_vs_subset.png",
            "Runtime vs subset size", "Training rows", "Runtime (sec)"
        )
        save_plot(
            plot_df, "subset_rows_train", "peak_memory_mb", "implementation",
            output_dir / "memory_vs_subset.png",
            "Peak memory vs subset size", "Training rows", "Peak memory (MB)"
        )
        save_plot(
            plot_df, "subset_rows_train", "subgroup_count", "implementation",
            output_dir / "subgroups_vs_subset.png",
            "Subgroup count vs subset size", "Training rows", "Subgroup count"
        )
        save_plot(
            plot_df, "subset_rows_train", "artifact_mb", "implementation",
            output_dir / "artifact_size_vs_subset.png",
            "Artifact size vs subset size", "Training rows", "Artifact size (MB)"
        )

    print()
    print(f"Saved results to: {output_dir}")
    print(f"CSV: {results_csv}")


if __name__ == "__main__":
    main()



"""
Run the full baseline and adjusted comparison on a few subset sizes:

python benchmark_adult.py `
  --project-root . `
  --python-exe .\.venv\Scripts\python.exe `
  --baseline-checkpoint adult_model `
  --adjusted-checkpoint adult_model_adjusted `
  --implementations baseline,adjusted `
  --stages full `
  --subset-spec very_small:200:200,small:1000:1000,medium:5000:5000 `
  --repetitions 2 `
  --minsup 0.05 `
  --n-targets 10 `
  --plots `
  --cleanup-before-run

"""

"""
Run only training and precompute:
python benchmark_adult.py `
  --project-root . `
  --python-exe .\.venv\Scripts\python.exe `
  --baseline-checkpoint adult_model `
  --adjusted-checkpoint adult_model_adjusted `
  --stages train,precompute `
  --subset-spec very_small:200:200,small:1000:1000,medium:5000:5000 `
  --repetitions 3
"""

"""Run positive and negative inject/drift-eval comparisons:
python benchmark_adult.py `
  --project-root . `
  --python-exe .\.venv\Scripts\python.exe `
  --baseline-checkpoint adult_model `
  --adjusted-checkpoint adult_model_adjusted `
  --stages inject,drift_eval `
  --subset-spec small:1000:1000,medium:5000:5000 `
  --repetitions 1 `
  --n-targets 10 `
  --run-negative
"""