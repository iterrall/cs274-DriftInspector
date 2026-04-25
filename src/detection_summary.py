#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from functools import reduce

@dataclass
class DetectionExperimentRecord:
    implementation: str
    checkpoint: str
    support_low: Optional[float]
    support_high: Optional[float]
    source_file: str
    gt: int
    pred: int
    threshold: float
    win_size: int
    policy: str
    score_used: float
    first_detect_window: int
    n_detect_windows: int
    n_windows_tested: int


def safe_pickle_load(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None


def normalize_itemset(x) -> Tuple[int, ...]:
    try:
        return tuple(sorted(int(v) for v in x))
    except Exception:
        return tuple()


def parse_support_and_noise_from_name(name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    support_match = re.search(r"support-([0-9.]+)-([0-9.]+)", name)
    noise_match = re.search(r"noise-([0-9.]+)", name)
    support_low = float(support_match.group(1)) if support_match else None
    support_high = float(support_match.group(2)) if support_match else None
    noise = float(noise_match.group(1)) if noise_match else None
    return support_low, support_high, noise

def build_global_subgroup_index(matches_obj: dict) -> pd.Index:
    fi_df = matches_obj["matches_train"].fi.reset_index(drop=True)
    subgroups = [normalize_itemset(x) for x in fi_df["itemsets"]]
    return pd.Index(subgroups, dtype=object)

def get_metric_values(df: pd.DataFrame, metric: str = "accuracy") -> np.ndarray:
    lower_map = {c.lower(): c for c in df.columns}

    if metric.lower() in lower_map:
        return df[lower_map[metric.lower()]].to_numpy(dtype=float)

    if metric.lower() == "accuracy":
        needed = ["tp", "tn", "fp", "fn"]
        if all(col in lower_map for col in needed):
            tp = df[lower_map["tp"]].to_numpy(dtype=float)
            tn = df[lower_map["tn"]].to_numpy(dtype=float)
            fp = df[lower_map["fp"]].to_numpy(dtype=float)
            fn = df[lower_map["fn"]].to_numpy(dtype=float)

            denom = tp + tn + fp + fn
            return np.divide(tp + tn, denom, out=np.zeros_like(denom), where=denom > 0)

    raise KeyError(f"Could not compute metric '{metric}' from columns: {list(df.columns)}")


def _empty_aligned_counts(global_index: pd.Index) -> dict[str, pd.Series]:
    return {
        "tp": pd.Series(np.nan, index=global_index, dtype=float),
        "tn": pd.Series(np.nan, index=global_index, dtype=float),
        "fp": pd.Series(np.nan, index=global_index, dtype=float),
        "fn": pd.Series(np.nan, index=global_index, dtype=float),
    }


def div_df_to_aligned_counts(df: pd.DataFrame, global_index: pd.Index) -> dict[str, pd.Series]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return _empty_aligned_counts(global_index)

    subgroup_col = None
    if "subgroup" in df.columns:
        subgroup_col = "subgroup"
    elif "itemsets" in df.columns:
        subgroup_col = "itemsets"

    if subgroup_col is None:
        raise KeyError(f"Expected 'subgroup' or 'itemsets' in columns, got {list(df.columns)}")

    lower_map = {c.lower(): c for c in df.columns}
    needed = ["tp", "tn", "fp", "fn"]
    if not all(k in lower_map for k in needed):
        raise KeyError(f"Need tp/tn/fp/fn columns, got {list(df.columns)}")

    work = df.copy()
    work["__sg__"] = work[subgroup_col].apply(normalize_itemset)

    grouped = (
        work.groupby("__sg__", dropna=False)[
            [lower_map["tp"], lower_map["tn"], lower_map["fp"], lower_map["fn"]]
        ]
        .sum()
    )

    grouped = grouped.rename(
        columns={
            lower_map["tp"]: "tp",
            lower_map["tn"]: "tn",
            lower_map["fp"]: "fp",
            lower_map["fn"]: "fn",
        }
    )

    grouped = grouped.reindex(global_index)

    return {
        "tp": grouped["tp"].astype(float),
        "tn": grouped["tn"].astype(float),
        "fp": grouped["fp"].astype(float),
        "fn": grouped["fn"].astype(float),
    }

def build_aligned_divs(divs_list, global_index: pd.Index) -> list[dict[str, pd.Series]]:
    return [div_df_to_aligned_counts(df, global_index) for df in divs_list]

def detect_singlebatch_exact(divs, metric, ref_win, curr_win):
    start_ref_win, ref_win_size = ref_win
    start_curr_win, curr_win_size = curr_win

    eps = 1e-12

    if metric == "accuracy":
        a1 = reduce(
            lambda a, b: a + b,
            [div["tp"] + div["tn"] for div in divs[start_ref_win:start_ref_win + ref_win_size]],
        )
        b1 = reduce(
            lambda a, b: a + b,
            [div["fp"] + div["fn"] for div in divs[start_ref_win:start_ref_win + ref_win_size]],
        )
    elif metric == "f1":
        a1 = reduce(
            lambda a, b: a + b,
            [2.0 * div["tp"] for div in divs[start_ref_win:start_ref_win + ref_win_size]],
        )
        b1 = reduce(
            lambda a, b: a + b,
            [div["fp"] + div["fn"] for div in divs[start_ref_win:start_ref_win + ref_win_size]],
        )
    else:
        raise ValueError(f"Unsupported metric for exact paper detector: {metric}")

    E1 = a1 / (a1 + b1)
    Var1 = (a1 * b1) / (((a1 + b1) ** 2) * (a1 + b1 + 1))

    if metric == "accuracy":
        a2 = reduce(
            lambda a, b: a + b,
            [div["tp"] + div["tn"] for div in divs[start_curr_win:start_curr_win + curr_win_size]],
        )
        b2 = reduce(
            lambda a, b: a + b,
            [div["fp"] + div["fn"] for div in divs[start_curr_win:start_curr_win + curr_win_size]],
        )
    elif metric == "f1":
        a2 = reduce(
            lambda a, b: a + b,
            [2.0 * div["tp"] for div in divs[start_curr_win:start_curr_win + curr_win_size]],
        )
        b2 = reduce(
            lambda a, b: a + b,
            [div["fp"] + div["fn"] for div in divs[start_curr_win:start_curr_win + curr_win_size]],
        )

    E2 = a2 / (a2 + b2)
    Var2 = (a2 * b2) / (((a2 + b2) ** 2) * (a2 + b2 + 1))

    delta = E2 - E1
    variance = Var1 + Var2
    t_stat = np.abs(delta) / np.sqrt(variance + eps)

    return delta, t_stat


def compute_window_scores_exact(divs, metric: str, win_size: int) -> np.ndarray:
    n_batches = len(divs)
    if n_batches < 2 * win_size:
        return np.array([], dtype=float)

    scores = []

    for end in range(2 * win_size - 1, n_batches):
        ref_win = (end - 2 * win_size + 1, win_size)
        curr_win = (end - win_size + 1, win_size)

        _, t_stat = detect_singlebatch_exact(divs, metric, ref_win, curr_win)

        if isinstance(t_stat, pd.Series):
            vals = t_stat.to_numpy(dtype=float)
        else:
            vals = np.asarray(t_stat, dtype=float)

        finite = np.isfinite(vals)
        score = float(np.max(vals[finite])) if np.any(finite) else 0.0
        scores.append(score)

    return np.asarray(scores, dtype=float)

def compute_detection_record_for_file(
    *,
    implementation: str,
    checkpoint: str,
    source_file: str,
    support_low: Optional[float],
    support_high: Optional[float],
    noise: Optional[float],
    aligned_divs,
    metric: str,
    threshold: float,
    win_size: int,
    policy: str,
) -> Optional[DetectionExperimentRecord]:
    gt = int(noise is not None and noise > 0.0)

    window_scores = compute_window_scores_exact(
        divs=aligned_divs,
        metric=metric,
        win_size=win_size,
    )

    if window_scores.size == 0:
        return None

    if policy == "final_t":
        score_used = float(window_scores[-1])
        pred = int(score_used >= threshold)
        detect_indices = np.array([len(window_scores) - 1], dtype=int) if pred else np.array([], dtype=int)
    else:
        # detect_singlebatch already returns abs(delta)/sqrt(var), so max_t and max_abs_t are equivalent
        detect_mask = window_scores >= threshold
        pred = int(np.any(detect_mask))
        score_used = float(np.max(window_scores))
        detect_indices = np.flatnonzero(detect_mask)

    first_detect_window = int(detect_indices[0]) if detect_indices.size else -1

    return DetectionExperimentRecord(
        implementation=implementation,
        checkpoint=checkpoint,
        support_low=support_low,
        support_high=support_high,
        source_file=source_file,
        gt=gt,
        pred=pred,
        threshold=float(threshold),
        win_size=int(win_size),
        policy=policy,
        score_used=score_used,
        first_detect_window=first_detect_window,
        n_detect_windows=int(detect_indices.size),
        n_windows_tested=int(window_scores.size),
    )


def group_files_by_support(files: List[Path]) -> Dict[Tuple[Optional[float], Optional[float]], List[Path]]:
    out: Dict[Tuple[Optional[float], Optional[float]], List[Path]] = {}
    for path in files:
        support_low, support_high, _ = parse_support_and_noise_from_name(path.name)
        key = (support_low, support_high)
        out.setdefault(key, []).append(path)
    return out


def sample_balanced_detection_files(
    models_ckpt: Path,
    checkpoint: str,
    max_per_support: int,
    seed: int,
) -> List[Path]:
    rng = np.random.default_rng(seed)
    supwise_dir = models_ckpt / "sup-wise"

    pos_patterns = [
        f"{checkpoint}-noise-0.50-support-*-target-*.pkl",
        f"{checkpoint}-mode-*-noise-0.50-support-*-target-*.pkl",
    ]
    neg_patterns = [
        f"{checkpoint}-noise-0.00-support-*-target-*.pkl",
        f"{checkpoint}-mode-*-noise-0.00-support-*-target-*.pkl",
    ]

    pos_files: List[Path] = []
    for pat in pos_patterns:
        pos_files.extend(supwise_dir.glob(pat))
    pos_files = sorted(set(pos_files), key=lambda p: str(p))

    neg_files: List[Path] = []
    for pat in neg_patterns:
        neg_files.extend(supwise_dir.glob(pat))
    neg_files = sorted(set(neg_files), key=lambda p: str(p))

    pos_by_sup = group_files_by_support(pos_files)
    neg_by_sup = group_files_by_support(neg_files)

    sampled: List[Path] = []
    all_supports = sorted(set(pos_by_sup) | set(neg_by_sup))

    for support_key in all_supports:
        pos_bucket = pos_by_sup.get(support_key, [])
        neg_bucket = neg_by_sup.get(support_key, [])

        if not pos_bucket or not neg_bucket:
            continue

        n = min(max_per_support, len(pos_bucket), len(neg_bucket))
        if n <= 0:
            continue

        pos_idx = rng.choice(len(pos_bucket), size=n, replace=False)
        neg_idx = rng.choice(len(neg_bucket), size=n, replace=False)

        sampled.extend(pos_bucket[i] for i in pos_idx)
        sampled.extend(neg_bucket[i] for i in neg_idx)

    return sorted(sampled, key=lambda p: str(p))

def append_rows_to_csv(csv_path: Path, rows: List[DetectionExperimentRecord], light_output: bool = False):
    if not rows:
        return

    dict_rows = [asdict(r) for r in rows]
    if light_output:
        keep_cols = [
            "implementation",
            "checkpoint",
            "support_low",
            "support_high",
            "source_file",
            "gt",
            "pred",
            "threshold",
            "win_size",
            "policy",
            "score_used",
        ]
        dict_rows = [{k: row[k] for k in keep_cols} for row in dict_rows]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(dict_rows)


def load_processed_keys(raw_csv_path: Path) -> Set[Tuple[str, str, str, float, int, str]]:
    if not raw_csv_path.exists():
        return set()

    try:
        df = pd.read_csv(raw_csv_path)
    except Exception:
        return set()

    needed = {"implementation", "checkpoint", "source_file", "threshold", "win_size", "policy"}
    if not needed.issubset(df.columns):
        return set()

    threshold_vals = pd.to_numeric(df["threshold"], errors="coerce").fillna(-1.0).astype(float)
    win_size_vals = pd.to_numeric(df["win_size"], errors="coerce").fillna(-1).astype(int)

    return set(
        zip(
            df["implementation"].astype(str),
            df["checkpoint"].astype(str),
            df["source_file"].astype(str),
            threshold_vals,
            win_size_vals,
            df["policy"].astype(str),
        )
    )


def summarize_detection(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        tp = int(((g["gt"] == 1) & (g["pred"] == 1)).sum())
        fp = int(((g["gt"] == 0) & (g["pred"] == 1)).sum())
        tn = int(((g["gt"] == 0) & (g["pred"] == 0)).sum())
        fn = int(((g["gt"] == 1) & (g["pred"] == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        accuracy = (tp + tn) / len(g) if len(g) > 0 else 0.0

        row = dict(zip(group_cols, keys))
        row.update(
            {
                "Accuracy": accuracy,
                "F1": f1,
                "FPR": fpr,
                "FNR": fnr,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "N": int(len(g)),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def save_partial_summaries(raw_csv_path: Path, output_dir: Path):
    if not raw_csv_path.exists():
        return

    raw_df = pd.read_csv(raw_csv_path)
    if raw_df.empty:
        return

    overall = summarize_detection(raw_df, ["implementation"])
    by_support = summarize_detection(raw_df, ["implementation", "support_low", "support_high"])

    overall.to_csv(output_dir / "detection_summary_overall.csv", index=False)
    by_support.to_csv(output_dir / "detection_summary_by_support.csv", index=False)


def log_progress(txt_path: Path, msg: str):
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    parser = argparse.ArgumentParser(description="Paper-aligned drift detection summary.")
    parser.add_argument("--models-ckpt", type=Path, default=Path("models-ckpt"))
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--adaptive-checkpoint", required=True)
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--win-size", type=int, default=5)
    parser.add_argument("--policy", choices=["max_t", "max_abs_t", "final_t"], default="max_t")
    parser.add_argument("--max-per-support", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("report-metrics"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--light-output", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = args.output_dir / "detection_raw_records.csv"
    progress_txt = args.output_dir / "detection_progress.txt"

    processed_keys = load_processed_keys(raw_csv_path) if args.resume else set()
    file_counter = 0

    implementations = [
        ("baseline", args.baseline_checkpoint),
        ("adaptive", args.adaptive_checkpoint),
    ]

    for implementation, checkpoint in tqdm(implementations, desc="Implementations"):
        matches_path = args.models_ckpt / f"matches-{checkpoint}.pkl"
        matches_obj = safe_pickle_load(matches_path)
        if matches_obj is None:
            msg = f"[WARN] Missing matches file for {checkpoint}"
            print(msg)
            log_progress(progress_txt, msg)
            continue

        global_index = build_global_subgroup_index(matches_obj)


        files = sample_balanced_detection_files(
            models_ckpt=args.models_ckpt,
            checkpoint=checkpoint,
            max_per_support=args.max_per_support,
            seed=args.seed,
        )

        if args.max_files is not None:
            files = files[: args.max_files]

        new_rows: List[DetectionExperimentRecord] = []

        for path in tqdm(files, desc=f"{implementation}:{checkpoint}", leave=False):
            key = (
                implementation,
                checkpoint,
                str(path),
                float(args.threshold),
                int(args.win_size),
                args.policy,
            )
            if key in processed_keys:
                continue

            payload = safe_pickle_load(path)
            if payload is None:
                continue

            support_low, support_high, noise = parse_support_and_noise_from_name(path.name)
            divs = payload.get("divs", [])
            if not divs:
                continue

            try:
                aligned_divs = build_aligned_divs(divs, global_index)
            except Exception as e:
                print(f"[WARN] Failed to build metric matrix for {path}: {e}")
                continue

            row = compute_detection_record_for_file(
                implementation=implementation,
                checkpoint=checkpoint,
                source_file=str(path),
                support_low=support_low,
                support_high=support_high,
                noise=noise,
                aligned_divs=aligned_divs,
                metric=args.metric,
                threshold=args.threshold,
                win_size=args.win_size,
                policy=args.policy,
            )
            if row is None:
                continue

            new_rows.append(row)
            processed_keys.add(key)

            file_counter += 1
            if file_counter % args.save_every == 0:
                append_rows_to_csv(raw_csv_path, new_rows, light_output=args.light_output)
                new_rows = []
                save_partial_summaries(raw_csv_path, args.output_dir)
                log_progress(progress_txt, f"Processed {file_counter} files")

        if new_rows:
            append_rows_to_csv(raw_csv_path, new_rows, light_output=args.light_output)

    save_partial_summaries(raw_csv_path, args.output_dir)
    log_progress(progress_txt, "Completed detection summary run.")

    overall_path = args.output_dir / "detection_summary_overall.csv"
    by_support_path = args.output_dir / "detection_summary_by_support.csv"

    if overall_path.exists():
        print("\n*** Detection Summary Overall ***")
        print(pd.read_csv(overall_path).to_string(index=False))
    else:
        raise SystemExit("No detection records produced.")

    if by_support_path.exists():
        print("\n*** Detection Summary By Support ***")
        print(pd.read_csv(by_support_path).to_string(index=False))


if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# from __future__ import annotations
#
# import argparse
# import csv
# import re
# import pickle
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import Dict, List, Optional, Set, Tuple
#
# import numpy as np
# import pandas as pd
# from tqdm.auto import tqdm
#
# import warnings
# warnings.filterwarnings("ignore", message="Mean of empty slice") # Does this warning indicate any issues with the run or reading of the file?
# warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
#
#
# @dataclass
# class DetectionRecord:
#     implementation: str
#     checkpoint: str
#     support_low: Optional[float]
#     support_high: Optional[float]
#     source_file: str
#     window_end: int
#     gt: int
#     pred: int
#     threshold: float
#     win_size: int
#
# def safe_pickle_load(path: Path):
#     try:
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     except Exception as e:
#         print(f"[WARN] Could not load {path}: {e}")
#         return None
#
#
# def normalize_itemset(x) -> Tuple[int, ...]:
#     try:
#         return tuple(sorted(int(v) for v in x))
#     except Exception:
#         return tuple()
#
#
# def parse_support_and_noise_from_name(name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
#     support_match = re.search(r"support-([0-9.]+)-([0-9.]+)", name)
#     noise_match = re.search(r"noise-([0-9.]+)", name)
#     support_low = float(support_match.group(1)) if support_match else None
#     support_high = float(support_match.group(2)) if support_match else None
#     noise = float(noise_match.group(1)) if noise_match else None
#     return support_low, support_high, noise
#
#
# def build_fi_maps(matches_obj: dict):
#     fi_df = matches_obj["matches_train"].fi.reset_index(drop=True)
#     subgroup_to_idx: Dict[Tuple[int, ...], int] = {}
#     for idx, row in fi_df.iterrows():
#         subgroup_to_idx[normalize_itemset(row["itemsets"])] = idx
#     return subgroup_to_idx, len(fi_df)
#
#
# def get_metric_values(df: pd.DataFrame, metric: str = "accuracy") -> np.ndarray:
#     lower_map = {c.lower(): c for c in df.columns}
#
#     if metric.lower() in lower_map:
#         return df[lower_map[metric.lower()]].to_numpy(dtype=float)
#
#     if metric.lower() == "accuracy":
#         needed = ["tp", "tn", "fp", "fn"]
#         if all(col in lower_map for col in needed):
#             tp = df[lower_map["tp"]].to_numpy(dtype=float)
#             tn = df[lower_map["tn"]].to_numpy(dtype=float)
#             fp = df[lower_map["fp"]].to_numpy(dtype=float)
#             fn = df[lower_map["fn"]].to_numpy(dtype=float)
#
#             denom = tp + tn + fp + fn
#             return np.divide(tp + tn, denom, out=np.zeros_like(denom), where=denom > 0)
#
#     raise KeyError(f"Could not compute metric '{metric}' from columns: {list(df.columns)}")
#
#
# def divs_to_metric_matrix(divs_list, subgroup_to_idx, n_groups, metric="accuracy") -> np.ndarray:
#     out = np.full((len(divs_list), n_groups), np.nan, dtype=float)
#
#     for b, df in enumerate(divs_list):
#         if not isinstance(df, pd.DataFrame) or df.empty:
#             continue
#
#         vals = get_metric_values(df, metric)
#
#         subgroup_col = None
#         if "subgroup" in df.columns:
#             subgroup_col = "subgroup"
#         elif "itemsets" in df.columns:
#             subgroup_col = "itemsets"
#
#         if subgroup_col is None:
#             k = min(len(vals), n_groups)
#             out[b, :k] = vals[:k]
#             continue
#
#         for row_idx, (_, row) in enumerate(df.iterrows()):
#             sg = normalize_itemset(row[subgroup_col])
#             if sg in subgroup_to_idx:
#                 out[b, subgroup_to_idx[sg]] = float(vals[row_idx])
#
#     return out
#
#
# def altered_masks_to_fraction_matrix(altered_list, matches_batches) -> np.ndarray:
#     if not altered_list or not matches_batches:
#         return np.empty((0, 0), dtype=float)
#
#     first_matches = getattr(matches_batches[0], "matches", None)
#     if first_matches is None:
#         return np.empty((0, 0), dtype=float)
#
#     n_batches = min(len(altered_list), len(matches_batches))
#     n_groups = first_matches.shape[1]
#     out = np.full((n_batches, n_groups), np.nan, dtype=float)
#
#     for b in range(n_batches):
#         mb = matches_batches[b]
#         M = getattr(mb, "matches", None)
#         if M is None:
#             continue
#
#         altered_mask = np.asarray(altered_list[b], dtype=bool).ravel()
#
#         if altered_mask.shape[0] != M.shape[0]:
#             m = min(altered_mask.shape[0], M.shape[0])
#             altered_mask = altered_mask[:m]
#             M = M[:m]
#
#         subgroup_sizes = np.asarray(M.sum(axis=0)).ravel().astype(float)
#         altered_counts = np.asarray(M[altered_mask].sum(axis=0)).ravel().astype(float)
#
#         frac = np.full_like(subgroup_sizes, np.nan, dtype=float)
#         valid = subgroup_sizes > 0
#         frac[valid] = altered_counts[valid] / subgroup_sizes[valid]
#         out[b] = frac
#
#     return out
#
# def compute_detection_records_for_file(
#     metric_mat: np.ndarray,
#     altered_frac_mat: np.ndarray,
#     implementation: str,
#     checkpoint: str,
#     support_low: Optional[float],
#     support_high: Optional[float],
#     source_file: str,
#     threshold: float,
#     win_size: int,
# ) -> List[DetectionRecord]:
#     if metric_mat.size == 0 or altered_frac_mat.size == 0:
#         return []
#
#     n_common = min(metric_mat.shape[0], altered_frac_mat.shape[0])
#     if n_common < 2 * win_size:
#         return []
#
#     metric_mat = metric_mat[:n_common]
#     altered_frac_mat = altered_frac_mat[:n_common]
#
#     records: List[DetectionRecord] = []
#
#     for end in range(2 * win_size - 1, n_common):
#         ref = metric_mat[end - 2 * win_size + 1 : end - win_size + 1, :]
#         cur = metric_mat[end - win_size + 1 : end + 1, :]
#         gt_cur = altered_frac_mat[end - win_size + 1 : end + 1, :]
#
#         with np.errstate(invalid="ignore", divide="ignore"):
#             ref_mean = np.nanmean(ref, axis=0)
#             cur_mean = np.nanmean(cur, axis=0)
#             delta = cur_mean - ref_mean
#
#             ref_std = np.nanstd(ref, axis=0, ddof=1)
#             cur_std = np.nanstd(cur, axis=0, ddof=1)
#             n_ref = np.sum(~np.isnan(ref), axis=0)
#             n_cur = np.sum(~np.isnan(cur), axis=0)
#
#             se = np.sqrt((ref_std ** 2) / np.maximum(n_ref, 1) + (cur_std ** 2) / np.maximum(n_cur, 1))
#
#         tstat = np.zeros_like(delta, dtype=float)
#         valid = np.isfinite(se) & (se > 0)
#         tstat[valid] = -delta[valid] / se[valid]
#         tstat = np.nan_to_num(tstat, nan=0.0, posinf=0.0, neginf=0.0)
#
#         gt_vec = np.nanmean(gt_cur, axis=0)
#         gt_vec = np.nan_to_num(gt_vec, nan=0.0, posinf=0.0, neginf=0.0)
#
#         pred_score = float(np.max(tstat)) if tstat.size else 0.0
#         gt_score = float(np.max(gt_vec)) if gt_vec.size else 0.0
#
#         pred = int(pred_score >= threshold)
#         gt = int(gt_score > 0.0)
#
#         records.append(
#             DetectionRecord(
#                 implementation=implementation,
#                 checkpoint=checkpoint,
#                 support_low=support_low,
#                 support_high=support_high,
#                 source_file=source_file,
#                 window_end=int(end),
#                 gt=gt,
#                 pred=pred,
#                 threshold=float(threshold),
#                 win_size=int(win_size),
#             )
#         )
#
#     return records
#
#
#
# def append_rows_to_csv(csv_path: Path, rows: List[DetectionRecord], light_output: bool = False):
#     if not rows:
#         return
#
#     dict_rows = [asdict(r) for r in rows]
#     if light_output:
#         keep_cols = [
#             "implementation",
#             "checkpoint",
#             "support_low",
#             "support_high",
#             "source_file",
#             "gt",
#             "pred",
#             "threshold",
#             "win_size",
#         ]
#         dict_rows = [{k: row[k] for k in keep_cols} for row in dict_rows]
#
#     write_header = not csv_path.exists()
#     with open(csv_path, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
#         if write_header:
#             writer.writeheader()
#         writer.writerows(dict_rows)
# def load_processed_files(raw_csv_path: Path) -> Set[Tuple[str, str, str, float, int]]:
#     if not raw_csv_path.exists():
#         return set()
#
#     try:
#         df = pd.read_csv(raw_csv_path)
#     except Exception:
#         return set()
#
#     needed = {"implementation", "checkpoint", "source_file", "threshold", "win_size"}
#     if not needed.issubset(df.columns):
#         return set()
#
#     threshold_vals = pd.to_numeric(df["threshold"], errors="coerce").fillna(-1.0)
#     win_size_vals = pd.to_numeric(df["win_size"], errors="coerce").fillna(-1).astype(int)
#
#     return set(
#         zip(
#             df["implementation"].astype(str),
#             df["checkpoint"].astype(str),
#             df["source_file"].astype(str),
#             threshold_vals.astype(float),
#             win_size_vals,
#         )
#     )
#
# def summarize_detection(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
#     rows = []
#     for keys, g in df.groupby(group_cols, dropna=False):
#         if not isinstance(keys, tuple):
#             keys = (keys,)
#
#         tp = int(((g["gt"] == 1) & (g["pred"] == 1)).sum())
#         fp = int(((g["gt"] == 0) & (g["pred"] == 1)).sum())
#         tn = int(((g["gt"] == 0) & (g["pred"] == 0)).sum())
#         fn = int(((g["gt"] == 1) & (g["pred"] == 0)).sum())
#
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
#         fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
#         fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
#         accuracy = (tp + tn) / len(g) if len(g) > 0 else 0.0
#
#         row = dict(zip(group_cols, keys))
#         row.update({
#             "Accuracy": accuracy,
#             "F1": f1,
#             "FPR": fpr,
#             "FNR": fnr,
#             "TP": tp,
#             "FP": fp,
#             "TN": tn,
#             "FN": fn,
#             "N": int(len(g)),
#         })
#         rows.append(row)
#
#     return pd.DataFrame(rows)
#
#
# def save_partial_summaries(raw_csv_path: Path, output_dir: Path):
#     if not raw_csv_path.exists():
#         return
#     raw_df = pd.read_csv(raw_csv_path)
#     if raw_df.empty:
#         return
#
#     overall = summarize_detection(raw_df, ["implementation"])
#     by_support = summarize_detection(raw_df, ["implementation", "support_low", "support_high"])
#
#     overall.to_csv(output_dir / "detection_summary_overall.csv", index=False)
#     by_support.to_csv(output_dir / "detection_summary_by_support.csv", index=False)
#
#
# def log_progress(txt_path: Path, msg: str):
#     with open(txt_path, "a", encoding="utf-8") as f:
#         f.write(msg + "\n")
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Summarize drift detection metrics.")
#     parser.add_argument("--models-ckpt", type=Path, default=Path("models-ckpt"))
#     parser.add_argument("--baseline-checkpoint", required=True)
#     parser.add_argument("--adaptive-checkpoint", required=True)
#     parser.add_argument("--metric", default="accuracy")
#     parser.add_argument("--threshold", type=float, default=5.0)
#     parser.add_argument("--win-size", type=int, default=5)
#     parser.add_argument("--output-dir", type=Path, default=Path("report-metrics"))
#     parser.add_argument("--resume", action="store_true")
#     parser.add_argument("--save-every", type=int, default=25)
#     parser.add_argument("--max-files", type=int, default=None)
#     parser.add_argument("--light-output", action="store_true")
#     args = parser.parse_args()
#
#     args.output_dir.mkdir(parents=True, exist_ok=True)
#     raw_csv_path = args.output_dir / "detection_raw_records.csv"
#     progress_txt = args.output_dir / "detection_progress.txt"
#
#     processed_files = load_processed_files(raw_csv_path) if args.resume else set()
#     file_counter = 0
#
#     for implementation, checkpoint in tqdm(
#         [("baseline", args.baseline_checkpoint), ("adaptive", args.adaptive_checkpoint)],
#         desc="Implementations"
#     ):
#         matches_path = args.models_ckpt / f"matches-{checkpoint}.pkl"
#         matches_obj = safe_pickle_load(matches_path)
#         if matches_obj is None:
#             msg = f"[WARN] Missing matches file for {checkpoint}"
#             print(msg)
#             log_progress(progress_txt, msg)
#             continue
#
#         subgroup_to_idx, n_groups = build_fi_maps(matches_obj)
#
#         matches_batches = matches_obj.get("matches_batches")
#         if not matches_batches:
#             msg = f"[WARN] Missing or empty matches_batches for {checkpoint}"
#             print(msg)
#             log_progress(progress_txt, msg)
#             continue
#
#         file_specs = [
#             ("pos", sorted((args.models_ckpt / "sup-wise").glob(f"{checkpoint}-noise-0.50-support-*-target-*.pkl"))),
#             ("neg", sorted((args.models_ckpt / "sup-wise").glob(f"{checkpoint}-noise-0.00-support-*-target-*.pkl"))),
#         ]
#
#         for label, files in file_specs:
#             if args.max_files is not None:
#                 files = files[: args.max_files]
#
#             for path in tqdm(files, desc=f"{implementation}:{label}", leave=False):
#                 key = (
#                     implementation,
#                     checkpoint,
#                     str(path),
#                     float(args.threshold),
#                     int(args.win_size),
#                 )
#                 if key in processed_files:
#                     continue
#
#                 payload = safe_pickle_load(path)
#                 if payload is None:
#                     continue
#
#                 support_low, support_high, noise = parse_support_and_noise_from_name(path.name)
#                 divs = payload.get("divs", [])
#                 altered = payload.get("altered", [])
#
#                 if not divs or not altered:
#                     continue
#
#                 metric_mat = divs_to_metric_matrix(divs, subgroup_to_idx, n_groups, metric=args.metric)
#                 altered_frac_mat = altered_masks_to_fraction_matrix(altered, matches_batches)
#
#                 rows = compute_detection_records_for_file(
#                     metric_mat=metric_mat,
#                     altered_frac_mat=altered_frac_mat,
#                     implementation=implementation,
#                     checkpoint=checkpoint,
#                     support_low=support_low,
#                     support_high=support_high,
#                     source_file=str(path),
#                     threshold=args.threshold,
#                     win_size=args.win_size,
#                 )
#
#                 append_rows_to_csv(raw_csv_path, rows, light_output=args.light_output)
#                 processed_files.add(key)
#
#                 file_counter += 1
#                 if file_counter % args.save_every == 0:
#                     save_partial_summaries(raw_csv_path, args.output_dir)
#                     log_progress(progress_txt, f"Processed {file_counter} files")
#
#     save_partial_summaries(raw_csv_path, args.output_dir)
#     log_progress(progress_txt, "Completed detection summary run.")
#
#     overall_path = args.output_dir / "detection_summary_overall.csv"
#     by_support_path = args.output_dir / "detection_summary_by_support.csv"
#
#     if overall_path.exists():
#         print("\n=== Detection Summary Overall ===")
#         print(pd.read_csv(overall_path).to_string(index=False))
#     if by_support_path.exists():
#         print("\n=== Detection Summary By Support ===")
#         print(pd.read_csv(by_support_path).to_string(index=False))
#
#
# if __name__ == "__main__":
#     main()
#
# # #!/usr/bin/env python3
# #
# # """
# # python detection_summary.py --baseline-checkpoint adult_model --adaptive-checkpoint adult_model_adaptive --threshold 5 --win-size 5
# # """
# # from __future__ import annotations
# #
# # import argparse
# # import math
# # import pickle
# # import re
# # from dataclasses import dataclass, asdict
# # from pathlib import Path
# # from typing import Dict, List, Optional, Tuple
# #
# # import numpy as np
# # import pandas as pd
# #
# #
# # @dataclass
# # class RunDetectionRecord:
# #     implementation: str
# #     checkpoint: str
# #     source_mode: str
# #     file: str
# #     support_low: Optional[float]
# #     support_high: Optional[float]
# #     noise: Optional[float]
# #     gt_label: int
# #     predicted_drift: int
# #     score_used: float
# #     threshold: float
# #     policy: str
# #     tp: int
# #     fp: int
# #     tn: int
# #     fn: int
# #
# #
# # def parse_support_and_noise_from_name(name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
# #     support_match = re.search(r"support-([0-9.]+)-([0-9.]+)", name)
# #     noise_match = re.search(r"noise-([0-9.]+)", name)
# #     support_low = float(support_match.group(1)) if support_match else None
# #     support_high = float(support_match.group(2)) if support_match else None
# #     noise = float(noise_match.group(1)) if noise_match else None
# #     return support_low, support_high, noise
# #
# #
# # def safe_pickle_load(path: Path):
# #     try:
# #         with open(path, "rb") as f:
# #             return pickle.load(f)
# #     except Exception as e:
# #         print(f"[WARN] Could not load pickle {path}: {e}")
# #         return None
# #
# #
# # def infer_metric_from_div(df: pd.DataFrame, metric: str = "accuracy") -> np.ndarray:
# #     lower_cols = {c.lower(): c for c in df.columns}
# #     if metric.lower() in lower_cols:
# #         return df[lower_cols[metric.lower()]].to_numpy(dtype=float)
# #
# #     if metric.lower() == "accuracy":
# #         required = ["tp", "tn", "fp", "fn"]
# #         if all(c in lower_cols for c in required):
# #             tp = df[lower_cols["tp"]].to_numpy(dtype=float)
# #             tn = df[lower_cols["tn"]].to_numpy(dtype=float)
# #             fp = df[lower_cols["fp"]].to_numpy(dtype=float)
# #             fn = df[lower_cols["fn"]].to_numpy(dtype=float)
# #             denom = tp + tn + fp + fn
# #             return np.divide(tp + tn, denom, out=np.zeros_like(denom), where=denom > 0)
# #
# #     raise KeyError(f"Could not infer metric '{metric}' from columns {list(df.columns)}")
# #
# #
# # def get_subgroup_series_from_supwise_payload(payload: dict, metric: str = "accuracy") -> np.ndarray:
# #     """
# #     For sup-wise files, return the target subgroup's per-batch metric trace.
# #
# #     Important for adaptive runs:
# #     if the target subgroup is no longer monitored in a batch, return NaN for
# #     that batch instead of incorrectly falling back to another subgroup row.
# #     """
# #     divs = payload.get("divs", [])
# #     if not divs:
# #         return np.array([], dtype=float)
# #
# #     subgroup = payload.get("subgroup")
# #     subgroup_tuple = tuple(sorted(int(x) for x in subgroup)) if subgroup is not None else None
# #
# #     values = []
# #     for df in divs:
# #         if not isinstance(df, pd.DataFrame) or df.empty:
# #             values.append(np.nan)
# #             continue
# #
# #         subgroup_col = None
# #         if "subgroup" in df.columns:
# #             subgroup_col = "subgroup"
# #         elif "itemsets" in df.columns:
# #             subgroup_col = "itemsets"
# #
# #         found = False
# #         if subgroup_col is not None and subgroup_tuple is not None:
# #             for _, row in df.iterrows():
# #                 try:
# #                     sg_t = tuple(sorted(int(x) for x in row[subgroup_col]))
# #                 except Exception:
# #                     sg_t = None
# #
# #                 if sg_t == subgroup_tuple:
# #                     row_df = pd.DataFrame([row])
# #                     values.append(float(infer_metric_from_div(row_df, metric)[0]))
# #                     found = True
# #                     break
# #
# #         if not found:
# #             # Adaptive run may have pruned the target subgroup from monitoring
# #             values.append(np.nan)
# #
# #     return np.asarray(values, dtype=float)
# #
# # def compute_window_score_from_series(
# #     series: np.ndarray,
# #     win_size: int,
# #     threshold_policy: str = "max_abs_t",
# # ) -> float:
# #     """
# #     Compute a single run-level detection score from a per-batch metric series.
# #
# #     We compare reference and current windows:
# #         delta = mean(curr) - mean(ref)
# #         t = -(delta / se)
# #     Positive t means degradation in current window.
# #
# #     policy:
# #       - max_abs_t: max absolute t over all valid windows
# #       - max_t: max positive t over windows
# #       - final_t: final window t
# #     """
# #     series = np.asarray(series, dtype=float)
# #     series = series[np.isfinite(series)]
# #     if len(series) < 2 * win_size:
# #         return 0.0
# #
# #     t_values = []
# #     for end in range(2 * win_size - 1, len(series)):
# #         ref = series[end - 2 * win_size + 1 : end - win_size + 1]
# #         cur = series[end - win_size + 1 : end + 1]
# #
# #         ref_mean = np.mean(ref)
# #         cur_mean = np.mean(cur)
# #         delta = cur_mean - ref_mean
# #
# #         ref_std = np.std(ref, ddof=1) if len(ref) > 1 else 0.0
# #         cur_std = np.std(cur, ddof=1) if len(cur) > 1 else 0.0
# #         se = math.sqrt((ref_std ** 2) / max(len(ref), 1) + (cur_std ** 2) / max(len(cur), 1))
# #         t = 0.0 if se == 0 else -(delta / se)
# #         t_values.append(float(t))
# #
# #     if not t_values:
# #         return 0.0
# #
# #     if threshold_policy == "max_abs_t":
# #         return max(abs(x) for x in t_values)
# #     if threshold_policy == "max_t":
# #         return max(t_values)
# #     if threshold_policy == "final_t":
# #         return t_values[-1]
# #     raise ValueError(f"Unknown threshold policy: {threshold_policy}")
# #
# #
# # def collect_supwise_records(
# #     implementation: str,
# #     checkpoint: str,
# #     supwise_dir: Path,
# #     win_size: int,
# #     threshold: float,
# #     policy: str,
# #     metric: str,
# # ) -> List[RunDetectionRecord]:
# #     pattern = f"{checkpoint}-noise-*-support-*-target-*.pkl"
# #     files = sorted(supwise_dir.glob(pattern))
# #     records: List[RunDetectionRecord] = []
# #
# #     for path in files:
# #         payload = safe_pickle_load(path)
# #         if payload is None:
# #             continue
# #
# #         support_low, support_high, noise = parse_support_and_noise_from_name(path.name)
# #         gt_label = 1 if (noise is not None and noise > 0) else 0
# #
# #         series = get_subgroup_series_from_supwise_payload(payload, metric=metric)
# #         score = compute_window_score_from_series(series, win_size=win_size, threshold_policy=policy)
# #         predicted = 1 if score >= threshold else 0
# #
# #         tp = int(gt_label == 1 and predicted == 1)
# #         fp = int(gt_label == 0 and predicted == 1)
# #         tn = int(gt_label == 0 and predicted == 0)
# #         fn = int(gt_label == 1 and predicted == 0)
# #
# #         records.append(
# #             RunDetectionRecord(
# #                 implementation=implementation,
# #                 checkpoint=checkpoint,
# #                 source_mode="supwise",
# #                 file=str(path),
# #                 support_low=support_low,
# #                 support_high=support_high,
# #                 noise=noise,
# #                 gt_label=gt_label,
# #                 predicted_drift=predicted,
# #                 score_used=score,
# #                 threshold=threshold,
# #                 policy=policy,
# #                 tp=tp,
# #                 fp=fp,
# #                 tn=tn,
# #                 fn=fn,
# #             )
# #         )
# #
# #     return records
# #
# #
# # def collect_results_v2_records(
# #     implementation: str,
# #     checkpoint: str,
# #     results_v2_path: Path,
# #     threshold: float,
# # ) -> List[RunDetectionRecord]:
# #     """
# #     If a *-results-v2.pkl exists, use it directly.
# #     Assumes items have attributes or dict keys:
# #       metric, gt, tstat, support
# #     We convert each result item into a binary prediction.
# #     """
# #     obj = safe_pickle_load(results_v2_path)
# #     if obj is None:
# #         return []
# #
# #     records: List[RunDetectionRecord] = []
# #     for item in obj:
# #         try:
# #             metric = getattr(item, "metric", None) if not isinstance(item, dict) else item.get("metric")
# #             if metric not in (None, "accuracy"):
# #                 continue
# #
# #             gt = getattr(item, "gt", None) if not isinstance(item, dict) else item.get("gt")
# #             tstat = getattr(item, "tstat", None) if not isinstance(item, dict) else item.get("tstat")
# #             support = getattr(item, "support", None) if not isinstance(item, dict) else item.get("support")
# #
# #             gt_label = 1 if gt == "pos" else 0
# #             score = float(np.nanmax(np.abs(np.asarray(tstat, dtype=float)))) if np.ndim(tstat) > 0 else float(abs(tstat))
# #             predicted = 1 if score >= threshold else 0
# #
# #             support_low = None
# #             support_high = None
# #             if isinstance(support, tuple) and len(support) == 2:
# #                 support_low, support_high = float(support[0]), float(support[1])
# #             elif isinstance(support, (list, np.ndarray)) and len(support) >= 2:
# #                 support_low, support_high = float(support[0]), float(support[1])
# #
# #             tp = int(gt_label == 1 and predicted == 1)
# #             fp = int(gt_label == 0 and predicted == 1)
# #             tn = int(gt_label == 0 and predicted == 0)
# #             fn = int(gt_label == 1 and predicted == 0)
# #
# #             records.append(
# #                 RunDetectionRecord(
# #                     implementation=implementation,
# #                     checkpoint=checkpoint,
# #                     source_mode="results_v2",
# #                     file=str(results_v2_path),
# #                     support_low=support_low,
# #                     support_high=support_high,
# #                     noise=None,
# #                     gt_label=gt_label,
# #                     predicted_drift=predicted,
# #                     score_used=score,
# #                     threshold=threshold,
# #                     policy="results_v2_abs_t",
# #                     tp=tp,
# #                     fp=fp,
# #                     tn=tn,
# #                     fn=fn,
# #                 )
# #             )
# #         except Exception as e:
# #             print(f"[WARN] Skipping results-v2 item due to parse error: {e}")
# #
# #     return records
# #
# #
# # def summarize_detection(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
# #     rows = []
# #     for keys, group in df.groupby(group_cols, dropna=False):
# #         if not isinstance(keys, tuple):
# #             keys = (keys,)
# #
# #         tp = int(group["tp"].sum())
# #         fp = int(group["fp"].sum())
# #         tn = int(group["tn"].sum())
# #         fn = int(group["fn"].sum())
# #
# #         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
# #         recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
# #         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
# #         fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
# #         fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
# #
# #         row = {col: val for col, val in zip(group_cols, keys)}
# #         row.update(
# #             {
# #                 "num_runs": len(group),
# #                 "tp": tp,
# #                 "fp": fp,
# #                 "tn": tn,
# #                 "fn": fn,
# #                 "F1": f1,
# #                 "FPR": fpr,
# #                 "FNR": fnr,
# #             }
# #         )
# #         rows.append(row)
# #
# #     return pd.DataFrame(rows)
# #
# #
# # def main():
# #     parser = argparse.ArgumentParser(description="Summarize binary drift-detection metrics.")
# #     parser.add_argument("--models-ckpt", type=Path, default=Path("models-ckpt"))
# #     parser.add_argument("--baseline-checkpoint", required=True)
# #     parser.add_argument("--adaptive-checkpoint", required=True)
# #     parser.add_argument("--win-size", type=int, default=5)
# #     parser.add_argument("--threshold", type=float, default=5.0)
# #     parser.add_argument("--policy", choices=["max_abs_t", "max_t", "final_t"], default="max_t")
# #     parser.add_argument("--metric", default="accuracy")
# #     parser.add_argument("--output-dir", type=Path, default=Path("report-metrics"))
# #     args = parser.parse_args()
# #
# #     args.output_dir.mkdir(parents=True, exist_ok=True)
# #
# #     implementations = [
# #         ("baseline", args.baseline_checkpoint),
# #         ("adaptive", args.adaptive_checkpoint),
# #     ]
# #
# #     all_records: List[RunDetectionRecord] = []
# #
# #     for impl_name, checkpoint in implementations:
# #         results_v2_path = args.models_ckpt / f"{checkpoint}-results-v2.pkl"
# #         if results_v2_path.exists():
# #             records = collect_results_v2_records(
# #                 implementation=impl_name,
# #                 checkpoint=checkpoint,
# #                 results_v2_path=results_v2_path,
# #                 threshold=args.threshold,
# #             )
# #         else:
# #             records = collect_supwise_records(
# #                 implementation=impl_name,
# #                 checkpoint=checkpoint,
# #                 supwise_dir=args.models_ckpt / "sup-wise",
# #                 win_size=args.win_size,
# #                 threshold=args.threshold,
# #                 policy=args.policy,
# #                 metric=args.metric,
# #             )
# #         all_records.extend(records)
# #
# #     if not all_records:
# #         raise SystemExit("No detection records found.")
# #
# #     raw_df = pd.DataFrame([asdict(r) for r in all_records])
# #     raw_csv = args.output_dir / "detection_raw_records.csv"
# #     raw_df.to_csv(raw_csv, index=False)
# #
# #     overall = summarize_detection(raw_df, ["implementation"])
# #     by_support = summarize_detection(raw_df, ["implementation", "support_low", "support_high"])
# #
# #     overall_csv = args.output_dir / "detection_summary_overall.csv"
# #     by_support_csv = args.output_dir / "detection_summary_by_support.csv"
# #     overall.to_csv(overall_csv, index=False)
# #     by_support.to_csv(by_support_csv, index=False)
# #
# #     print("\n=== Detection Summary Overall ===")
# #     print(overall.to_string(index=False))
# #     print("\n=== Detection Summary By Support ===")
# #     print(by_support.to_string(index=False))
# #
# # if __name__ == "__main__":
# #     main()