#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
import re
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.metrics._ranking import _ndcg_sample_scores
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")


@dataclass
class RankingExperimentRecord:
    implementation: str
    checkpoint: str
    support_low: Optional[float]
    support_high: Optional[float]
    method: str
    nDCG: float
    nDCG_10: float
    nDCG_100: float
    Pearson: float
    Spearman: float
    source_file: str
    win_size: int
    comparison: str


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


def build_fi_maps(matches_obj: dict):
    fi_df = matches_obj["matches_train"].fi.reset_index(drop=True)
    subgroup_to_idx: Dict[Tuple[int, ...], int] = {}
    subgroup_to_support: Dict[Tuple[int, ...], float] = {}

    for idx, row in fi_df.iterrows():
        sg = normalize_itemset(row["itemsets"])
        subgroup_to_idx[sg] = idx
        subgroup_to_support[sg] = float(row["support"])

    return subgroup_to_idx, subgroup_to_support, len(fi_df)


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


def divs_to_metric_matrix(divs_list, subgroup_to_idx, n_groups, metric="accuracy") -> np.ndarray:
    out = np.full((len(divs_list), n_groups), np.nan, dtype=float)

    for b, df in enumerate(divs_list):
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        vals = get_metric_values(df, metric)

        subgroup_col = None
        if "subgroup" in df.columns:
            subgroup_col = "subgroup"
        elif "itemsets" in df.columns:
            subgroup_col = "itemsets"

        if subgroup_col is None:
            k = min(len(vals), n_groups)
            out[b, :k] = vals[:k]
            continue

        for row_idx, (_, row) in enumerate(df.iterrows()):
            sg = normalize_itemset(row[subgroup_col])
            if sg in subgroup_to_idx:
                out[b, subgroup_to_idx[sg]] = float(vals[row_idx])

    return out


def altered_fraction_in_window(
    altered_list,
    matches_batches,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    if not altered_list or not matches_batches:
        return np.array([], dtype=float)

    first_matches = getattr(matches_batches[0], "matches", None)
    if first_matches is None:
        return np.array([], dtype=float)

    n_groups = first_matches.shape[1]
    altered_counts = np.zeros(n_groups, dtype=float)
    subgroup_sizes = np.zeros(n_groups, dtype=float)

    for b in range(start_idx, end_idx):
        if b < 0 or b >= len(altered_list) or b >= len(matches_batches):
            continue

        M = getattr(matches_batches[b], "matches", None)
        if M is None:
            continue

        altered_mask = np.asarray(altered_list[b], dtype=bool).ravel()
        if altered_mask.shape[0] != M.shape[0]:
            m = min(altered_mask.shape[0], M.shape[0])
            altered_mask = altered_mask[:m]
            M = M[:m]

        subgroup_sizes += np.asarray(M.sum(axis=0)).ravel().astype(float)
        altered_counts += np.asarray(M[altered_mask].sum(axis=0)).ravel().astype(float)

    out = np.zeros_like(subgroup_sizes, dtype=float)
    valid = subgroup_sizes > 0
    out[valid] = altered_counts[valid] / subgroup_sizes[valid]
    return out


def compute_final_window_arrays(
    metric_mat: np.ndarray,
    altered_list,
    matches_batches,
    win_size: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Paper-closer ranking evaluation:
    - compare a reference window to the latest/current window
    - use altered fraction in the final/current window as relevance
    """
    if metric_mat.size == 0:
        return None, None, None

    n_common = min(metric_mat.shape[0], len(altered_list), len(matches_batches))
    if n_common < 2 * win_size:
        return None, None, None

    metric_mat = metric_mat[:n_common]

    ref = metric_mat[:win_size, :]
    cur = metric_mat[n_common - win_size : n_common, :]

    with np.errstate(invalid="ignore", divide="ignore"):
        ref_mean = np.nanmean(ref, axis=0)
        cur_mean = np.nanmean(cur, axis=0)
        delta = cur_mean - ref_mean

        ref_std = np.nanstd(ref, axis=0, ddof=1)
        cur_std = np.nanstd(cur, axis=0, ddof=1)
        n_ref = np.sum(~np.isnan(ref), axis=0)
        n_cur = np.sum(~np.isnan(cur), axis=0)

        se = np.sqrt((ref_std ** 2) / np.maximum(n_ref, 1) + (cur_std ** 2) / np.maximum(n_cur, 1))

    tstat = np.zeros_like(delta, dtype=float)
    valid = np.isfinite(se) & (se > 0)
    tstat[valid] = -delta[valid] / se[valid]

    gt = altered_fraction_in_window(
        altered_list=altered_list,
        matches_batches=matches_batches,
        start_idx=n_common - win_size,
        end_idx=n_common,
    )

    gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
    delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    tstat = np.nan_to_num(tstat, nan=0.0, posinf=0.0, neginf=0.0)

    return gt.reshape(1, -1), delta.reshape(1, -1), tstat.reshape(1, -1)


def _safe_mean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if x.size else float("nan")


def compute_metrics_from_scores(all_gt: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float, float, float]:
    ndcg_full = np.asarray(_ndcg_sample_scores(all_gt, y_score, k=None), dtype=float)
    ndcg_10 = np.asarray(_ndcg_sample_scores(all_gt, y_score, k=10), dtype=float)
    ndcg_100 = np.asarray(_ndcg_sample_scores(all_gt, y_score, k=100), dtype=float)

    pearson_vals = []
    spearman_vals = []

    for gt_row, score_row in zip(all_gt, y_score):
        gt_row = np.asarray(gt_row, dtype=float).ravel()
        score_row = np.asarray(score_row, dtype=float).ravel()

        valid = np.isfinite(gt_row) & np.isfinite(score_row)
        gt_row = gt_row[valid]
        score_row = score_row[valid]

        if gt_row.size < 2 or score_row.size < 2:
            pearson_vals.append(np.nan)
            spearman_vals.append(np.nan)
            continue

        if np.all(gt_row == gt_row[0]) or np.all(score_row == score_row[0]):
            pearson_vals.append(np.nan)
            spearman_vals.append(np.nan)
            continue

        try:
            pearson_vals.append(float(pearsonr(gt_row, score_row).statistic))
        except Exception:
            pearson_vals.append(np.nan)

        try:
            spearman_vals.append(float(spearmanr(gt_row, score_row).statistic))
        except Exception:
            spearman_vals.append(np.nan)

    return (
        _safe_mean(ndcg_full),
        _safe_mean(ndcg_10),
        _safe_mean(ndcg_100),
        _safe_mean(pearson_vals),
        _safe_mean(spearman_vals),
    )


def group_files_by_support(files: List[Path]) -> Dict[Tuple[Optional[float], Optional[float]], List[Path]]:
    out: Dict[Tuple[Optional[float], Optional[float]], List[Path]] = {}
    for path in files:
        support_low, support_high, _ = parse_support_and_noise_from_name(path.name)
        key = (support_low, support_high)
        out.setdefault(key, []).append(path)
    return out

def sample_positive_files_by_support(
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

    pos_files: List[Path] = []
    for pat in pos_patterns:
        pos_files.extend(supwise_dir.glob(pat))

    pos_files = sorted(set(pos_files))
    pos_by_sup = group_files_by_support(pos_files)

    sampled: List[Path] = []
    for _, bucket in sorted(pos_by_sup.items()):
        n = min(max_per_support, len(bucket))
        if n <= 0:
            continue
        idx = rng.choice(len(bucket), size=n, replace=False)
        sampled.extend(bucket[i] for i in idx)

    return sorted(sampled, key=lambda p: str(p))


def append_rows_to_csv(csv_path: Path, rows: List[RankingExperimentRecord]):
    if not rows:
        return

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def load_processed_keys(raw_csv_path: Path) -> Set[Tuple[str, str, str, str, int, str]]:
    if not raw_csv_path.exists():
        return set()

    try:
        df = pd.read_csv(raw_csv_path)
    except Exception:
        return set()

    needed = {"implementation", "checkpoint", "source_file", "method", "win_size", "comparison"}
    if not needed.issubset(df.columns):
        return set()

    return set(
        zip(
            df["implementation"].astype(str),
            df["checkpoint"].astype(str),
            df["source_file"].astype(str),
            df["method"].astype(str),
            pd.to_numeric(df["win_size"], errors="coerce").fillna(-1).astype(int),
            df["comparison"].astype(str),
        )
    )


def summarize_rows(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)[["nDCG", "nDCG_10", "nDCG_100", "Pearson", "Spearman"]]
        .mean()
        .reset_index()
    )


def save_partial_summaries(raw_csv_path: Path, output_dir: Path):
    if not raw_csv_path.exists():
        return

    raw_df = pd.read_csv(raw_csv_path)
    if raw_df.empty:
        return

    overall = summarize_rows(raw_df, ["implementation", "method"])
    by_support = summarize_rows(raw_df, ["implementation", "method", "support_low", "support_high"])

    overall.to_csv(output_dir / "ranking_summary_overall.csv", index=False)
    by_support.to_csv(output_dir / "ranking_summary_by_support.csv", index=False)


def log_progress(txt_path: Path, msg: str):
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    parser = argparse.ArgumentParser(description="Paper-aligned subgroup ranking summary.")
    parser.add_argument("--models-ckpt", type=Path, default=Path("models-ckpt"))
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--adaptive-checkpoint", required=True)
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--win-size", type=int, default=5)
    parser.add_argument("--max-per-support", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comparison", choices=["first_vs_last"], default="first_vs_last")
    parser.add_argument("--methods", nargs="+", default=["delta", "tstat", "random"])
    parser.add_argument("--output-dir", type=Path, default=Path("report-metrics"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = args.output_dir / "ranking_raw_records.csv"
    progress_txt = args.output_dir / "ranking_progress.txt"

    processed_keys = load_processed_keys(raw_csv_path) if args.resume else set()
    rng = np.random.default_rng(args.seed)

    implementations = [
        ("baseline", args.baseline_checkpoint),
        ("adaptive", args.adaptive_checkpoint),
    ]

    file_counter = 0

    for implementation, checkpoint in tqdm(implementations, desc="Implementations"):
        matches_path = args.models_ckpt / f"matches-{checkpoint}.pkl"
        matches_obj = safe_pickle_load(matches_path)
        if matches_obj is None:
            msg = f"[WARN] Missing matches file for {checkpoint}"
            print(msg)
            log_progress(progress_txt, msg)
            continue

        subgroup_to_idx, _, n_groups = build_fi_maps(matches_obj)
        matches_batches = matches_obj.get("matches_batches")
        if not matches_batches:
            msg = f"[WARN] Missing or empty matches_batches for {checkpoint}"
            print(msg)
            log_progress(progress_txt, msg)
            continue

        files = sample_positive_files_by_support(
            models_ckpt=args.models_ckpt,
            checkpoint=checkpoint,
            max_per_support=args.max_per_support,
            seed=args.seed,
        )

        if args.max_files is not None:
            files = files[: args.max_files]

        new_rows: List[RankingExperimentRecord] = []

        for path in tqdm(files, desc=f"{implementation}:{checkpoint}", leave=False):
            payload = safe_pickle_load(path)
            if payload is None:
                continue

            support_low, support_high, noise = parse_support_and_noise_from_name(path.name)
            if noise is None or noise <= 0:
                continue

            divs = payload.get("divs", [])
            altered = payload.get("altered", [])
            if not divs or not altered:
                continue

            try:
                metric_mat = divs_to_metric_matrix(divs, subgroup_to_idx, n_groups, metric=args.metric)
                gt, delta, tstat = compute_final_window_arrays(
                    metric_mat=metric_mat,
                    altered_list=altered,
                    matches_batches=matches_batches,
                    win_size=args.win_size,
                )
            except Exception as e:
                print(f"[WARN] Failed on {path}: {e}")
                continue

            if gt is None:
                continue

            method_map = {}
            if "delta" in args.methods:
                method_map["delta"] = -delta
            if "tstat" in args.methods:
                method_map["tstat"] = tstat
            if "random" in args.methods:
                method_map["random"] = rng.random(gt.shape)

            for method_name, y_score in method_map.items():
                key = (
                    implementation,
                    checkpoint,
                    str(path),
                    method_name,
                    int(args.win_size),
                    args.comparison,
                )
                if key in processed_keys:
                    continue

                ndcg_full, ndcg_10, ndcg_100, pearson, spearman = compute_metrics_from_scores(gt, y_score)

                new_rows.append(
                    RankingExperimentRecord(
                        implementation=implementation,
                        checkpoint=checkpoint,
                        support_low=support_low,
                        support_high=support_high,
                        method=method_name,
                        nDCG=ndcg_full,
                        nDCG_10=ndcg_10,
                        nDCG_100=ndcg_100,
                        Pearson=pearson,
                        Spearman=spearman,
                        source_file=str(path),
                        win_size=int(args.win_size),
                        comparison=args.comparison,
                    )
                )
                processed_keys.add(key)

            file_counter += 1
            if file_counter % args.save_every == 0:
                append_rows_to_csv(raw_csv_path, new_rows)
                new_rows = []
                save_partial_summaries(raw_csv_path, args.output_dir)
                log_progress(progress_txt, f"Processed {file_counter} files")

        if new_rows:
            append_rows_to_csv(raw_csv_path, new_rows)

    save_partial_summaries(raw_csv_path, args.output_dir)
    log_progress(progress_txt, "Completed ranking summary run.")

    overall_path = args.output_dir / "ranking_summary_overall.csv"
    by_support_path = args.output_dir / "ranking_summary_by_support.csv"

    if overall_path.exists():
        print("\n=== Ranking Summary Overall ===")
        print(pd.read_csv(overall_path).to_string(index=False))
    else:
        raise SystemExit("No ranking records produced.")

    if by_support_path.exists():
        print("\n=== Ranking Summary By Support ===")
        print(pd.read_csv(by_support_path).to_string(index=False))


if __name__ == "__main__":
    main()

# # #!/usr/bin/env python3
# # """
# # python ranking_summary.py --baseline-checkpoint adult_model --adaptive-checkpoint adult_model_adaptive --win-size 5
# # """
# #!/usr/bin/env python3
# """
# python src/ranking_summary.py --baseline-checkpoint adult_model --adaptive-checkpoint adult_model_adaptive --win-size 5 --resume
# """
# from __future__ import annotations
#
# import argparse
# import csv
# import pickle
# import re
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Set
#
# import numpy as np
# import pandas as pd
# from scipy.stats import pearsonr, spearmanr, ConstantInputWarning
# import warnings
# from sklearn.metrics._ranking import _ndcg_sample_scores
# from tqdm import tqdm
#
#
# warnings.filterwarnings("ignore", category=ConstantInputWarning)
#
# @dataclass
# class RankingWindowRecord:
#     implementation: str
#     checkpoint: str
#     support_low: Optional[float]
#     support_high: Optional[float]
#     method: str
#     nDCG: float
#     nDCG_10: float
#     nDCG_100: float
#     Pearson: float
#     Spearman: float
#     source_file: str
#
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
#     fi_df = matches_obj["matches_train"].fi
#     subgroup_to_idx: Dict[Tuple[int, ...], int] = {}
#     subgroup_to_support: Dict[Tuple[int, ...], float] = {}
#
#     fi_df = fi_df.reset_index(drop=True)
#     for idx, row in fi_df.iterrows():
#         sg = normalize_itemset(row["itemsets"])
#         subgroup_to_idx[sg] = idx
#         subgroup_to_support[sg] = float(row["support"])
#     return subgroup_to_idx, subgroup_to_support, len(fi_df)
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
# def divs_to_metric_matrix(divs_list, subgroup_to_idx, n_groups, metric="accuracy"):
#     out = np.full((len(divs_list), n_groups), np.nan, dtype=float)
#
#     for b, df in enumerate(divs_list):
#         vals = get_metric_values(df, metric)
#
#         if "subgroup" in df.columns:
#             for row_idx, (_, row) in enumerate(df.iterrows()):
#                 sg = normalize_itemset(row["subgroup"])
#                 if sg in subgroup_to_idx:
#                     out[b, subgroup_to_idx[sg]] = float(vals[row_idx])
#
#         elif "itemsets" in df.columns:
#             for row_idx, (_, row) in enumerate(df.iterrows()):
#                 sg = normalize_itemset(row["itemsets"])
#                 if sg in subgroup_to_idx:
#                     out[b, subgroup_to_idx[sg]] = float(vals[row_idx])
#
#         else:
#             k = min(len(vals), n_groups)
#             out[b, :k] = vals[:k]
#
#     return out
#
#
# def altered_masks_to_fraction_matrix(altered_list, matches_batches):
#     n_batches = min(len(altered_list), len(matches_batches))
#     n_groups = matches_batches[0].matches.shape[1]
#     out = np.full((n_batches, n_groups), np.nan, dtype=float)
#
#     for b in range(n_batches):
#         altered_mask = np.asarray(altered_list[b], dtype=bool)
#         M = matches_batches[b].matches
#
#         subgroup_sizes = np.asarray(M.sum(axis=0)).ravel().astype(float)
#         altered_counts = np.asarray(M[altered_mask].sum(axis=0)).ravel().astype(float)
#
#         frac = np.zeros_like(subgroup_sizes, dtype=float)
#         valid = subgroup_sizes > 0
#         frac[valid] = altered_counts[valid] / subgroup_sizes[valid]
#         frac[~valid] = np.nan
#         out[b] = frac
#
#     return out
#
#
# def compute_window_arrays(metric_mat: np.ndarray, altered_frac_mat: np.ndarray, win_size: int):
#     n_common = min(metric_mat.shape[0], altered_frac_mat.shape[0])
#     metric_mat = metric_mat[:n_common]
#     altered_frac_mat = altered_frac_mat[:n_common]
#
#     GT, DELTA, TSTAT = [], [], []
#
#     for end in range(2 * win_size - 1, n_common):
#         ref = metric_mat[end - 2 * win_size + 1 : end - win_size + 1, :]
#         cur = metric_mat[end - win_size + 1 : end + 1, :]
#         gt_cur = altered_frac_mat[end - win_size + 1 : end + 1, :]
#
#         ref_mean = np.nanmean(ref, axis=0)
#         cur_mean = np.nanmean(cur, axis=0)
#         delta = cur_mean - ref_mean
#
#         ref_std = np.nanstd(ref, axis=0, ddof=1)
#         cur_std = np.nanstd(cur, axis=0, ddof=1)
#         n_ref = np.sum(~np.isnan(ref), axis=0)
#         n_cur = np.sum(~np.isnan(cur), axis=0)
#
#         se = np.sqrt((ref_std ** 2) / np.maximum(n_ref, 1) + (cur_std ** 2) / np.maximum(n_cur, 1))
#         tstat = np.zeros_like(delta)
#         valid = se > 0
#         tstat[valid] = -delta[valid] / se[valid]
#
#         gt = np.nanmean(gt_cur, axis=0)
#         GT.append(np.nan_to_num(gt, nan=0.0))
#         DELTA.append(np.nan_to_num(delta, nan=0.0))
#         TSTAT.append(np.nan_to_num(tstat, nan=0.0))
#
#     if not GT:
#         return None, None, None
#
#     return np.vstack(GT), np.vstack(DELTA), np.vstack(TSTAT)
#
#
# def _safe_mean(x) -> float:
#     x = np.asarray(x, dtype=float)
#     x = x[~np.isnan(x)]
#     return float(np.mean(x)) if x.size else float("nan")
#
#
# def compute_metrics_from_scores(all_GT: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float, float, float]:
#     ndcg_full = np.asarray(_ndcg_sample_scores(all_GT, y_score, k=None), dtype=float)
#     ndcg_10 = np.asarray(_ndcg_sample_scores(all_GT, y_score, k=10), dtype=float)
#     ndcg_100 = np.asarray(_ndcg_sample_scores(all_GT, y_score, k=100), dtype=float)
#
#     pearson_vals = []
#     spearman_vals = []
#
#     for gt_row, score_row in zip(all_GT, y_score):
#         gt_row = np.asarray(gt_row, dtype=float).ravel()
#         score_row = np.asarray(score_row, dtype=float).ravel()
#
#         valid = np.isfinite(gt_row) & np.isfinite(score_row)
#         gt_row = gt_row[valid]
#         score_row = score_row[valid]
#
#         if gt_row.size < 2 or score_row.size < 2:
#             pearson_vals.append(np.nan)
#             spearman_vals.append(np.nan)
#             continue
#
#         if np.all(gt_row == gt_row[0]) or np.all(score_row == score_row[0]):
#             pearson_vals.append(np.nan)
#             spearman_vals.append(np.nan)
#             continue
#
#         try:
#             pearson_vals.append(float(pearsonr(gt_row, score_row).statistic))
#         except Exception:
#             pearson_vals.append(np.nan)
#
#         try:
#             spearman_vals.append(float(spearmanr(gt_row, score_row).statistic))
#         except Exception:
#             spearman_vals.append(np.nan)
#
#     return (
#         _safe_mean(ndcg_full),
#         _safe_mean(ndcg_10),
#         _safe_mean(ndcg_100),
#         _safe_mean(pearson_vals),
#         _safe_mean(spearman_vals),
#     )
#
# def summarize_rows(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
#     return (
#         df.groupby(group_cols, dropna=False)[["nDCG", "nDCG_10", "nDCG_100", "Pearson", "Spearman"]]
#         .mean()
#         .reset_index()
#     )
#
#
# def load_processed_keys(raw_csv_path: Path) -> Set[Tuple[str, str, str, str]]:
#     if not raw_csv_path.exists():
#         return set()
#     df = pd.read_csv(raw_csv_path)
#     needed = {"implementation", "checkpoint", "source_file", "method"}
#     if not needed.issubset(df.columns):
#         return set()
#     return set(
#         zip(
#             df["implementation"].astype(str),
#             df["checkpoint"].astype(str),
#             df["source_file"].astype(str),
#             df["method"].astype(str),
#         )
#     )
#
#
# def append_rows_to_csv(csv_path: Path, rows: List[RankingWindowRecord]):
#     if not rows:
#         return
#
#     write_header = not csv_path.exists()
#     with open(csv_path, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
#         if write_header:
#             writer.writeheader()
#         for row in rows:
#             writer.writerow(asdict(row))
#
#
# def save_partial_summaries(raw_csv_path: Path, output_dir: Path):
#     if not raw_csv_path.exists():
#         return
#
#     raw_df = pd.read_csv(raw_csv_path)
#     if raw_df.empty:
#         return
#
#     overall = summarize_rows(raw_df, ["implementation", "method"])
#     by_support = summarize_rows(raw_df, ["implementation", "method", "support_low", "support_high"])
#
#     overall.to_csv(output_dir / "ranking_summary_overall.csv", index=False)
#     by_support.to_csv(output_dir / "ranking_summary_by_support.csv", index=False)
#
#
# def log_progress(txt_path: Path, msg: str):
#     with open(txt_path, "a", encoding="utf-8") as f:
#         f.write(msg + "\n")
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Summarize subgroup ranking metrics.")
#     parser.add_argument("--models-ckpt", type=Path, default=Path("models-ckpt"))
#     parser.add_argument("--baseline-checkpoint", required=True)
#     parser.add_argument("--adaptive-checkpoint", required=True)
#     parser.add_argument("--metric", default="accuracy")
#     parser.add_argument("--win-size", type=int, default=5)
#     parser.add_argument("--output-dir", type=Path, default=Path("report-metrics"))
#     parser.add_argument("--resume", action="store_true")
#     parser.add_argument("--max-files", type=int, default=None)
#     parser.add_argument("--methods", nargs="+", default=["delta", "tstat", "random"])
#     parser.add_argument("--save-every", type=int, default=25)
#     args = parser.parse_args()
#
#     args.output_dir.mkdir(parents=True, exist_ok=True)
#
#     raw_csv_path = args.output_dir / "ranking_raw_records.csv"
#     progress_txt = args.output_dir / "ranking_progress.txt"
#
#     processed_keys = load_processed_keys(raw_csv_path) if args.resume else set()
#     rng = np.random.default_rng(42)
#
#     implementations = [
#         ("baseline", args.baseline_checkpoint),
#         ("adaptive", args.adaptive_checkpoint),
#     ]
#
#     file_counter = 0
#
#     for implementation, checkpoint in tqdm(implementations, desc="Implementations"):
#         matches_path = args.models_ckpt / f"matches-{checkpoint}.pkl"
#         matches_obj = safe_pickle_load(matches_path)
#         if matches_obj is None:
#             msg = f"[WARN] Missing matches file for {checkpoint}"
#             print(msg)
#             log_progress(progress_txt, msg)
#             continue
#
#         subgroup_to_idx, _, n_groups = build_fi_maps(matches_obj)
#         matches_batches = matches_obj["matches_batches"]
#
#         supwise_pattern = f"{checkpoint}-noise-0.50-support-*-target-*.pkl"
#         supwise_files = sorted((args.models_ckpt / "sup-wise").glob(supwise_pattern))
#
#         if args.max_files is not None:
#             supwise_files = supwise_files[: args.max_files]
#
#         file_iter = tqdm(supwise_files, desc=f"{implementation}:{checkpoint}", leave=False)
#
#         for path in file_iter:
#             payload = safe_pickle_load(path)
#             if payload is None:
#                 continue
#
#             support_low, support_high, noise = parse_support_and_noise_from_name(path.name)
#             if noise is None or noise <= 0:
#                 continue
#
#             divs = payload.get("divs", [])
#             altered = payload.get("altered", [])
#             if not divs or not altered:
#                 continue
#
#             metric_mat = divs_to_metric_matrix(divs, subgroup_to_idx, n_groups, metric=args.metric)
#             altered_frac_mat = altered_masks_to_fraction_matrix(altered, matches_batches)
#
#             GT, DELTA, TSTAT = compute_window_arrays(metric_mat, altered_frac_mat, args.win_size)
#             if GT is None:
#                 continue
#
#             method_map = {}
#             if "delta" in args.methods:
#                 method_map["delta"] = -DELTA
#             if "tstat" in args.methods:
#                 method_map["tstat"] = TSTAT
#             if "random" in args.methods:
#                 method_map["random"] = rng.random(GT.shape)
#
#             new_rows: List[RankingWindowRecord] = []
#
#             for method_name, y_score in method_map.items():
#                 key = (implementation, checkpoint, str(path), method_name)
#                 if key in processed_keys:
#                     continue
#
#                 ndcg_full, ndcg_10, ndcg_100, pearson, spearman = compute_metrics_from_scores(GT, y_score)
#                 row = RankingWindowRecord(
#                     implementation=implementation,
#                     checkpoint=checkpoint,
#                     support_low=support_low,
#                     support_high=support_high,
#                     method=method_name,
#                     nDCG=ndcg_full,
#                     nDCG_10=ndcg_10,
#                     nDCG_100=ndcg_100,
#                     Pearson=pearson,
#                     Spearman=spearman,
#                     source_file=str(path),
#                 )
#                 new_rows.append(row)
#                 processed_keys.add(key)
#
#             if new_rows:
#                 append_rows_to_csv(raw_csv_path, new_rows)
#
#             file_counter += 1
#             if file_counter % args.save_every == 0:
#                 save_partial_summaries(raw_csv_path, args.output_dir)
#                 log_progress(progress_txt, f"Processed {file_counter} files so far...")
#
#     save_partial_summaries(raw_csv_path, args.output_dir)
#     log_progress(progress_txt, "Completed ranking summary run.")
#
#     if raw_csv_path.exists():
#         raw_df = pd.read_csv(raw_csv_path)
#         print("\n=== Ranking Summary Overall ===")
#         overall = pd.read_csv(args.output_dir / "ranking_summary_overall.csv")
#         print(overall.to_string(index=False))
#
#         print("\n=== Ranking Summary By Support ===")
#         by_support = pd.read_csv(args.output_dir / "ranking_summary_by_support.csv")
#         print(by_support.to_string(index=False))
#     else:
#         raise SystemExit("No ranking records produced.")
#
#
# if __name__ == "__main__":
#     main()
#
#
# # from __future__ import annotations
# #
# # import argparse
# # import pickle
# # import re
# # from dataclasses import dataclass, asdict
# # from pathlib import Path
# # from typing import Dict, List, Optional, Tuple
# #
# # import numpy as np
# # import pandas as pd
# # from scipy.stats import spearmanr
# # from sklearn.metrics._ranking import _ndcg_sample_scores
# #
# #
# # @dataclass
# # class RankingWindowRecord:
# #     implementation: str
# #     checkpoint: str
# #     support_low: Optional[float]
# #     support_high: Optional[float]
# #     method: str
# #     nDCG: float
# #     nDCG_10: float
# #     nDCG_100: float
# #     Pearson: float
# #     Spearman: float
# #     source_file: str
# #
# #
# # def safe_pickle_load(path: Path):
# #     try:
# #         with open(path, "rb") as f:
# #             return pickle.load(f)
# #     except Exception as e:
# #         print(f"[WARN] Could not load {path}: {e}")
# #         return None
# #
# #
# # def normalize_itemset(x) -> Tuple[int, ...]:
# #     try:
# #         return tuple(sorted(int(v) for v in x))
# #     except Exception:
# #         return tuple()
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
# # def build_fi_maps(matches_obj: dict):
# #     fi_df = matches_obj["matches_train"].fi
# #     subgroup_to_idx: Dict[Tuple[int, ...], int] = {}
# #     subgroup_to_support: Dict[Tuple[int, ...], float] = {}
# #
# #     fi_df = fi_df.reset_index(drop=True)
# #     for idx, row in fi_df.iterrows():
# #         sg = normalize_itemset(row["itemsets"])
# #         subgroup_to_idx[sg] = idx
# #         subgroup_to_support[sg] = float(row["support"])
# #     return subgroup_to_idx, subgroup_to_support, len(fi_df)
# #
# #
# # def get_metric_values(df: pd.DataFrame, metric: str = "accuracy") -> np.ndarray:
# #     lower_map = {c.lower(): c for c in df.columns}
# #
# #     if metric.lower() in lower_map:
# #         return df[lower_map[metric.lower()]].to_numpy(dtype=float)
# #
# #     if metric.lower() == "accuracy":
# #         needed = ["tp", "tn", "fp", "fn"]
# #         if all(col in lower_map for col in needed):
# #             tp = df[lower_map["tp"]].to_numpy(dtype=float)
# #             tn = df[lower_map["tn"]].to_numpy(dtype=float)
# #             fp = df[lower_map["fp"]].to_numpy(dtype=float)
# #             fn = df[lower_map["fn"]].to_numpy(dtype=float)
# #
# #             denom = tp + tn + fp + fn
# #             return np.divide(tp + tn, denom, out=np.zeros_like(denom), where=denom > 0)
# #
# #     raise KeyError(f"Could not compute metric '{metric}' from columns: {list(df.columns)}")
# #
# #
# # def divs_to_metric_matrix(divs_list, subgroup_to_idx, n_groups, metric="accuracy"):
# #     out = np.full((len(divs_list), n_groups), np.nan, dtype=float)
# #
# #     for b, df in enumerate(divs_list):
# #         vals = get_metric_values(df, metric)
# #
# #         if "subgroup" in df.columns:
# #             for row_idx, (_, row) in enumerate(df.iterrows()):
# #                 sg = normalize_itemset(row["subgroup"])
# #                 if sg in subgroup_to_idx:
# #                     out[b, subgroup_to_idx[sg]] = float(vals[row_idx])
# #         elif "itemsets" in df.columns:
# #             for row_idx, (_, row) in enumerate(df.iterrows()):
# #                 sg = normalize_itemset(row["itemsets"])
# #                 if sg in subgroup_to_idx:
# #                     out[b, subgroup_to_idx[sg]] = float(vals[row_idx])
# #         else:
# #             k = min(len(vals), n_groups)
# #             out[b, :k] = vals[:k]
# #
# #     return out
# #
# #
# # def altered_masks_to_fraction_matrix(altered_list, matches_batches):
# #     """
# #     Ground-truth subgroup relevance:
# #     altered fraction = altered_count_in_subgroup / subgroup_size
# #     """
# #     n_batches = min(len(altered_list), len(matches_batches))
# #     n_groups = matches_batches[0].matches.shape[1]
# #     out = np.full((n_batches, n_groups), np.nan, dtype=float)
# #
# #     for b in range(n_batches):
# #         altered_mask = np.asarray(altered_list[b], dtype=bool)
# #         M = matches_batches[b].matches
# #
# #         subgroup_sizes = np.asarray(M.sum(axis=0)).ravel().astype(float)
# #         altered_counts = np.asarray(M[altered_mask].sum(axis=0)).ravel().astype(float)
# #
# #         frac = np.zeros_like(subgroup_sizes, dtype=float)
# #         valid = subgroup_sizes > 0
# #         frac[valid] = altered_counts[valid] / subgroup_sizes[valid]
# #         frac[~valid] = np.nan
# #         out[b] = frac
# #
# #     return out
# #
# #
# # def compute_window_arrays(metric_mat: np.ndarray, altered_frac_mat: np.ndarray, win_size: int):
# #     n_common = min(metric_mat.shape[0], altered_frac_mat.shape[0])
# #     metric_mat = metric_mat[:n_common]
# #     altered_frac_mat = altered_frac_mat[:n_common]
# #
# #     GT, DELTA, TSTAT = [], [], []
# #
# #     for end in range(2 * win_size - 1, n_common):
# #         ref = metric_mat[end - 2 * win_size + 1 : end - win_size + 1, :]
# #         cur = metric_mat[end - win_size + 1 : end + 1, :]
# #         gt_cur = altered_frac_mat[end - win_size + 1 : end + 1, :]
# #
# #         ref_mean = np.nanmean(ref, axis=0)
# #         cur_mean = np.nanmean(cur, axis=0)
# #         delta = cur_mean - ref_mean
# #
# #         ref_std = np.nanstd(ref, axis=0, ddof=1)
# #         cur_std = np.nanstd(cur, axis=0, ddof=1)
# #         n_ref = np.sum(~np.isnan(ref), axis=0)
# #         n_cur = np.sum(~np.isnan(cur), axis=0)
# #
# #         se = np.sqrt((ref_std ** 2) / np.maximum(n_ref, 1) + (cur_std ** 2) / np.maximum(n_cur, 1))
# #         tstat = np.zeros_like(delta)
# #         valid = se > 0
# #         tstat[valid] = -delta[valid] / se[valid]
# #
# #         gt = np.nanmean(gt_cur, axis=0)
# #         GT.append(np.nan_to_num(gt, nan=0.0))
# #         DELTA.append(np.nan_to_num(delta, nan=0.0))
# #         TSTAT.append(np.nan_to_num(tstat, nan=0.0))
# #
# #     if not GT:
# #         return None, None, None
# #
# #     return np.vstack(GT), np.vstack(DELTA), np.vstack(TSTAT)
# #
# #
# # def compute_metrics_from_scores(all_GT: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float, float, float]:
# #     ndcg_full = _ndcg_sample_scores(all_GT, y_score, k=None)
# #     ndcg_10 = _ndcg_sample_scores(all_GT, y_score, k=10)
# #     ndcg_100 = _ndcg_sample_scores(all_GT, y_score, k=100)
# #
# #     gt_centered = all_GT - all_GT.mean(axis=1, keepdims=True)
# #     ys_centered = y_score - y_score.mean(axis=1, keepdims=True)
# #     denom = all_GT.std(axis=1) * y_score.std(axis=1)
# #     pearson = np.divide(
# #         (gt_centered * ys_centered).mean(axis=1),
# #         denom,
# #         out=np.zeros_like(denom),
# #         where=denom > 0,
# #     )
# #
# #     sp = spearmanr(all_GT, y_score, axis=1).statistic
# #     n = all_GT.shape[0]
# #     spearman = np.diagonal(sp, offset=n)
# #
# #     return (
# #         float(np.mean(ndcg_full)),
# #         float(np.mean(ndcg_10)),
# #         float(np.mean(ndcg_100)),
# #         float(np.mean(pearson)),
# #         float(np.mean(spearman)),
# #     )
# #
# #
# # def summarize_rows(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
# #     return (
# #         df.groupby(group_cols, dropna=False)[["nDCG", "nDCG_10", "nDCG_100", "Pearson", "Spearman"]]
# #         .mean()
# #         .reset_index()
# #     )
# #
# #
# # def main():
# #     parser = argparse.ArgumentParser(description="Summarize subgroup ranking metrics.")
# #     parser.add_argument("--models-ckpt", type=Path, default=Path("models-ckpt"))
# #     parser.add_argument("--baseline-checkpoint", required=True)
# #     parser.add_argument("--adaptive-checkpoint", required=True)
# #     parser.add_argument("--metric", default="accuracy")
# #     parser.add_argument("--win-size", type=int, default=5)
# #     parser.add_argument("--output-dir", type=Path, default=Path("report-metrics"))
# #     args = parser.parse_args()
# #
# #     args.output_dir.mkdir(parents=True, exist_ok=True)
# #
# #     rows: List[RankingWindowRecord] = []
# #     rng = np.random.default_rng(42)
# #
# #     for implementation, checkpoint in [
# #         ("baseline", args.baseline_checkpoint),
# #         ("adaptive", args.adaptive_checkpoint),
# #     ]:
# #         matches_path = args.models_ckpt / f"matches-{checkpoint}.pkl"
# #         matches_obj = safe_pickle_load(matches_path)
# #         if matches_obj is None:
# #             print(f"[WARN] Missing matches file for {checkpoint}")
# #             continue
# #
# #         subgroup_to_idx, _, n_groups = build_fi_maps(matches_obj)
# #         matches_batches = matches_obj["matches_batches"]
# #
# #         supwise_pattern = f"{checkpoint}-noise-0.50-support-*-target-*.pkl"
# #         supwise_files = sorted((args.models_ckpt / "sup-wise").glob(supwise_pattern))
# #
# #         for path in supwise_files:
# #             payload = safe_pickle_load(path)
# #             if payload is None:
# #                 continue
# #
# #             support_low, support_high, noise = parse_support_and_noise_from_name(path.name)
# #             if noise is None or noise <= 0:
# #                 continue
# #
# #             divs = payload.get("divs", [])
# #             altered = payload.get("altered", [])
# #             if not divs or not altered:
# #                 continue
# #
# #             metric_mat = divs_to_metric_matrix(divs, subgroup_to_idx, n_groups, metric=args.metric)
# #             altered_frac_mat = altered_masks_to_fraction_matrix(altered, matches_batches)
# #
# #             GT, DELTA, TSTAT = compute_window_arrays(metric_mat, altered_frac_mat, args.win_size)
# #             if GT is None:
# #                 continue
# #
# #             methods = {
# #                 "delta": -DELTA,
# #                 "tstat": TSTAT,
# #                 "random": rng.random(GT.shape),
# #             }
# #
# #             for method_name, y_score in methods.items():
# #                 ndcg_full, ndcg_10, ndcg_100, pearson, spearman = compute_metrics_from_scores(GT, y_score)
# #                 rows.append(
# #                     RankingWindowRecord(
# #                         implementation=implementation,
# #                         checkpoint=checkpoint,
# #                         support_low=support_low,
# #                         support_high=support_high,
# #                         method=method_name,
# #                         nDCG=ndcg_full,
# #                         nDCG_10=ndcg_10,
# #                         nDCG_100=ndcg_100,
# #                         Pearson=pearson,
# #                         Spearman=spearman,
# #                         source_file=str(path),
# #                     )
# #                 )
# #
# #     if not rows:
# #         raise SystemExit("No ranking records produced.")
# #
# #     raw_df = pd.DataFrame([asdict(r) for r in rows])
# #     raw_df.to_csv(args.output_dir / "ranking_raw_records.csv", index=False)
# #
# #     overall = summarize_rows(raw_df, ["implementation", "method"])
# #     by_support = summarize_rows(raw_df, ["implementation", "method", "support_low", "support_high"])
# #
# #     overall.to_csv(args.output_dir / "ranking_summary_overall.csv", index=False)
# #     by_support.to_csv(args.output_dir / "ranking_summary_by_support.csv", index=False)
# #
# #     print("\n=== Ranking Summary Overall ===")
# #     print(overall.to_string(index=False))
# #     print("\n=== Ranking Summary By Support ===")
# #     print(by_support.to_string(index=False))
# #
# #
# # if __name__ == "__main__":
# #     main()