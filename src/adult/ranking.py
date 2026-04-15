from __future__ import annotations

import os
import re
import sys
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics._ranking import _ndcg_sample_scores
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.adult.config import ckpt_dir  # noqa


# ----------------------------
# Settings
# ----------------------------
checkpoint = "adult_model"
metric_name = "accuracy"
noise = 0.50           # use 0.50 for positive runs
win_size = 5
n_samples = 50         # cap experiments per support bucket
compute_corr = True
source_mode = "supwise"   # "supwise" or "drifteval"

# If using source_mode = "drifteval", this is the folder name pattern:
# models-ckpt/adult_model-accuracy-noise-0.50/target-*.pkl


# ----------------------------
# Helpers
# ----------------------------
def normalize_itemset(x):
    """Convert frozenset/list/tuple-like subgroup IDs into a sorted tuple of ints."""
    if isinstance(x, tuple):
        return tuple(sorted(int(v) for v in x))
    if isinstance(x, frozenset):
        return tuple(sorted(int(v) for v in x))
    if isinstance(x, list):
        return tuple(sorted(int(v) for v in x))
    return tuple(sorted(int(v) for v in list(x)))


def parse_support_bucket_from_supwise_name(filename: str):
    """
    Parse support bucket from names like:
    adult_model-noise-0.50-support-0.0428-0.0546-target-34-5-20-55-88.pkl
    """
    m = re.search(r"support-([0-9.]+)-([0-9.]+)-target-", filename)
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2))
    return (lo, hi)


def build_fi_maps(fi_df: pd.DataFrame):
    """
    Build:
      subgroup -> column index
      subgroup -> support
    from fi dataframe.
    """
    subgroup_to_idx = {}
    subgroup_to_support = {}

    for idx, row in fi_df.reset_index(drop=True).iterrows():
        sg = normalize_itemset(row["itemsets"])
        subgroup_to_idx[sg] = idx
        subgroup_to_support[sg] = float(row["support"])

    return subgroup_to_idx, subgroup_to_support


def get_metric_values(df: pd.DataFrame, metric: str = "accuracy"):
    """
    Return a vector of metric values from a div_explorer dataframe.

    Supports:
      - direct metric columns like 'accuracy' or 'acc'
      - confusion-matrix columns ['tp', 'tn', 'fp', 'fn'] for accuracy
    """
    lower_map = {c.lower(): c for c in df.columns}

    # Direct metric column
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
            acc = np.divide(
                tp + tn,
                denom,
                out=np.zeros_like(denom, dtype=float),
                where=denom > 0
            )
            return acc

    raise KeyError(
        f"Could not compute metric '{metric}' from columns: {list(df.columns)}"
    )

def divs_to_metric_matrix(divs_list, subgroup_to_idx, n_groups, metric="accuracy"):
    """
    Convert list of div DataFrames -> matrix [n_batches, n_groups]
    aligned to the fi.itemsets order.
    """
    out = np.full((len(divs_list), n_groups), np.nan, dtype=float)

    for b, df in enumerate(divs_list):
        vals = get_metric_values(df, metric)

        if "itemsets" in df.columns:
            for row_idx, (_, row) in enumerate(df.iterrows()):
                sg = normalize_itemset(row["itemsets"])
                if sg in subgroup_to_idx:
                    j = subgroup_to_idx[sg]
                    out[b, j] = float(vals[row_idx])

        elif "subgroup" in df.columns:
            for row_idx, (_, row) in enumerate(df.iterrows()):
                sg = normalize_itemset(row["subgroup"])
                if sg in subgroup_to_idx:
                    j = subgroup_to_idx[sg]
                    out[b, j] = float(vals[row_idx])

        else:
            # last-resort fallback: assume row order matches fi order
            k = min(len(vals), n_groups)
            out[b, :k] = vals[:k]

    return out

def altered_masks_to_fraction_matrix(altered_list, matches_batches):
    """
    Compute altered fraction per subgroup per batch:
        altered_count_in_subgroup / subgroup_size
    using Matches objects from matches_batches.

    Uses the overlapping batch range if altered_list and matches_batches
    have different lengths.
    """
    n_batches = min(len(altered_list), len(matches_batches))
    n_groups = matches_batches[0].matches.shape[1]
    out = np.full((n_batches, n_groups), np.nan, dtype=float)

    for b in range(n_batches):
        altered_mask = np.asarray(altered_list[b], dtype=bool)
        M = matches_batches[b].matches   # sparse matrix [n_examples, n_groups]

        subgroup_sizes = np.asarray(M.sum(axis=0)).ravel().astype(float)
        altered_counts = np.asarray(M[altered_mask].sum(axis=0)).ravel().astype(float)

        frac = np.zeros_like(subgroup_sizes, dtype=float)
        valid = subgroup_sizes > 0
        frac[valid] = altered_counts[valid] / subgroup_sizes[valid]
        frac[~valid] = np.nan

        out[b] = frac

    return out
def window_scores(metric_mat, altered_frac_mat, win_size=5):
    """
    For each valid window endpoint, compute:
      - gt relevance = mean altered fraction in current window
      - delta = mean(curr) - mean(ref)
      - tstat = -(delta / se), so bigger means stronger degradation
    Returns lists of vectors, one per window.
    """
    n_batches, n_groups = metric_mat.shape

    GT = []
    DELTA = []
    TSTAT = []

    for end in range(2 * win_size - 1, n_batches):
        ref = metric_mat[end - 2 * win_size + 1 : end - win_size + 1, :]
        cur = metric_mat[end - win_size + 1 : end + 1, :]

        print(
            f"{name}: len(divs)={len(payload['divs'])}, "
            f"len(altered)={len(payload['altered'])}, "
            f"len(matches_batches)={len(matches_batches)}"
        )
        gt_cur = altered_frac_mat[end - win_size + 1 : end + 1, :]

        ref_mean = np.nanmean(ref, axis=0)
        cur_mean = np.nanmean(cur, axis=0)
        delta = cur_mean - ref_mean

        ref_std = np.nanstd(ref, axis=0, ddof=1)
        cur_std = np.nanstd(cur, axis=0, ddof=1)

        n_ref = np.sum(~np.isnan(ref), axis=0)
        n_cur = np.sum(~np.isnan(cur), axis=0)

        se = np.sqrt((ref_std ** 2) / np.maximum(n_ref, 1) + (cur_std ** 2) / np.maximum(n_cur, 1))
        tstat = np.zeros_like(delta, dtype=float)

        valid = se > 0
        # positive tstat means degradation (current lower than reference)
        tstat[valid] = -delta[valid] / se[valid]
        tstat[~valid] = 0.0

        gt = np.nanmean(gt_cur, axis=0)
        gt = np.nan_to_num(gt, nan=0.0)
        delta = np.nan_to_num(delta, nan=0.0)
        tstat = np.nan_to_num(tstat, nan=0.0)

        GT.append(gt)
        DELTA.append(delta)
        TSTAT.append(tstat)

    return GT, DELTA, TSTAT


def compute_table_rows(all_GT, all_delta, all_tstat, compute_corr=False):
    """
    Build summary rows for delta / tstat / random.
    """
    table = {}

    for method in ["delta", "tstat", "random"]:
        if method == "delta":
            y_score = -all_delta  # more negative delta = more drift
        elif method == "tstat":
            y_score = all_tstat   # larger positive tstat = more drift
        elif method == "random":
            rng = np.random.default_rng(42)
            y_score = rng.random(all_GT.shape)
        else:
            raise ValueError("Invalid method")

        for k in [None, 10, 100]:
            scores = _ndcg_sample_scores(all_GT, y_score, k=k)
            key = "nDCG" if k is None else f"nDCG@{k}"
            table[(key, method)] = f"{scores.mean():.4f} ± {scores.std():.4f}"

        if compute_corr:
            # Pearson
            gt_centered = all_GT - all_GT.mean(axis=1, keepdims=True)
            ys_centered = y_score - y_score.mean(axis=1, keepdims=True)
            denom = all_GT.std(axis=1) * y_score.std(axis=1)
            pearson = np.divide(
                (gt_centered * ys_centered).mean(axis=1),
                denom,
                out=np.zeros_like(denom),
                where=denom > 0
            )
            table[("Pearson", method)] = f"{pearson.mean():.4f} ± {pearson.std():.4f}"

            # Spearman
            sp = spearmanr(all_GT, y_score, axis=1).statistic
            # extract diagonal cross-block
            n = all_GT.shape[0]
            spearman = np.diagonal(sp, offset=n)
            table[("Spearman", method)] = f"{spearman.mean():.4f} ± {spearman.std():.4f}"

    return table


# ----------------------------
# Load matches file
# ----------------------------
matches_path = os.path.join(os.path.abspath(ckpt_dir), f"matches-{checkpoint}.pkl")
if not os.path.exists(matches_path):
    raise FileNotFoundError(f"Missing matches file: {matches_path}")

with open(matches_path, "rb") as f:
    matches_obj = pickle.load(f)

fi_df = matches_obj["matches_train"].fi
matches_batches = matches_obj["matches_batches"]

subgroup_to_idx, subgroup_to_support = build_fi_maps(fi_df)
n_groups = len(fi_df)

print("Loaded matches:", matches_path)
print("Number of monitored subgroups:", n_groups)


# ----------------------------
# Collect experiment files
# ----------------------------
root_ckpt = os.path.abspath(ckpt_dir)

if source_mode == "supwise":
    pattern = os.path.join(root_ckpt, "sup-wise", f"{checkpoint}-noise-{noise:.2f}-support-*-target-*.pkl")
elif source_mode == "drifteval":
    pattern = os.path.join(root_ckpt, f"{checkpoint}-accuracy-noise-{noise:.2f}", "target-*.pkl")
else:
    raise ValueError("source_mode must be 'supwise' or 'drifteval'")

files = sorted(glob.glob(pattern))
print("Source mode:", source_mode)
print("File pattern:", pattern)
print("Files found:", len(files))

if len(files) == 0:
    raise FileNotFoundError(f"No files found for pattern: {pattern}")


# ----------------------------
# Build per-support experiment collections
# ----------------------------
per_support_GT = {}
per_support_DELTA = {}
per_support_TSTAT = {}

for path in files:
    name = os.path.basename(path)

    with open(path, "rb") as f:
        payload = pickle.load(f)

    target_sg = normalize_itemset(payload["subgroup"])

    if source_mode == "supwise":
        support_key = parse_support_bucket_from_supwise_name(name)
        if support_key is None:
            continue
    else:
        # use exact support rounded into a string key if not using sup-wise
        support_val = subgroup_to_support.get(target_sg, None)
        if support_val is None:
            continue
        support_key = round(float(support_val), 4)

    metric_mat = divs_to_metric_matrix(
        payload["divs"],
        subgroup_to_idx=subgroup_to_idx,
        n_groups=n_groups,
        metric=metric_name
    )

    altered_frac_mat = altered_masks_to_fraction_matrix(
        payload["altered"],
        matches_batches=matches_batches
    )

    n_common = min(metric_mat.shape[0], altered_frac_mat.shape[0])
    metric_mat = metric_mat[:n_common]
    altered_frac_mat = altered_frac_mat[:n_common]

    GT_list, DELTA_list, TSTAT_list = window_scores(
        metric_mat=metric_mat,
        altered_frac_mat=altered_frac_mat,
        win_size=win_size
    )

    if support_key not in per_support_GT:
        per_support_GT[support_key] = []
        per_support_DELTA[support_key] = []
        per_support_TSTAT[support_key] = []

    per_support_GT[support_key].extend(GT_list)
    per_support_DELTA[support_key].extend(DELTA_list)
    per_support_TSTAT[support_key].extend(TSTAT_list)


# ----------------------------
# Sample up to n_samples per support bucket
# ----------------------------
support_keys = sorted(per_support_GT.keys(), key=lambda x: x[0] if isinstance(x, tuple) else x)

all_GT_blocks = []
all_delta_blocks = []
all_tstat_blocks = []

rng = np.random.default_rng(42)

for key in support_keys:
    n = len(per_support_GT[key])
    if n == 0:
        continue

    idx = np.arange(n)
    if n > n_samples:
        idx = rng.choice(idx, size=n_samples, replace=False)

    GT_block = np.vstack([per_support_GT[key][i] for i in idx])
    D_block = np.vstack([per_support_DELTA[key][i] for i in idx])
    T_block = np.vstack([per_support_TSTAT[key][i] for i in idx])

    all_GT_blocks.append(GT_block)
    all_delta_blocks.append(D_block)
    all_tstat_blocks.append(T_block)

if not all_GT_blocks:
    raise RuntimeError("No valid experiment windows were collected.")

all_GT = np.vstack(all_GT_blocks)
all_delta = np.vstack(all_delta_blocks)
all_tstat = np.vstack(all_tstat_blocks)

print("All GT shape:", all_GT.shape)
print("All delta shape:", all_delta.shape)
print("All tstat shape:", all_tstat.shape)


# ----------------------------
# Compute summary table
# ----------------------------
summary = compute_table_rows(
    all_GT=all_GT,
    all_delta=all_delta,
    all_tstat=all_tstat,
    compute_corr=compute_corr
)

df = pd.DataFrame({"adult": summary}).sort_index(level=(0, 1))
print(df.to_latex())




# from glob import glob
# import pickle
# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
# from tqdm import tqdm
# import pandas as pd
# from scipy.sparse import csr_array
# from functools import reduce
# from sklearn.metrics._ranking import _ndcg_sample_scores
# import config
# import sys
#
# sys.path.append("../")
#
# from sklearn.metrics import ndcg_score
# from src.detect import detect_singlebatch, detect_multibatch, _get_altered_in_window
# from src.utils import get_support_bucket
#
# from distill import Result
#
# from scipy.stats import spearmanr
# from tqdm import tqdm
#
# exp_type = "adult"
# n_samples = 50
# N = None
# win_size = 5
# table = {}
# compute_ndcg = True
# compute_corr = False
#
# sup_thresholds = {
#     "adult": 1.0,
#     "celeba": 0.5
# }
#
# for exp_type in ["adult"]:
#     if exp_type == "adult":
#         noise = 0.5
#         checkpoint = "adult_model"
#         ckpt_dir = os.path.abspath(config.ckpt_dir)
#
#         filepath = os.path.join(ckpt_dir, f"{checkpoint}-results-v2.pkl")
#         print("Loading:", filepath)
#         print("Exists:", os.path.exists(filepath))
#
#         with open(filepath, "rb") as f:
#             results = pickle.load(f)
#
#         print("Total results loaded:", len(results))
#         # checkpoint = "xgb-adult"
#         # ckpt_dir = "/data2/fgiobergia/drift-experiments/"
#     else:
#         ckpt_dir = os.path.join(config.ckpt_dir, "sup-wise")
#         noise = 1.0
#         checkpoint = "resnet50"
#
#     data = []
#     for r in results:
#         if r.metric != "accuracy" or r.gt == "neg" or r.support[0] > sup_thresholds[exp_type]:
#             continue
#         d = {
#             "support": r.support[0],
#             "window": r.window,
#             "result": r
#         }
#         data.append(d)
#
#     df_results = pd.DataFrame(data=data)
#
#     results_subset = df_results.groupby(["support", "window"]).apply(
#         lambda gb: gb.sample(n=min(len(gb), n_samples))).result.tolist()
#
#     GT = {}
#     tstats = {}
#     deltas = {}
#     supports = set()
#
#     for r in results_subset:
#         if r.gt != "pos" or r.metric != "accuracy" or r.window != win_size:
#             continue
#         if r.support not in GT:  # assume that , if not in GT, also not in tstats and deltas
#             GT[r.support] = []
#             tstats[r.support] = []
#             deltas[r.support] = []
#         GT[r.support].append(r.altered)
#         tstats[r.support].append(r.tstat)
#         deltas[r.support].append(r.delta)
#         supports.add(r.support)
#     supports = sorted(list(supports))
#
#     GT = [np.vstack(GT[sup]) for sup in supports]
#     tstats = [np.vstack(tstats[sup]) for sup in supports]
#     deltas = [np.vstack(deltas[sup]) for sup in supports]
#
#     all_GT = np.vstack([GT[i] for i in range(len(supports))])
#     all_delta = np.vstack([deltas[i] for i in range(len(supports))])
#     all_tstat = np.vstack([tstats[i] for i in range(len(supports))])
#     all_GT.shape, all_delta.shape, all_tstat.shape
#
#     table[exp_type] = {}
#
#     for method in ["delta", "tstat", "random"]:
#         if method == "delta":
#             y_score = -all_delta
#         elif method == "tstat":
#             y_score = all_tstat
#         elif method == "random":
#             y_score = np.random.random(all_GT.shape)
#         else:
#             raise ValueError("Invalid method")
#
#         # various nDCG metrics
#         if compute_ndcg:
#             for k in [None, 10, 100]:
#                 scores = _ndcg_sample_scores(all_GT[:N], y_score[:N], k=k)
#                 key = "nDCG" if k is None else f"nDCG@{k}"
#                 table[exp_type][(key, method)] = f"{scores.mean():.4f} ± {scores.std():.4f}"
#
#         if compute_corr:
#             p = ((((all_GT[:N] - all_GT[:N].mean(axis=1, keepdims=True)) * (
#                         y_score[:N] - y_score[:N].mean(axis=1, keepdims=True))).mean(axis=1)) / (
#                              all_GT[:N].std(axis=1) * y_score[:N].std(axis=1)))
#             table[exp_type][("Pearson", method)] = f"{p.mean():.4f} ± {p.std():.4f}"
#             print(f"{exp_type} {method} Pearson: {p.mean():.4f} ± {p.std():.4f}")
#
#             s = np.diagonal(spearmanr(all_GT[:N], y_score[:N], axis=1).statistic, offset=all_GT[:N].shape[0])
#             table[exp_type][("Spearman", method)] = f"{s.mean():.4f} ± {s.std():.4f}"
#             print(f"{exp_type} {method} Spearman: {s.mean():.4f} ± {s.std():.4f}")
#
# df = pd.DataFrame(table).sort_index(level=(0, 1))
# print(df.to_latex())