### Updated 4/22
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def build_fi_itemset_cache(fi_df: pd.DataFrame) -> List[Tuple[int, ...]]:
    """
    Precompute normalized subgroup itemsets for the mined subgroup table.
    Call this once per run and reuse it.
    """
    if "itemsets" not in fi_df.columns:
        raise KeyError("fi_df must contain an 'itemsets' column")
    return [normalize_itemset(x) for x in fi_df["itemsets"].tolist()]

def normalize_itemset(x) -> Tuple[int, ...]:
    """Convert subgroup identifiers into a sorted tuple of ints."""
    try:
        return tuple(sorted(int(v) for v in x))
    except Exception:
        return tuple()


def get_metric_values(df: pd.DataFrame, metric: str = "accuracy") -> np.ndarray:
    """
    Extract a metric vector from a div_explorer dataframe.

    Supports:
      - direct metric columns like 'accuracy'
      - confusion matrix columns tp, tn, fp, fn for accuracy
    """
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


def subset_matches(matches_obj, keep_global_idx: Iterable[int]):
    """
    Subset a Matches namedtuple-like object to a chosen set of subgroup columns.

    Expects the Matches object to expose:
      - matches_obj.fi
      - matches_obj.matches
      - matches_obj._replace(...)
    """
    keep = np.asarray(sorted(set(int(x) for x in keep_global_idx)), dtype=int)
    fi_sub = matches_obj.fi.iloc[keep].reset_index(drop=True)
    matches_sub = matches_obj.matches[:, keep]
    return matches_obj._replace(fi=fi_sub, matches=matches_sub)

def divs_to_active_metric_matrix(
    recent_divs: List[pd.DataFrame],
    active_idx: np.ndarray,
    fi_df: pd.DataFrame,
    fi_itemset_cache: Optional[List[Tuple[int, ...]]] = None,
    active_sg_tuples: Optional[List[Tuple[int, ...]]] = None,
    metric: str = "accuracy",
) -> np.ndarray:
    """
    Build a metric matrix [n_recent_batches, n_active_groups] for the currently
    active subgroup set, aligned to the order in active_idx.
    """
    if len(active_idx) == 0:
        return np.zeros((len(recent_divs), 0), dtype=float)

    active_idx = np.asarray(active_idx, dtype=int)

    if fi_itemset_cache is None:
        fi_itemset_cache = build_fi_itemset_cache(fi_df)

    if active_sg_tuples is None:
        active_sg_tuples = [fi_itemset_cache[idx] for idx in active_idx]

    target_pos: Dict[Tuple[int, ...], int] = {
        sg: j for j, sg in enumerate(active_sg_tuples)
    }

    out = np.full((len(recent_divs), len(active_idx)), np.nan, dtype=float)

    for b, df in enumerate(recent_divs):
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        vals = get_metric_values(df, metric)

        subgroup_col = None
        if "subgroup" in df.columns:
            subgroup_col = "subgroup"
        elif "itemsets" in df.columns:
            subgroup_col = "itemsets"

        if subgroup_col is None:
            k = min(len(vals), len(active_idx))
            out[b, :k] = vals[:k]
            continue

        subgroup_values = df[subgroup_col].tolist()
        for row_idx, sg_raw in enumerate(subgroup_values):
            j = target_pos.get(normalize_itemset(sg_raw))
            if j is not None:
                out[b, j] = float(vals[row_idx])

    return out

def compute_recent_scores(
    recent_divs: List[pd.DataFrame],
    active_idx: np.ndarray,
    fi_df: pd.DataFrame,
    fi_itemset_cache: Optional[List[Tuple[int, ...]]] = None,
    win_size: int = 5,
    score_method: str = "abs_tstat",
    metric: str = "accuracy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute subgroup scores using the most recent 2 * win_size divs.

    Returns:
      scores, delta, tstat
    all aligned to active_idx.
    """
    if len(active_idx) == 0:
        z = np.zeros(0, dtype=float)
        return z, z, z

    if len(recent_divs) < 2 * win_size:
        z = np.zeros(len(active_idx), dtype=float)
        return z, z, z

    if fi_itemset_cache is None:
        fi_itemset_cache = build_fi_itemset_cache(fi_df)

    active_idx = np.asarray(active_idx, dtype=int)
    active_sg_tuples = [fi_itemset_cache[idx] for idx in active_idx]

    metric_mat = divs_to_active_metric_matrix(
        recent_divs[-(2 * win_size):],
        active_idx=active_idx,
        fi_df=fi_df,
        fi_itemset_cache=fi_itemset_cache,
        active_sg_tuples=active_sg_tuples,
        metric=metric,
    )

    ref = metric_mat[:win_size, :]
    cur = metric_mat[win_size:, :]

    n_groups = ref.shape[1]
    delta = np.zeros(n_groups, dtype=float)
    tstat = np.zeros(n_groups, dtype=float)

    n_ref = np.sum(np.isfinite(ref), axis=0)
    n_cur = np.sum(np.isfinite(cur), axis=0)

    has_ref = n_ref > 0
    has_cur = n_cur > 0
    valid_mean = has_ref & has_cur

    ref_mean = np.zeros(n_groups, dtype=float)
    cur_mean = np.zeros(n_groups, dtype=float)

    if np.any(has_ref):
        ref_sum = np.nansum(ref, axis=0)
        ref_mean[has_ref] = ref_sum[has_ref] / n_ref[has_ref]

    if np.any(has_cur):
        cur_sum = np.nansum(cur, axis=0)
        cur_mean[has_cur] = cur_sum[has_cur] / n_cur[has_cur]

    delta[valid_mean] = cur_mean[valid_mean] - ref_mean[valid_mean]

    ref_std = np.zeros(n_groups, dtype=float)
    cur_std = np.zeros(n_groups, dtype=float)

    valid_ref_std = n_ref >= 2
    valid_cur_std = n_cur >= 2

    if np.any(valid_ref_std):
        ref_centered = np.where(np.isfinite(ref), ref - ref_mean, np.nan)
        ref_ss = np.nansum(ref_centered ** 2, axis=0)
        ref_std[valid_ref_std] = np.sqrt(ref_ss[valid_ref_std] / (n_ref[valid_ref_std] - 1))

    if np.any(valid_cur_std):
        cur_centered = np.where(np.isfinite(cur), cur - cur_mean, np.nan)
        cur_ss = np.nansum(cur_centered ** 2, axis=0)
        cur_std[valid_cur_std] = np.sqrt(cur_ss[valid_cur_std] / (n_cur[valid_cur_std] - 1))

    se = np.zeros(n_groups, dtype=float)
    se[valid_mean] = np.sqrt(
        (ref_std[valid_mean] ** 2) / np.maximum(n_ref[valid_mean], 1)
        + (cur_std[valid_mean] ** 2) / np.maximum(n_cur[valid_mean], 1)
    )

    valid_t = valid_mean & np.isfinite(se) & (se > 0)
    tstat[valid_t] = -delta[valid_t] / se[valid_t]

    delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    tstat = np.nan_to_num(tstat, nan=0.0, posinf=0.0, neginf=0.0)

    if score_method == "delta":
        scores = np.abs(delta)
    elif score_method == "tstat":
        scores = tstat
    elif score_method == "abs_tstat":
        scores = np.abs(tstat)
    else:
        raise ValueError(f"Unknown score_method: {score_method}")

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores, delta, tstat

def select_active_groups(
    active_idx: np.ndarray,
    scores: np.ndarray,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    min_groups: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select a new active set from the current active set.

    Policy:
      - keep all with score >= threshold, if threshold is set
      - keep top_k by score, if top_k is set
      - use the union of both
      - enforce a minimum floor of min_groups
    """
    if len(active_idx) == 0:
        return np.asarray(active_idx, dtype=int), np.array([], dtype=int)

    active_idx = np.asarray(active_idx, dtype=int)
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Always have a descending order available for floor enforcement
    full_order = np.argsort(scores)[::-1]

    if top_k is not None and top_k > 0:
        k = min(int(top_k), len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        order = top_idx[np.argsort(scores[top_idx])[::-1]]
    else:
        order = np.array([], dtype=int)

    keep_mask = np.zeros(len(active_idx), dtype=bool)

    if threshold is not None:
        keep_mask |= scores >= float(threshold)

    if top_k is not None and top_k > 0:
        keep_mask[order[: min(int(top_k), len(order))]] = True

    if not keep_mask.any():
        keep_mask[full_order[: min(max(1, min_groups), len(full_order))]] = True
    elif keep_mask.sum() < min_groups:
        keep_mask[full_order[: min(min_groups, len(full_order))]] = True

    new_active = active_idx[keep_mask]
    new_active = np.sort(new_active)
    return new_active, full_order

def split_active_inactive(
    universe_idx: np.ndarray,
    active_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted active and inactive subsets from a fixed universe."""
    universe_idx = np.asarray(universe_idx, dtype=int)
    active_idx = np.asarray(sorted(set(int(x) for x in active_idx)), dtype=int)
    mask = np.isin(universe_idx, active_idx)
    inactive_idx = universe_idx[~mask]
    return np.sort(active_idx), np.sort(inactive_idx)


def update_stability_counts(
    active_idx: np.ndarray,
    scores: np.ndarray,
    stability_counts: np.ndarray,
    stable_score_threshold: float,
    stable_rounds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update per-group stability counters for the current active set.

    A group is considered stable on an update if its score is <= threshold.
    Groups that reach stable_rounds consecutive stable updates are returned
    as prune candidates.
    """
    active_idx = np.asarray(active_idx, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(active_idx) == 0:
        return stability_counts, np.array([], dtype=int)

    stable_mask = np.nan_to_num(scores, nan=0.0) <= float(stable_score_threshold)
    unstable_mask = ~stable_mask

    stability_counts[active_idx[stable_mask]] += 1
    stability_counts[active_idx[unstable_mask]] = 0

    if stable_rounds <= 0:
        return stability_counts, np.array([], dtype=int)

    prune_idx = active_idx[stability_counts[active_idx] >= int(stable_rounds)]
    return stability_counts, np.sort(prune_idx)


def refresh_inactive_groups(
    recent_divs: List[pd.DataFrame],
    inactive_idx: np.ndarray,
    fi_df: pd.DataFrame,
    fi_itemset_cache: Optional[List[Tuple[int, ...]]],
    win_size: int,
    score_method: str,
    metric: str,
    refresh_top_k: Optional[int],
    refresh_threshold: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Score the inactive pool and return candidate groups for re-entry.

    Policy:
      - keep all inactive groups with score >= refresh_threshold, if set
      - keep top refresh_top_k inactive groups by score, if set
      - use the union of both
      - if both are None, return no reactivations
    """
    inactive_idx = np.asarray(inactive_idx, dtype=int)
    if len(inactive_idx) == 0:
        z = np.zeros(0, dtype=float)
        return np.array([], dtype=int), z, z, z

    scores, delta, tstat = compute_recent_scores(
        recent_divs=recent_divs,
        active_idx=inactive_idx,
        fi_df=fi_df,
        fi_itemset_cache=fi_itemset_cache,
        win_size=win_size,
        score_method=score_method,
        metric=metric,
    )

    if refresh_top_k is None and refresh_threshold is None:
        return np.array([], dtype=int), scores, delta, tstat

    order = np.argsort(scores)[::-1]
    keep_mask = np.zeros(len(inactive_idx), dtype=bool)

    if refresh_threshold is not None:
        keep_mask |= scores >= float(refresh_threshold)

    if refresh_top_k is not None and refresh_top_k > 0:
        keep_mask[order[: min(int(refresh_top_k), len(order))]] = True

    reactivated_idx = np.sort(inactive_idx[keep_mask])
    return reactivated_idx, scores, delta, tstat

### Updated 4/20
# from __future__ import annotations
#
# from typing import Dict, Iterable, List, Optional, Tuple
#
# import numpy as np
# import pandas as pd
#
#
# def normalize_itemset(x) -> Tuple[int, ...]:
#     """Convert subgroup identifiers into a sorted tuple of ints."""
#     try:
#         return tuple(sorted(int(v) for v in x))
#     except Exception:
#         return tuple()
#
#
# def get_metric_values(df: pd.DataFrame, metric: str = "accuracy") -> np.ndarray:
#     """
#     Extract a metric vector from a div_explorer dataframe.
#
#     Supports:
#       - direct metric columns like 'accuracy'
#       - confusion matrix columns tp, tn, fp, fn for accuracy
#     """
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
# def subset_matches(matches_obj, keep_global_idx: Iterable[int]):
#     """
#     Subset a Matches namedtuple-like object to a chosen set of subgroup columns.
#
#     Expects the Matches object to expose:
#       - matches_obj.fi
#       - matches_obj.matches
#       - matches_obj._replace(...)
#     """
#     keep = np.asarray(sorted(set(int(x) for x in keep_global_idx)), dtype=int)
#     fi_sub = matches_obj.fi.iloc[keep].reset_index(drop=True)
#     matches_sub = matches_obj.matches[:, keep]
#     return matches_obj._replace(fi=fi_sub, matches=matches_sub)
#
#
# def divs_to_active_metric_matrix(
#     recent_divs: List[pd.DataFrame],
#     active_idx: np.ndarray,
#     fi_df: pd.DataFrame,
#     metric: str = "accuracy",
# ) -> np.ndarray:
#     """
#     Build a metric matrix [n_recent_batches, n_active_groups] for the currently
#     active subgroup set, aligned to the order in active_idx.
#     """
#     if len(active_idx) == 0:
#         return np.zeros((len(recent_divs), 0), dtype=float)
#
#     target_sgs = [normalize_itemset(fi_df.iloc[idx]["itemsets"]) for idx in active_idx]
#     out = np.full((len(recent_divs), len(active_idx)), np.nan, dtype=float)
#
#     for b, df in enumerate(recent_divs):
#         if not isinstance(df, pd.DataFrame) or df.empty:
#             continue
#
#         vals = get_metric_values(df, metric=metric)
#
#         subgroup_col = None
#         if "subgroup" in df.columns:
#             subgroup_col = "subgroup"
#         elif "itemsets" in df.columns:
#             subgroup_col = "itemsets"
#
#         if subgroup_col is None:
#             # Fallback: assume row order matches current active set order
#             k = min(len(vals), len(active_idx))
#             out[b, :k] = vals[:k]
#             continue
#
#         row_map: Dict[Tuple[int, ...], float] = {}
#         for row_idx, (_, row) in enumerate(df.iterrows()):
#             sg = normalize_itemset(row[subgroup_col])
#             row_map[sg] = float(vals[row_idx])
#
#         for j, sg in enumerate(target_sgs):
#             if sg in row_map:
#                 out[b, j] = row_map[sg]
#
#     return out
#
#
# def compute_recent_scores(
#     recent_divs: List[pd.DataFrame],
#     active_idx: np.ndarray,
#     fi_df: pd.DataFrame,
#     win_size: int = 5,
#     score_method: str = "abs_tstat",
#     metric: str = "accuracy",
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute subgroup scores using the most recent 2 * win_size divs.
#
#     Returns:
#       scores, delta, tstat
#     all aligned to active_idx.
#     """
#     if len(active_idx) == 0:
#         z = np.zeros(0, dtype=float)
#         return z, z, z
#
#     if len(recent_divs) < 2 * win_size:
#         z = np.zeros(len(active_idx), dtype=float)
#         return z, z, z
#
#     metric_mat = divs_to_active_metric_matrix(
#         recent_divs[-(2 * win_size):],
#         active_idx=active_idx,
#         fi_df=fi_df,
#         metric=metric,
#     )
#
#     ref = metric_mat[:win_size, :]
#     cur = metric_mat[win_size:, :]
#
#     ref_mean = np.nanmean(ref, axis=0)
#     cur_mean = np.nanmean(cur, axis=0)
#     delta = cur_mean - ref_mean
#
#     ref_std = np.nanstd(ref, axis=0, ddof=1)
#     cur_std = np.nanstd(cur, axis=0, ddof=1)
#
#     n_ref = np.sum(~np.isnan(ref), axis=0)
#     n_cur = np.sum(~np.isnan(cur), axis=0)
#
#     se = np.sqrt((ref_std ** 2) / np.maximum(n_ref, 1) + (cur_std ** 2) / np.maximum(n_cur, 1))
#     tstat = np.zeros_like(delta, dtype=float)
#     valid = se > 0
#     tstat[valid] = -delta[valid] / se[valid]  # positive means degradation
#
#     delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
#     tstat = np.nan_to_num(tstat, nan=0.0, posinf=0.0, neginf=0.0)
#
#     if score_method == "delta":
#         scores = np.abs(delta)
#     elif score_method == "tstat":
#         scores = tstat
#     elif score_method == "abs_tstat":
#         scores = np.abs(tstat)
#     else:
#         raise ValueError(f"Unknown score_method: {score_method}")
#
#     scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
#     return scores, delta, tstat
#
#
# def select_active_groups(
#     active_idx: np.ndarray,
#     scores: np.ndarray,
#     top_k: Optional[int] = None,
#     threshold: Optional[float] = None,
#     min_groups: int = 50,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Select a new active set from the current active set.
#
#     Policy:
#       - keep all with score >= threshold, if threshold is set
#       - keep top_k by score, if top_k is set
#       - use the union of both
#       - enforce a minimum floor of min_groups
#     """
#     if len(active_idx) == 0:
#         return active_idx, np.array([], dtype=int)
#
#     scores = np.asarray(scores, dtype=float)
#     scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
#
#     order = np.argsort(-scores)  # descending
#     keep_mask = np.zeros(len(active_idx), dtype=bool)
#
#     if threshold is not None:
#         keep_mask |= scores >= threshold
#
#     if top_k is not None and top_k > 0:
#         keep_mask[order[: min(top_k, len(order))]] = True
#
#     # If nothing selected, keep at least the best min_groups
#     if not keep_mask.any():
#         keep_mask[order[: min(max(1, min_groups), len(order))]] = True
#     elif keep_mask.sum() < min_groups:
#         keep_mask[order[: min(min_groups, len(order))]] = True
#
#     new_active = np.asarray(active_idx, dtype=int)[keep_mask]
#     new_active = np.sort(new_active)
#     return new_active, order