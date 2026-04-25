#!/usr/bin/env python3
#Updated 4/22
from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.divexp import div_explorer
from src.adult.config import ckpt_dir


from src.adult.adaptive_select import (
    build_fi_itemset_cache,
    compute_recent_scores,
    normalize_itemset,
    refresh_inactive_groups,
    select_active_groups,
    split_active_inactive,
    subset_matches,
    update_stability_counts,
)

ADAPTIVE_MODES = ("active_only", "stability", "refresh")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def build_support_buckets(fi_df: pd.DataFrame, n_bins: int = 13) -> List[Tuple[float, float, List[int]]]:
    q = min(n_bins, int(fi_df["support"].nunique()))
    if q <= 0:
        return []

    cats = pd.qcut(fi_df["support"], q=q, duplicates="drop")
    buckets = []

    for interval in cats.cat.categories:
        mask = cats == interval
        idx = fi_df.index[mask].tolist()
        if not idx:
            continue
        lo = float(round(interval.left, 4))
        hi = float(round(interval.right, 4))
        buckets.append((lo, hi, idx))

    return buckets


def make_noise_schedule(n_batches: int, start_noise: int, transitory: int, frac_noise: float) -> np.ndarray:
    noise_fracs = np.zeros(n_batches, dtype=float)
    assert start_noise + transitory < n_batches, "Noise schedule extends beyond the available batches."
    noise_fracs[start_noise : start_noise + transitory] = np.linspace(0, frac_noise, transitory)
    noise_fracs[start_noise + transitory :] = frac_noise
    return noise_fracs

def append_manifest_row(csv_path: Path, row: Dict):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def load_completed_outputs(manifest_csv: Path) -> set[str]:
    if not manifest_csv.exists():
        return set()
    try:
        df = pd.read_csv(manifest_csv)
    except Exception:
        return set()
    if "outfile" not in df.columns:
        return set()
    return set(df["outfile"].astype(str))

def log_progress(txt_path: Path, msg: str):
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def initialize_adaptive_state(n_groups: int) -> Dict[str, np.ndarray]:
    return {
        "active_idx": np.arange(n_groups, dtype=int),
        "inactive_idx": np.array([], dtype=int),
        "stability_counts": np.zeros(n_groups, dtype=int),
        "last_scores": np.zeros(n_groups, dtype=float),
        "reactivation_counts": np.zeros(n_groups, dtype=int),
        "last_seen_active_round": np.full(n_groups, -1, dtype=int),
    }

def _score_map_from_indices(indices: np.ndarray, scores: np.ndarray) -> Dict[int, float]:
    return {int(i): float(s) for i, s in zip(np.asarray(indices, dtype=int), np.asarray(scores, dtype=float))}


def _scores_for_indices(indices: np.ndarray, score_map: Dict[int, float]) -> np.ndarray:
    return np.asarray([float(score_map.get(int(i), 0.0)) for i in np.asarray(indices, dtype=int)], dtype=float)

def apply_adaptive_update(
    *,
    adaptive_state: Dict[str, np.ndarray],
    recent_divs: List[pd.DataFrame],
    fi_df: pd.DataFrame,
    fi_itemset_cache: List[Tuple[int, ...]],
    all_group_idx: np.ndarray,
    metric: str,
    win_size: int,
    update_round: int,
    mode: str,
    top_k: Optional[int],
    threshold: Optional[float],
    min_groups: int,
    score_method: str,
    stable_score_threshold: float,
    stable_rounds: int,
    refresh_interval: int,
    refresh_top_k: Optional[int],
    refresh_threshold: Optional[float],
    light_output: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    if mode not in ADAPTIVE_MODES:
        raise ValueError(f"Unknown adaptive mode: {mode}")

    active_idx_before = np.asarray(adaptive_state["active_idx"], dtype=int)

    scores, delta, tstat = compute_recent_scores(
        recent_divs=recent_divs,
        active_idx=active_idx_before,
        fi_df=fi_df,
        fi_itemset_cache=fi_itemset_cache,
        win_size=win_size,
        score_method=score_method,
        metric=metric,
    )

    adaptive_state["last_scores"][active_idx_before] = scores
    adaptive_state["last_seen_active_round"][active_idx_before] = int(update_round)

    score_map = _score_map_from_indices(active_idx_before, scores)
    score_pruned_idx = np.array([], dtype=int)
    stability_pruned_idx = np.array([], dtype=int)
    reactivated_idx = np.array([], dtype=int)
    refresh_triggered = False
    refresh_scores = np.zeros(0, dtype=float)

    candidate_active = active_idx_before.copy()
    candidate_scores = scores.copy()

    if mode in {"stability", "refresh"}:
        adaptive_state["stability_counts"], stability_pruned_idx = update_stability_counts(
            active_idx=active_idx_before,
            scores=scores,
            stability_counts=adaptive_state["stability_counts"],
            stable_score_threshold=stable_score_threshold,
            stable_rounds=stable_rounds,
        )
        if len(stability_pruned_idx) > 0:
            keep_mask = ~np.isin(candidate_active, stability_pruned_idx)
            candidate_active = candidate_active[keep_mask]
            candidate_scores = candidate_scores[keep_mask]

    selected_active, _ = select_active_groups(
        active_idx=candidate_active,
        scores=candidate_scores,
        top_k=top_k,
        threshold=threshold,
        min_groups=min_groups,
    )
    score_pruned_idx = np.setdiff1d(candidate_active, selected_active, assume_unique=False)
    active_after_score = selected_active

    if mode == "refresh" and refresh_interval > 0 and (update_round % refresh_interval == 0):
        refresh_triggered = True
        temp_active, temp_inactive = split_active_inactive(all_group_idx, active_after_score)
        reactivated_idx, refresh_scores, _, _ = refresh_inactive_groups(
            recent_divs=recent_divs,
            inactive_idx=temp_inactive,
            fi_df=fi_df,
            fi_itemset_cache=fi_itemset_cache,
            win_size=win_size,
            score_method=score_method,
            metric=metric,
            refresh_top_k=refresh_top_k,
            refresh_threshold=refresh_threshold,
        )
        if len(reactivated_idx) > 0:
            refresh_map = _score_map_from_indices(temp_inactive, refresh_scores)

            adaptive_state["last_scores"][reactivated_idx] = np.array(
                [refresh_map[int(i)] for i in reactivated_idx], dtype=float
            )
            adaptive_state["reactivation_counts"][reactivated_idx] += 1
            adaptive_state["stability_counts"][reactivated_idx] = 0
            adaptive_state["last_seen_active_round"][reactivated_idx] = int(update_round)

            merged_active = np.union1d(active_after_score, reactivated_idx).astype(int)
            merged_score_map = dict(score_map)
            merged_score_map.update(refresh_map)
            merged_scores = _scores_for_indices(merged_active, merged_score_map)

            active_after_score, _ = select_active_groups(
                active_idx=merged_active,
                scores=merged_scores,
                top_k=top_k,
                threshold=threshold,
                min_groups=min_groups,
            )

    active_idx_after, inactive_idx_after = split_active_inactive(all_group_idx, active_after_score)
    pruned_idx = np.setdiff1d(active_idx_before, active_idx_after, assume_unique=False)

    adaptive_state["active_idx"] = active_idx_after
    adaptive_state["inactive_idx"] = inactive_idx_after
    adaptive_state["stability_counts"][pruned_idx] = 0

    top_score = float(scores[np.argmax(scores)]) if len(scores) > 0 else 0.0
    mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
    stability_snapshot = adaptive_state["stability_counts"].copy()

    update_info: Dict[str, object] = {
        "update_round": int(update_round),
        "before_count": int(len(active_idx_before)),
        "after_count": int(len(active_idx_after)),
        "inactive_after_count": int(len(inactive_idx_after)),
        "mode": mode,
        "score_method": score_method,
        "top_score": top_score,
        "mean_score": mean_score,
        "stable_score_threshold": float(stable_score_threshold),
        "stable_rounds": int(stable_rounds),
        "score_pruned_count": int(len(score_pruned_idx)),
        "stability_pruned_count": int(len(stability_pruned_idx)),
        "reactivated_count": int(len(reactivated_idx)),
        "refresh_triggered": bool(refresh_triggered),
        "max_stability_count": int(np.max(stability_snapshot)) if len(stability_snapshot) > 0 else 0,
        "num_nonzero_stability": int(np.count_nonzero(stability_snapshot)),
    }

    if not light_output:
        update_info.update(
            {
                "selected_global_indices": active_idx_after.tolist(),
                "pruned_global_indices": pruned_idx.tolist(),
                "score_pruned_global_indices": score_pruned_idx.tolist(),
                "stability_pruned_global_indices": stability_pruned_idx.tolist(),
                "reactivated_global_indices": reactivated_idx.tolist(),
            }
        )

    return adaptive_state, update_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive subgroup injection for Adult DriftInspector.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--metric", type=str, choices=["accuracy"], default="accuracy")
    parser.add_argument("--frac-noise", type=float, default=0.5)
    parser.add_argument("--n-targets", type=int, default=10)
    parser.add_argument("--n-support-buckets", type=int, default=13)
    parser.add_argument("--start-noise", type=int, default=10)
    parser.add_argument("--transitory", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--adaptive-enable", action="store_true")
    parser.add_argument("--adaptive-top-k", type=int, default=None)
    parser.add_argument("--adaptive-threshold", type=float, default=None)
    parser.add_argument("--adaptive-interval", type=int, default=15)
    parser.add_argument("--adaptive-score", choices=["delta", "tstat", "abs_tstat"], default="abs_tstat")
    parser.add_argument("--adaptive-min-groups", type=int, default=100)
    parser.add_argument("--win-size", type=int, default=5)
    parser.add_argument("--adaptive-mode", choices=list(ADAPTIVE_MODES), default="active_only")
    parser.add_argument("--stable-score-threshold", type=float, default=0.0)
    parser.add_argument("--stable-rounds", type=int, default=2)
    parser.add_argument("--refresh-interval", type=int, default=3)
    parser.add_argument("--refresh-top-k", type=int, default=None)
    parser.add_argument("--refresh-threshold", type=float, default=None)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--light-output", action="store_true")

    args = parser.parse_args()

    if args.adaptive_enable and args.adaptive_mode == "active_only":
        if args.adaptive_top_k is None and args.adaptive_threshold is None:
            raise ValueError(
                "With --adaptive-enable and --adaptive-mode active_only, set at least one of "
                "--adaptive-top-k or --adaptive-threshold."
            )

    rng = np.random.default_rng(args.seed)
    ckpt_root = Path(ckpt_dir).resolve()
    supwise_dir = ckpt_root / "sup-wise"
    ensure_dir(supwise_dir)

    progress_csv = supwise_dir / f"{args.checkpoint}-{args.adaptive_mode}-adaptive_progress.csv"
    progress_txt = supwise_dir / f"{args.checkpoint}-{args.adaptive_mode}-adaptive_progress.txt"
    completed_outputs = load_completed_outputs(progress_csv) if args.resume else set()

    model_filename = ckpt_root / f"{args.checkpoint}.pkl"
    ds_filename = ckpt_root / f"{args.checkpoint}.dataset.pkl"
    matches_filename = ckpt_root / f"matches-{args.checkpoint}.pkl"

    if not model_filename.exists():
        raise FileNotFoundError(f"Missing model file: {model_filename}")
    if not ds_filename.exists():
        raise FileNotFoundError(f"Missing dataset file: {ds_filename}")
    if not matches_filename.exists():
        raise FileNotFoundError(f"Missing matches file: {matches_filename}")

    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    with open(ds_filename, "rb") as f:
        ds = pickle.load(f)
        test_sets = ds["test_chunks"]
        preprocessor = ds["transform"]

    with open(matches_filename, "rb") as f:
        matches_obj = pickle.load(f)
        fi_df = matches_obj["matches_train"].fi.reset_index(drop=True)
        fi_itemset_cache = build_fi_itemset_cache(fi_df)
        matches_batches = matches_obj["matches_batches"]
        metadata_batches = matches_obj["metadata_batches"]

    n_groups = len(fi_df)
    all_group_idx = np.arange(n_groups, dtype=int)
    n_batches = len(test_sets)

    buckets = build_support_buckets(fi_df, n_bins=args.n_support_buckets)
    if not buckets:
        raise RuntimeError("No support buckets could be built from the fi dataframe.")

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Total mined subgroups: {n_groups}")
    print(f"Support buckets: {len(buckets)}")
    print(f"Adaptive enabled: {args.adaptive_enable}")
    print(f"Adaptive mode: {args.adaptive_mode}")
    print(f"Light output: {args.light_output}")

    completed_counter = 0

    for support_low, support_high, bucket_idx in tqdm(buckets, desc="Support buckets"):
        if len(bucket_idx) == 0:
            continue

        sample_size = min(args.n_targets, len(bucket_idx))
        chosen_rows = rng.choice(bucket_idx, size=sample_size, replace=False)

        for row_idx in tqdm(chosen_rows, desc=f"Targets {support_low:.4f}-{support_high:.4f}", leave=False):
            start_time = time.perf_counter()

            target_row = fi_df.iloc[int(row_idx)]
            target_sg = normalize_itemset(target_row["itemsets"])
            target_support = float(target_row["support"])

            out_name = (
                f"{args.checkpoint}-mode-{args.adaptive_mode}-noise-{args.frac_noise:.2f}-"
                f"support-{support_low:.4f}-{support_high:.4f}-"
                f"target-{'-'.join(map(str, target_sg))}.pkl"
            )
            out_path = supwise_dir / out_name

            if args.resume and (str(out_path) in completed_outputs or out_path.exists()):
                log_progress(progress_txt, f"[SKIP] {out_path}")
                continue

            noise_fracs = make_noise_schedule(
                n_batches=n_batches,
                start_noise=args.start_noise,
                transitory=args.transitory,
                frac_noise=args.frac_noise,
            )

            accuracies: List[float] = []
            f1_scores: List[float] = []
            divs: List[pd.DataFrame] = []
            altered_masks: List[np.ndarray] = []

            y_trues = [] if not args.light_output else None
            y_preds = [] if not args.light_output else None

            adaptive_state = initialize_adaptive_state(n_groups)
            active_group_counts: List[int] = []
            inactive_group_counts: List[int] = []
            active_group_indices_per_batch = [] if not args.light_output else None
            adaptive_updates: List[Dict[str, object]] = []
            reactivated_group_counts: List[int] = []
            refresh_events: List[Dict[str, object]] = []
            stability_counts_per_update: List[Dict[str, int]] = []

            batch_iter = tqdm(
                zip(test_sets, noise_fracs, matches_batches, metadata_batches),
                total=n_batches,
                desc="Batches",
                leave=False,
            )

            update_round = 0

            for pos, (ts, noise_frac, matches_ts, df_ohe) in enumerate(batch_iter):
                X = preprocessor.transform(ts.drop(columns=["target"]))
                y_pred = model.predict(X)
                y_true = np.copy(ts["target"].values)

                subgroup_cols = list(target_sg)
                target_mask = df_ohe.values[:, subgroup_cols].sum(axis=1) == len(subgroup_cols)
                noise_mask = rng.random(len(y_true)) < noise_frac
                flip_mask = target_mask & noise_mask

                if flip_mask.sum() > 0:
                    y_true[flip_mask] = 1 - y_true[flip_mask]

                altered_masks.append(flip_mask)

                if not args.light_output:
                    y_trues.append(y_true)
                    y_preds.append(y_pred)

                current_active_idx = np.asarray(adaptive_state["active_idx"], dtype=int)
                if args.adaptive_enable:
                    active_matches_ts = subset_matches(matches_ts, current_active_idx)
                else:
                    active_matches_ts = matches_ts

                div_df = div_explorer(active_matches_ts, y_true, y_pred, [args.metric])
                divs.append(div_df)

                batch_acc = accuracy_score(y_true, y_pred)
                batch_f1 = f1_score(y_true, y_pred, zero_division=0)
                accuracies.append(batch_acc)
                f1_scores.append(batch_f1)

                active_group_counts.append(int(len(current_active_idx)))
                inactive_group_counts.append(int(len(adaptive_state["inactive_idx"])))
                if not args.light_output:
                    active_group_indices_per_batch.append(current_active_idx.tolist())

                batch_iter.set_postfix(
                    altered=int(flip_mask.sum()),
                    active=len(current_active_idx),
                    acc=f"{batch_acc:.3f}",
                )

                enough_history = len(divs) >= 2 * args.win_size
                at_update_boundary = (pos + 1) % args.adaptive_interval == 0

                if args.adaptive_enable and enough_history and at_update_boundary:
                    update_round += 1
                    recent_divs = divs[-(2 * args.win_size):]
                    adaptive_state, update_info = apply_adaptive_update(
                        adaptive_state=adaptive_state,
                        recent_divs=recent_divs,
                        fi_df=fi_df,
                        fi_itemset_cache=fi_itemset_cache,
                        all_group_idx=all_group_idx,
                        metric=args.metric,
                        win_size=args.win_size,
                        update_round=update_round,
                        mode=args.adaptive_mode,
                        top_k=args.adaptive_top_k,
                        threshold=args.adaptive_threshold,
                        min_groups=args.adaptive_min_groups,
                        score_method=args.adaptive_score,
                        stable_score_threshold=args.stable_score_threshold,
                        stable_rounds=args.stable_rounds,
                        refresh_interval=args.refresh_interval,
                        refresh_top_k=args.refresh_top_k,
                        refresh_threshold=args.refresh_threshold,
                        light_output=args.light_output,
                    )
                    adaptive_updates.append(update_info)
                    reactivated_group_counts.append(int(update_info["reactivated_count"]))
                    stability_counts_per_update.append(
                        {
                            "update_round": int(update_info["update_round"]),
                            "max_stability_count": int(update_info["max_stability_count"]),
                            "num_nonzero_stability": int(update_info["num_nonzero_stability"]),
                        }
                    )
                    if bool(update_info["refresh_triggered"]):
                        refresh_events.append(
                            {
                                "update_round": int(update_info["update_round"]),
                                "reactivated_count": int(update_info["reactivated_count"]),
                                "inactive_after_count": int(update_info["inactive_after_count"]),
                            }
                        )

            payload: Dict[str, object] = {
                "subgroup": target_sg,
                "accuracies": accuracies,
                "f1": f1_scores,
                "divs": divs,
                "noise_fracs": noise_fracs,
                "altered": altered_masks,
                "support_bucket": (support_low, support_high),
                "target_support": target_support,
                "full_group_count": int(n_groups),
                "adaptive_enabled": bool(args.adaptive_enable),
                "adaptive_mode": args.adaptive_mode,
                "adaptive_params": {
                    "adaptive_top_k": args.adaptive_top_k,
                    "adaptive_threshold": args.adaptive_threshold,
                    "adaptive_interval": args.adaptive_interval,
                    "adaptive_score": args.adaptive_score,
                    "adaptive_min_groups": args.adaptive_min_groups,
                    "win_size": args.win_size,
                    "light_output": args.light_output,
                    "adaptive_mode": args.adaptive_mode,
                    "stable_score_threshold": args.stable_score_threshold,
                    "stable_rounds": args.stable_rounds,
                    "refresh_interval": args.refresh_interval,
                    "refresh_top_k": args.refresh_top_k,
                    "refresh_threshold": args.refresh_threshold,
                },
                "active_group_counts": active_group_counts,
                "inactive_group_counts": inactive_group_counts,
                "adaptive_updates": adaptive_updates,
                "reactivated_group_counts": reactivated_group_counts,
                "refresh_events": refresh_events,
                "stability_counts_per_update": stability_counts_per_update,
            }

            if not args.light_output:
                payload["y_trues"] = y_trues
                payload["y_preds"] = y_preds
                payload["active_group_indices_per_batch"] = active_group_indices_per_batch

            with open(out_path, "wb") as f:
                pickle.dump(payload, f)

            elapsed = time.perf_counter() - start_time
            manifest_row = {
                "checkpoint": args.checkpoint,
                "outfile": str(out_path),
                "noise": args.frac_noise,
                "support_low": support_low,
                "support_high": support_high,
                "target": "-".join(map(str, target_sg)),
                "target_support": target_support,
                "adaptive_enabled": bool(args.adaptive_enable),
                "adaptive_mode": args.adaptive_mode,
                "full_group_count": int(n_groups),
                "avg_active_groups": float(np.mean(active_group_counts)) if active_group_counts else None,
                "min_active_groups": int(np.min(active_group_counts)) if active_group_counts else None,
                "max_active_groups": int(np.max(active_group_counts)) if active_group_counts else None,
                "avg_inactive_groups": float(np.mean(inactive_group_counts)) if inactive_group_counts else None,
                "elapsed_sec": elapsed,
                "light_output": args.light_output,
                "status": "completed",
            }
            append_manifest_row(progress_csv, manifest_row)
            completed_outputs.add(str(out_path))

            completed_counter += 1
            if completed_counter % args.save_every == 0:
                log_progress(progress_txt, f"[SAVE] completed {completed_counter} targets so far")

    log_progress(progress_txt, "[DONE] adaptive injection completed")

#
# #!/usr/bin/env python3
# #Updated 4/20
# from __future__ import annotations
#
# import argparse
# import csv
# import os
# import pickle
# import time
# from pathlib import Path
# from typing import Dict, List, Tuple
#
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, f1_score
# from tqdm.auto import tqdm
#
# import sys
# HERE = Path(__file__).resolve().parent
# ROOT = HERE.parent.parent
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))
#
# from src.divexp import div_explorer
# from src.adult.config import ckpt_dir
# from src.adult.adaptive_select import (
#     compute_recent_scores,
#     normalize_itemset,
#     select_active_groups,
#     subset_matches,
# )
#
#
# def ensure_dir(path: Path) -> None:
#     path.mkdir(parents=True, exist_ok=True)
#
#
# def build_support_buckets(fi_df: pd.DataFrame, n_bins: int = 13) -> List[Tuple[float, float, List[int]]]:
#     q = min(n_bins, int(fi_df["support"].nunique()))
#     if q <= 0:
#         return []
#
#     cats = pd.qcut(fi_df["support"], q=q, duplicates="drop")
#     buckets = []
#
#     for interval in cats.cat.categories:
#         mask = cats == interval
#         idx = fi_df.index[mask].tolist()
#         if not idx:
#             continue
#         lo = float(round(interval.left, 4))
#         hi = float(round(interval.right, 4))
#         buckets.append((lo, hi, idx))
#
#     return buckets
#
#
# def make_noise_schedule(n_batches: int, start_noise: int, transitory: int, frac_noise: float) -> np.ndarray:
#     noise_fracs = np.zeros(n_batches, dtype=float)
#     assert start_noise + transitory < n_batches, "Noise schedule extends beyond the available batches."
#     noise_fracs[start_noise : start_noise + transitory] = np.linspace(0, frac_noise, transitory)
#     noise_fracs[start_noise + transitory :] = frac_noise
#     return noise_fracs
#
#
# def append_manifest_row(csv_path: Path, row: Dict):
#     write_header = not csv_path.exists()
#     with open(csv_path, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(row.keys()))
#         if write_header:
#             writer.writeheader()
#         writer.writerow(row)
#
#
# def load_completed_outputs(manifest_csv: Path) -> set[str]:
#     if not manifest_csv.exists():
#         return set()
#     try:
#         df = pd.read_csv(manifest_csv)
#     except Exception:
#         return set()
#     if "outfile" not in df.columns:
#         return set()
#     return set(df["outfile"].astype(str))
#
#
# def log_progress(txt_path: Path, msg: str):
#     with open(txt_path, "a", encoding="utf-8") as f:
#         f.write(msg + "\n")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Adaptive subgroup injection for Adult DriftInspector.")
#     parser.add_argument("--checkpoint", type=str, required=True)
#     parser.add_argument("--metric", type=str, choices=["accuracy"], default="accuracy")
#     parser.add_argument("--frac-noise", type=float, default=0.5)
#     parser.add_argument("--n-targets", type=int, default=10)
#     parser.add_argument("--n-support-buckets", type=int, default=13)
#     parser.add_argument("--start-noise", type=int, default=10)
#     parser.add_argument("--transitory", type=int, default=10)
#     parser.add_argument("--seed", type=int, default=42)
#
#     parser.add_argument("--adaptive-enable", action="store_true")
#     parser.add_argument("--adaptive-top-k", type=int, default=None)
#     parser.add_argument("--adaptive-threshold", type=float, default=None)
#     parser.add_argument("--adaptive-interval", type=int, default=5)
#     parser.add_argument("--adaptive-score", choices=["delta", "tstat", "abs_tstat"], default="abs_tstat")
#     parser.add_argument("--adaptive-min-groups", type=int, default=100)
#     parser.add_argument("--win-size", type=int, default=5)
#
#     parser.add_argument("--resume", action="store_true")
#     parser.add_argument("--save-every", type=int, default=10)
#     parser.add_argument("--light-output", action="store_true")
#
#     args = parser.parse_args()
#
#     if args.adaptive_enable and args.adaptive_top_k is None and args.adaptive_threshold is None:
#         raise ValueError("With --adaptive-enable, set at least one of --adaptive-top-k or --adaptive-threshold.")
#
#     rng = np.random.default_rng(args.seed)
#     ckpt_root = Path(ckpt_dir).resolve()
#     supwise_dir = ckpt_root / "sup-wise"
#     ensure_dir(supwise_dir)
#
#     progress_csv = supwise_dir / f"{args.checkpoint}-adaptive_progress.csv"
#     progress_txt = supwise_dir / f"{args.checkpoint}-adaptive_progress.txt"
#     completed_outputs = load_completed_outputs(progress_csv) if args.resume else set()
#
#     model_filename = ckpt_root / f"{args.checkpoint}.pkl"
#     ds_filename = ckpt_root / f"{args.checkpoint}.dataset.pkl"
#     matches_filename = ckpt_root / f"matches-{args.checkpoint}.pkl"
#
#     if not model_filename.exists():
#         raise FileNotFoundError(f"Missing model file: {model_filename}")
#     if not ds_filename.exists():
#         raise FileNotFoundError(f"Missing dataset file: {ds_filename}")
#     if not matches_filename.exists():
#         raise FileNotFoundError(f"Missing matches file: {matches_filename}")
#
#     with open(model_filename, "rb") as f:
#         model = pickle.load(f)
#
#     with open(ds_filename, "rb") as f:
#         ds = pickle.load(f)
#         test_sets = ds["test_chunks"]
#         preprocessor = ds["transform"]
#
#     with open(matches_filename, "rb") as f:
#         matches_obj = pickle.load(f)
#         fi_df = matches_obj["matches_train"].fi.reset_index(drop=True)
#         matches_batches = matches_obj["matches_batches"]
#         metadata_batches = matches_obj["metadata_batches"]
#
#     n_groups = len(fi_df)
#     all_group_idx = np.arange(n_groups, dtype=int)
#     n_batches = len(test_sets)
#
#     buckets = build_support_buckets(fi_df, n_bins=args.n_support_buckets)
#     if not buckets:
#         raise RuntimeError("No support buckets could be built from the fi dataframe.")
#
#     print(f"Loaded checkpoint: {args.checkpoint}")
#     print(f"Total mined subgroups: {n_groups}")
#     print(f"Support buckets: {len(buckets)}")
#     print(f"Adaptive enabled: {args.adaptive_enable}")
#     print(f"Light output: {args.light_output}")
#
#     completed_counter = 0
#
#     for support_low, support_high, bucket_idx in tqdm(buckets, desc="Support buckets"):
#         if len(bucket_idx) == 0:
#             continue
#
#         sample_size = min(args.n_targets, len(bucket_idx))
#         chosen_rows = rng.choice(bucket_idx, size=sample_size, replace=False)
#
#         for row_idx in tqdm(chosen_rows, desc=f"Targets {support_low:.4f}-{support_high:.4f}", leave=False):
#             start_time = time.perf_counter()
#
#             target_row = fi_df.iloc[int(row_idx)]
#             target_sg = normalize_itemset(target_row["itemsets"])
#             target_support = float(target_row["support"])
#
#             out_name = (
#                 f"{args.checkpoint}-noise-{args.frac_noise:.2f}-"
#                 f"support-{support_low:.4f}-{support_high:.4f}-"
#                 f"target-{'-'.join(map(str, target_sg))}.pkl"
#             )
#             out_path = supwise_dir / out_name
#
#             if args.resume and (str(out_path) in completed_outputs or out_path.exists()):
#                 log_progress(progress_txt, f"[SKIP] {out_path}")
#                 continue
#
#             noise_fracs = make_noise_schedule(
#                 n_batches=n_batches,
#                 start_noise=args.start_noise,
#                 transitory=args.transitory,
#                 frac_noise=args.frac_noise,
#             )
#
#             accuracies = []
#             f1_scores = []
#             divs = []
#             altered_masks = []
#
#             y_trues = [] if not args.light_output else None
#             y_preds = [] if not args.light_output else None
#
#             active_idx = all_group_idx.copy()
#             active_group_counts = []
#             active_group_indices_per_batch = [] if not args.light_output else None
#             adaptive_updates = []
#
#             batch_iter = tqdm(
#                 zip(test_sets, noise_fracs, matches_batches, metadata_batches),
#                 total=n_batches,
#                 desc="Batches",
#                 leave=False,
#             )
#
#             for pos, (ts, noise_frac, matches_ts, df_ohe) in enumerate(batch_iter):
#                 X = preprocessor.transform(ts.drop(columns=["target"]))
#                 y_pred = model.predict(X)
#                 y_true = np.copy(ts["target"].values)
#
#                 subgroup_cols = list(target_sg)
#                 target_mask = df_ohe.values[:, subgroup_cols].sum(axis=1) == len(subgroup_cols)
#                 noise_mask = rng.random(len(y_true)) < noise_frac
#                 flip_mask = target_mask & noise_mask
#
#                 if flip_mask.sum() > 0:
#                     y_true[flip_mask] = 1 - y_true[flip_mask]
#
#                 altered_masks.append(flip_mask)
#
#                 if not args.light_output:
#                     y_trues.append(y_true)
#                     y_preds.append(y_pred)
#
#                 if args.adaptive_enable:
#                     active_matches_ts = subset_matches(matches_ts, active_idx)
#                 else:
#                     active_matches_ts = matches_ts
#
#                 div_df = div_explorer(active_matches_ts, y_true, y_pred, [args.metric])
#                 divs.append(div_df)
#
#                 batch_acc = accuracy_score(y_true, y_pred)
#                 batch_f1 = f1_score(y_true, y_pred)
#                 accuracies.append(batch_acc)
#                 f1_scores.append(batch_f1)
#
#                 active_group_counts.append(int(len(active_idx)))
#                 if not args.light_output:
#                     active_group_indices_per_batch.append(active_idx.tolist())
#
#                 batch_iter.set_postfix(
#                     altered=int(flip_mask.sum()),
#                     active=len(active_idx),
#                     acc=f"{batch_acc:.3f}",
#                 )
#
#                 enough_history = len(divs) >= 2 * args.win_size
#                 at_update_boundary = (pos + 1) % args.adaptive_interval == 0
#
#                 if args.adaptive_enable and enough_history and at_update_boundary:
#                     recent_divs = divs[-(2 * args.win_size):]
#
#                     scores, delta, tstat = compute_recent_scores(
#                         recent_divs=recent_divs,
#                         active_idx=active_idx,
#                         fi_df=fi_df,
#                         win_size=args.win_size,
#                         score_method=args.adaptive_score,
#                         metric=args.metric,
#                     )
#
#                     old_active_idx = active_idx.copy()
#                     active_idx, order = select_active_groups(
#                         active_idx=active_idx,
#                         scores=scores,
#                         top_k=args.adaptive_top_k,
#                         threshold=args.adaptive_threshold,
#                         min_groups=args.adaptive_min_groups,
#                     )
#
#                     adaptive_updates.append(
#                         {
#                             "batch": int(pos),
#                             "before_count": int(len(old_active_idx)),
#                             "after_count": int(len(active_idx)),
#                             "score_method": args.adaptive_score,
#                             "top_score": float(scores[order[0]]) if len(order) > 0 else 0.0,
#                             "mean_score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
#                             "selected_global_indices": active_idx.tolist() if not args.light_output else None,
#                         }
#                     )
#
#             payload = {
#                 "subgroup": target_sg,
#                 "accuracies": accuracies,
#                 "f1": f1_scores,
#                 "divs": divs,
#                 "noise_fracs": noise_fracs,
#                 "altered": altered_masks,
#                 "support_bucket": (support_low, support_high),
#                 "target_support": target_support,
#                 "full_group_count": int(n_groups),
#                 "adaptive_enabled": bool(args.adaptive_enable),
#                 "adaptive_params": {
#                     "adaptive_top_k": args.adaptive_top_k,
#                     "adaptive_threshold": args.adaptive_threshold,
#                     "adaptive_interval": args.adaptive_interval,
#                     "adaptive_score": args.adaptive_score,
#                     "adaptive_min_groups": args.adaptive_min_groups,
#                     "win_size": args.win_size,
#                     "light_output": args.light_output,
#                 },
#                 "active_group_counts": active_group_counts,
#                 "adaptive_updates": adaptive_updates,
#             }
#
#             if not args.light_output:
#                 payload["y_trues"] = y_trues
#                 payload["y_preds"] = y_preds
#                 payload["active_group_indices_per_batch"] = active_group_indices_per_batch
#
#             with open(out_path, "wb") as f:
#                 pickle.dump(payload, f)
#
#             elapsed = time.perf_counter() - start_time
#             manifest_row = {
#                 "checkpoint": args.checkpoint,
#                 "outfile": str(out_path),
#                 "noise": args.frac_noise,
#                 "support_low": support_low,
#                 "support_high": support_high,
#                 "target": "-".join(map(str, target_sg)),
#                 "target_support": target_support,
#                 "adaptive_enabled": bool(args.adaptive_enable),
#                 "full_group_count": int(n_groups),
#                 "avg_active_groups": float(np.mean(active_group_counts)) if active_group_counts else None,
#                 "min_active_groups": int(np.min(active_group_counts)) if active_group_counts else None,
#                 "max_active_groups": int(np.max(active_group_counts)) if active_group_counts else None,
#                 "elapsed_sec": elapsed,
#                 "light_output": args.light_output,
#                 "status": "completed",
#             }
#             append_manifest_row(progress_csv, manifest_row)
#             completed_outputs.add(str(out_path))
#
#             completed_counter += 1
#             if completed_counter % args.save_every == 0:
#                 log_progress(progress_txt, f"[SAVE] completed {completed_counter} targets so far")
#
#     log_progress(progress_txt, "[DONE] adaptive injection completed")