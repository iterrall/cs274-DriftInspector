#!/usr/bin/env python3
"""
Baseline support-wise subgroup injection for Adult DriftInspector.

This file takes as input:
- a trained checkpoint from models.py
- subgroup matches from precompute.py

It produces support-bucketed subgroup drift experiments in:
    models-ckpt/sup-wise/

New features added:
- tqdm progress bars
- periodic CSV/TXT checkpoints
- --resume
- --light-output
- --n-proc
- --chunk-size
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.divexp import compute_matches, div_explorer
from src.adult.config import ckpt_dir


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_itemset(x) -> Tuple[int, ...]:
    try:
        return tuple(sorted(int(v) for v in x))
    except Exception:
        return tuple()


def build_support_buckets(fi_df: pd.DataFrame, n_bins: int = 13) -> List[Tuple[float, float, List[int]]]:
    """
    Build support buckets from fi support values using quantile bins.

    Returns:
        [(support_low, support_high, row_indices), ...]
    """
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
    if start_noise + transitory >= n_batches:
        raise ValueError(
            f"Invalid noise schedule: start_noise({start_noise}) + transitory({transitory}) "
            f"must be < n_batches({n_batches})"
        )

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


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="Run baseline subgroup drift injection experiments.")
    argParser.add_argument("--checkpoint", type=str, required=True)
    argParser.add_argument("--n-targets", type=int, default=100, help="Targets sampled per support bucket.")

    # injection schedule
    argParser.add_argument("--start-noise", type=int, default=10, help="Start adding noise after this batch index.")
    argParser.add_argument("--transitory", type=int, default=10, help="Duration of the transitory ramp-up.")
    argParser.add_argument("--frac-noise", type=float, default=0.5, help="Final fraction of subgroup points flipped.")
    argParser.add_argument("--metric", type=str, choices=["accuracy"], default="accuracy")

    # smoothing / control
    argParser.add_argument("--n-support-buckets", type=int, default=13)
    argParser.add_argument("--n-proc", type=int, default=2, help="Number of processes for compute_matches.")
    argParser.add_argument("--chunk-size", type=int, default=10, help="Checkpoint/log frequency in number of targets.")
    argParser.add_argument("--seed", type=int, default=42)

    # durability / storage
    argParser.add_argument("--resume", action="store_true")
    argParser.add_argument("--light-output", action="store_true", help="Do not store bulky y_trues/y_preds/batches.")

    args = argParser.parse_args()

    rng = np.random.default_rng(args.seed)

    ckpt_root = Path(ckpt_dir).resolve()
    supwise_dir = ckpt_root / "sup-wise"
    ensure_dir(supwise_dir)

    progress_csv = supwise_dir / f"{args.checkpoint}-baseline_inject_progress.csv"
    progress_txt = supwise_dir / f"{args.checkpoint}-baseline_inject_progress.txt"
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
        matches_train = matches_obj["matches_train"]
        matches_batches = matches_obj["matches_batches"]
        fi_df = matches_train.fi.reset_index(drop=True)

    n_batches = len(test_sets)
    buckets = build_support_buckets(fi_df, n_bins=args.n_support_buckets)

    if not buckets:
        raise RuntimeError("No support buckets could be built from the fi dataframe.")

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Total mined subgroups: {len(fi_df)}")
    print(f"Support buckets: {len(buckets)}")
    print(f"Light output: {args.light_output}")
    print(f"Resume: {args.resume}")

    completed_counter = 0

    for support_low, support_high, bucket_idx in tqdm(buckets, desc="Support buckets"):
        if len(bucket_idx) == 0:
            continue

        sample_size = min(args.n_targets, len(bucket_idx))
        chosen_rows = rng.choice(bucket_idx, size=sample_size, replace=False)

        target_iter = tqdm(
            chosen_rows,
            desc=f"Targets {support_low:.4f}-{support_high:.4f}",
            leave=False,
        )

        for row_idx in target_iter:
            start_time = time.perf_counter()

            target_row = fi_df.iloc[int(row_idx)]
            target_sg = normalize_itemset(target_row["itemsets"])
            target_support = float(target_row["support"])

            out_name = (
                f"{args.checkpoint}-noise-{args.frac_noise:.2f}-"
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

            # optional bulky outputs
            y_trues = [] if not args.light_output else None
            y_preds = [] if not args.light_output else None
            batches = [] if not args.light_output else None

            batch_iter = tqdm(
                zip(test_sets, noise_fracs, matches_batches),
                total=n_batches,
                desc="Batches",
                leave=False,
            )

            for batch_idx, (batch_df, noise_frac, matches_ts) in enumerate(batch_iter):
                X = preprocessor.transform(batch_df.drop(columns=["target"]))
                y_pred = model.predict(X)
                y_true = np.copy(batch_df["target"].values)

                # Reconstruct the boolean OHE view needed for target mask.
                # We derive it from the sparse matches matrix columns directly.
                # The matches_ts.fi contains the itemset column indices (in OHE space).
                # To find which points are in the target subgroup we use the matches matrix.
                subgroup_cols = list(target_sg)
                # Find the row index in fi that corresponds to the target subgroup
                fi_row = matches_ts.fi[
                    matches_ts.fi["itemsets"].apply(
                        lambda x: normalize_itemset(x) == target_sg
                    )
                ]
                if not fi_row.empty:
                    sg_fi_idx = fi_row.index[0]
                    target_mask = np.asarray(
                        matches_ts.matches[:, sg_fi_idx].todense()
                    ).ravel().astype(bool)
                else:
                    # Subgroup not found in this batch's fi – no points match
                    target_mask = np.zeros(len(y_true), dtype=bool)

                noise_mask = rng.random(len(y_true)) < noise_frac
                flip_mask = target_mask & noise_mask

                if flip_mask.sum() > 0:
                    y_true[flip_mask] = 1 - y_true[flip_mask]

                altered_masks.append(flip_mask)

                if not args.light_output:
                    y_trues.append(y_true)
                    y_preds.append(y_pred)
                    batches.append(batch_df)

                div_df = div_explorer(matches_ts, y_true, y_pred, [args.metric])
                divs.append(div_df)

                batch_acc = accuracy_score(y_true, y_pred)
                batch_f1 = f1_score(y_true, y_pred)
                accuracies.append(batch_acc)
                f1_scores.append(batch_f1)

                batch_iter.set_postfix(
                    altered=int(flip_mask.sum()),
                    support=int(target_mask.sum()),
                    acc=f"{batch_acc:.3f}",
                )

            payload = {
                "subgroup": target_sg,
                "accuracies": accuracies,
                "f1": f1_scores,
                "divs": divs,
                "noise_fracs": noise_fracs,
                "altered": altered_masks,
                "support_bucket": (support_low, support_high),
                "target_support": target_support,
                "light_output": bool(args.light_output),
            }

            if not args.light_output:
                payload.update({
                    "y_trues": y_trues,
                    "y_preds": y_preds,
                    "batches": batches,
                })

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
                "elapsed_sec": elapsed,
                "light_output": args.light_output,
                "status": "completed",
            }
            append_manifest_row(progress_csv, manifest_row)
            completed_outputs.add(str(out_path))

            completed_counter += 1
            if completed_counter % args.chunk_size == 0:
                log_progress(
                    progress_txt,
                    f"[SAVE] completed {completed_counter} targets so far for checkpoint={args.checkpoint}"
                )

    log_progress(progress_txt, "[DONE] baseline injection completed")

# #!/usr/bin/env python3
# """
# Baseline support-wise subgroup injection for Adult DriftInspector.
#
# This file takes as input:
# - a trained checkpoint from models.py
# - subgroup matches from precompute.py
#
# It produces support-bucketed subgroup drift experiments in:
#     models-ckpt/sup-wise/
#
# New features added:
# - tqdm progress bars
# - periodic CSV/TXT checkpoints
# - --resume
# - --light-output
# - --n-proc
# - --chunk-size
# """
#
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
# if str(HERE) not in sys.path:
#     sys.path.insert(0, str(HERE))
#
# from src.divexp import compute_matches, div_explorer
# from src.adult.config import ckpt_dir
#
#
# def ensure_dir(path: Path) -> None:
#     path.mkdir(parents=True, exist_ok=True)
#
#
# def normalize_itemset(x) -> Tuple[int, ...]:
#     try:
#         return tuple(sorted(int(v) for v in x))
#     except Exception:
#         return tuple()
#
#
# def build_support_buckets(fi_df: pd.DataFrame, n_bins: int = 13) -> List[Tuple[float, float, List[int]]]:
#     """
#     Build support buckets from fi support values using quantile bins.
#
#     Returns:
#         [(support_low, support_high, row_indices), ...]
#     """
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
#     if start_noise + transitory >= n_batches:
#         raise ValueError(
#             f"Invalid noise schedule: start_noise({start_noise}) + transitory({transitory}) "
#             f"must be < n_batches({n_batches})"
#         )
#
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
#
#     try:
#         df = pd.read_csv(manifest_csv)
#     except Exception:
#         return set()
#
#     if "outfile" not in df.columns:
#         return set()
#
#     return set(df["outfile"].astype(str))
#
#
# def log_progress(txt_path: Path, msg: str):
#     with open(txt_path, "a", encoding="utf-8") as f:
#         f.write(msg + "\n")
#
#
# if __name__ == "__main__":
#     argParser = argparse.ArgumentParser(description="Run baseline subgroup drift injection experiments.")
#     argParser.add_argument("--checkpoint", type=str, required=True)
#     argParser.add_argument("--n-targets", type=int, default=100, help="Targets sampled per support bucket.")
#
#     # injection schedule
#     argParser.add_argument("--start-noise", type=int, default=10, help="Start adding noise after this batch index.")
#     argParser.add_argument("--transitory", type=int, default=10, help="Duration of the transitory ramp-up.")
#     argParser.add_argument("--frac-noise", type=float, default=0.5, help="Final fraction of subgroup points flipped.")
#     argParser.add_argument("--metric", type=str, choices=["accuracy"], default="accuracy")
#
#     # smoothing / control
#     argParser.add_argument("--n-support-buckets", type=int, default=13)
#     argParser.add_argument("--n-proc", type=int, default=2, help="Number of processes for compute_matches.")
#     argParser.add_argument("--chunk-size", type=int, default=10, help="Checkpoint/log frequency in number of targets.")
#     argParser.add_argument("--seed", type=int, default=42)
#
#     # durability / storage
#     argParser.add_argument("--resume", action="store_true")
#     argParser.add_argument("--light-output", action="store_true", help="Do not store bulky y_trues/y_preds/batches.")
#
#     args = argParser.parse_args()
#
#     rng = np.random.default_rng(args.seed)
#
#     ckpt_root = Path(ckpt_dir).resolve()
#     supwise_dir = ckpt_root / "sup-wise"
#     ensure_dir(supwise_dir)
#
#     progress_csv = supwise_dir / f"{args.checkpoint}-baseline_inject_progress.csv"
#     progress_txt = supwise_dir / f"{args.checkpoint}-baseline_inject_progress.txt"
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
#         matches_train = matches_obj["matches_train"]
#         fi_df = matches_train.fi.reset_index(drop=True)
#
#     n_batches = len(test_sets)
#     buckets = build_support_buckets(fi_df, n_bins=args.n_support_buckets)
#
#     if not buckets:
#         raise RuntimeError("No support buckets could be built from the fi dataframe.")
#
#     print(f"Loaded checkpoint: {args.checkpoint}")
#     print(f"Total mined subgroups: {len(fi_df)}")
#     print(f"Support buckets: {len(buckets)}")
#     print(f"Light output: {args.light_output}")
#     print(f"Resume: {args.resume}")
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
#         target_iter = tqdm(
#             chosen_rows,
#             desc=f"Targets {support_low:.4f}-{support_high:.4f}",
#             leave=False,
#         )
#
#         for row_idx in target_iter:
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
#             accuracies: List[float] = []
#             f1_scores: List[float] = []
#             divs: List[pd.DataFrame] = []
#             altered_masks: List[np.ndarray] = []
#
#             # optional bulky outputs
#             y_trues = [] if not args.light_output else None
#             y_preds = [] if not args.light_output else None
#             batches = [] if not args.light_output else None
#             batches_unsup = [] if not args.light_output else None
#
#             batch_iter = tqdm(
#                 zip(test_sets, noise_fracs),
#                 total=n_batches,
#                 desc="Batches",
#                 leave=False,
#             )
#
#             for batch_idx, (batch_df, noise_frac) in enumerate(batch_iter):
#                 X = preprocessor.transform(batch_df.drop(columns=["target"]))
#                 y_pred = model.predict(X)
#                 y_true = np.copy(batch_df["target"].values)
#
#                 batch_unsup = pd.DataFrame(
#                     preprocessor.transform(batch_df.drop(columns=["target"])),
#                     columns=preprocessor.get_feature_names_out()
#                 )
#                 batch_unsup = batch_unsup.astype(bool)
#
#                 matches_ts = compute_matches(batch_unsup, fi=matches_train.fi, n_proc=args.n_proc)
#
#                 subgroup_cols = list(target_sg)
#                 target_mask = batch_unsup.values[:, subgroup_cols].sum(axis=1) == len(subgroup_cols)
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
#                     batches.append(batch_df)
#                     batches_unsup.append(batch_unsup)
#
#                 div_df = div_explorer(matches_ts, y_true, y_pred, [args.metric])
#                 divs.append(div_df)
#
#                 batch_acc = accuracy_score(y_true, y_pred)
#                 batch_f1 = f1_score(y_true, y_pred)
#                 accuracies.append(batch_acc)
#                 f1_scores.append(batch_f1)
#
#                 batch_iter.set_postfix(
#                     altered=int(flip_mask.sum()),
#                     support=int(target_mask.sum()),
#                     acc=f"{batch_acc:.3f}",
#                 )
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
#                 "light_output": bool(args.light_output),
#             }
#
#             if not args.light_output:
#                 payload.update({
#                     "y_trues": y_trues,
#                     "y_preds": y_preds,
#                     "batches": batches,
#                     "batches_unsup": batches_unsup,
#                 })
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
#                 "elapsed_sec": elapsed,
#                 "light_output": args.light_output,
#                 "status": "completed",
#             }
#             append_manifest_row(progress_csv, manifest_row)
#             completed_outputs.add(str(out_path))
#
#             completed_counter += 1
#             if completed_counter % args.chunk_size == 0:
#                 log_progress(
#                     progress_txt,
#                     f"[SAVE] completed {completed_counter} targets so far for checkpoint={args.checkpoint}"
#                 )
#
#     log_progress(progress_txt, "[DONE] baseline injection completed")