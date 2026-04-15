from collections import defaultdict

import config
import os
from glob import glob
import sys

sys.path.append("..")

import numpy as np
import pickle

import matplotlib.pyplot as plt
from src.detect import detect_singlebatch, detect_multibatch, _get_altered_in_window
from tqdm import tqdm

import re


class Result:
    def __init__(self, delta, tstat, altered, gt, support, window, metric, sg):
        self.delta = delta
        self.tstat = tstat
        self.altered = altered
        self.gt = gt
        self.support = support
        self.window = window
        self.metric = metric
        self.sg = sg  # sg will be useless if gt="neg"


def get_support_bucket(fname):
    try:
        return tuple(map(float, re.findall(r"support-((?:0|1)\.\d+)-((?:0|1)\.\d+)-", fname)[0]))
    except:
        print(fname)
        raise ValueError


if __name__ == "__main__":
    # checkpoint = "xgb-adult"
    # noise_frac = 0.5

    checkpoint = "resnet50"
    noise_frac = 1.0
    skip = 0

    # ckpt_dir = "/data2/fgiobergia/drift-experiments/"
    ckpt_dir = os.path.join(config.ckpt_dir, "sup-wise")

    neg = glob(os.path.join(ckpt_dir, f"{checkpoint}-noise-0.00-support-*pkl"))
    pos = glob(os.path.join(ckpt_dir, f"{checkpoint}-noise-{noise_frac:.2f}-support-*pkl"))

    # count number of files in each bucket, separately for pos and neg
    counts = {"pos": defaultdict(lambda: 0), "neg": 0}
    for fname in pos:
        support_bucket = get_support_bucket(fname)
        counts["pos"][support_bucket] += 1
    for fname in neg:
        counts["neg"] += 1
    print(dict(counts["pos"]))
    print(counts["neg"])

    max_pos = max(counts["pos"].values())

    results = []

    for label, fnames in zip(["pos", "neg"], [pos, neg]):
        for tgt_file in tqdm(fnames):
            with open(tgt_file, "rb") as f:
                try:
                    obj = pickle.load(f)
                except pickle.UnpicklingError:
                    print(f"Error with {tgt_file}")
                    continue
                except EOFError:
                    print(f"EOFError with {tgt_file}")
                    continue
                sg = frozenset(obj["subgroup"])
                accuracies = obj["accuracies"]
                f1 = obj["f1"]
                divs = obj["divs"]
                y_trues = obj["y_trues"]
                y_preds = obj["y_preds"]
                altered = obj["altered"]
                matches_ts_list = obj["matches_batches"]

            for d in divs:
                d.set_index("subgroup", inplace=True)
                d["accuracy"] = (d["tp"] + d["tn"]) / (d["tp"] + d["tn"] + d["fp"] + d["fn"])
                # d["f1"] = 2 * d["tp"] / (2 * d["tp"] + d["fp"] + d["fn"])

            ndx = divs[0].index
            ndx_target = (ndx == sg).nonzero()[0][0]

            bucket = get_support_bucket(tgt_file)

            # compute ground truth (only done once, for each injection -- since it is not affected by the prediction(s))
            altered_per_sg = np.vstack(
                [matches_ts_list[i].matches[altered[i]].sum(axis=0) for i in range(len(matches_ts_list))])
            count_per_sg = np.vstack([matches_ts_list[i].matches.sum(axis=0) for i in range(len(matches_ts_list))])
            count_per_sg[count_per_sg == 0] = 1  # avoid division by 0 (fraction will still be 0)
            altered_frac = altered_per_sg / count_per_sg

            curr_gt = []
            for window in range(2, 11):
                altered2 = _get_altered_in_window(matches_ts_list, altered_frac, (
                len(divs) - window, window))  # how many were altered in the last window? (fraction, for each subgroup)
                altered2[np.isnan(altered2)] = 0
                curr_gt.append(altered2)
                # for metric in ["accuracy", "f1"]:
                for metric in ["accuracy"]:
                    delta, t_stat = detect_singlebatch(divs, metric, (skip, window), (len(divs) - window, window))

                    delta_values = delta.values
                    delta_values[np.isnan(delta_values)] = 0

                    tstat_values = t_stat.values
                    tstat_values[np.isnan(tstat_values)] = 0

                    result = Result(delta_values.astype(np.float32), tstat_values.astype(np.float32),
                                    altered2.astype(np.float32), label, bucket, window, metric, sg)
                    results.append(result)

    # # store buckets
    with open(f"{checkpoint}-results-v2.pkl", "wb") as f:
        pickle.dump(results, f)