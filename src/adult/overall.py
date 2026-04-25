from collections import defaultdict

from config import *
import os
from glob import glob
import sys

sys.path.append("..")

import numpy as np
import pickle

import matplotlib.pyplot as plt
from src.detect import detect_singlebatch, detect_multibatch
from tqdm import tqdm

import re


def load_file(fname):
    with open(fname, "rb") as f:
        obj = pickle.load(f)

        sg = obj["subgroup"]  # affected sg
        batches = obj["batches"]
        batches_unsup = obj["batches_unsup"]
        accuracies = obj["accuracies"]
        f1 = obj["f1"]
        divs = obj["divs"]
        y_trues = obj["y_trues"]
        y_preds = obj["y_preds"]
        noise_fracs = obj["noise_fracs"]
        altered = obj["altered"]
        matches_batches = obj["matches_batches"]

    for d in divs:
        d.set_index("subgroup", inplace=True)

    ndx = divs[0].index
    ndx_target = (ndx == sg).nonzero()[0][0]

    # compute metrics of interest
    for div in divs:
        div["accuracy"] = (div["tp"] + div["tn"]) / (div["tp"] + div["tn"] + div["fp"] + div["fn"])
        div["f1"] = 2 * div["tp"] / (2 * div["tp"] + div["fp"] + div["fn"])

    return divs


def get_support_bucket(fname):
    try:
        return tuple(map(float, re.findall(r"support-((?:0|1)\.\d+)-((?:0|1)\.\d+)-", fname)[0]))
    except:
        print(fname)
        raise ValueError


if __name__ == "__main__":
    checkpoint = "xgb-adult"
    noise_frac = 0.5

    # ckpt_dir = "/data2/fgiobergia/drift-experiments/"

    neg = glob(os.path.join(ckpt_dir, f"sup-wise/{checkpoint}-noise-0.00-support-*pkl"))
    pos = glob(os.path.join(ckpt_dir, f"sup-wise/{checkpoint}-noise-{noise_frac:.2f}-support-*pkl"))

    buckets = {
        "pos": {},
        "neg": []
    }

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

    for label, fnames in zip(["pos", "neg"], [pos, neg]):
        for fname in tqdm(fnames[:max_pos if label == "neg" else None]):
            try:
                divs = load_file(fname)
            except EOFError:
                print("EOF Error loading file", fname)
                continue
            except pickle.UnpicklingError:
                print("Unpickling Error loading file", fname)
                continue
            delta, tstat = detect_singlebatch(divs, "accuracy", (0, 5), (len(divs) - 5, 5))

            if label == "pos":
                bucket = get_support_bucket(fname)
                if bucket not in buckets[label]:
                    buckets[label][bucket] = []
                buckets[label][bucket].append((delta.values, tstat.values))
            else:
                # no need to bucketize for negatives
                buckets[label].append((delta.values, tstat.values))

    # store buckets
    with open("buckets.pkl", "wb") as f:
        pickle.dump(buckets, f)