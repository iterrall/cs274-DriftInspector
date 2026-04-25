from sklearn.metrics import f1_score, accuracy_score

import os

import argparse
import pickle

import sys
sys.path.append("../../../..")
from src.divexp import *
from src.adult.config import *

def closest_odd(n):
    n = int(n)
    if n % 2 == 0:
        return n + 1
    else:
        return n

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--checkpoint", type=str)
    argParser.add_argument("--n-targets", type=int, default=100)
    argParser.add_argument("--start-noise", type=int, default=10) # add noise after 10 batches
    argParser.add_argument("--transitory", type=int, default=10) # duration of the transitory
    argParser.add_argument("--frac-noise", type=float, default=1.0) # % of points to add noise to (transitory from 0 to this)
    argParser.add_argument("--metric", type=str, choices=["accuracy"], default="accuracy")

    args = argParser.parse_args()

    model_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pkl")
    ds_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.dataset.pkl")
    matches_filename = os.path.join(ckpt_dir, f"matches-{args.checkpoint}.pkl")

    with open(model_filename, "rb") as f:
        xgb = pickle.load(f)

    with open(ds_filename, "rb") as f:
        ds = pickle.load(f)
        train_set, test_sets, num, categ, preprocessor = ds["train"], ds["test_chunks"], ds["numerical"], ds[
            "categorical"], ds["transform"]

    with open(matches_filename, "rb") as f:
        matches_obj = pickle.load(f)
        df_train = matches_obj["metadata_train"]
        df_tests = matches_obj["metadata_batches"]
        matches = matches_obj["matches_train"]
        matches_ts_list = matches_obj["matches_batches"]

    # output_dir = os.path.join(ckpt_dir, f"{args.checkpoint}-{args.metric}-noise-{args.frac_noise:.2f}")
    # os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(ckpt_dir, f"{args.checkpoint}-{args.metric}-noise-{args.frac_noise:.2f}")
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    for target_sg in matches.fi.itemsets.sample(n=args.n_targets, random_state=42):
        # sg = tuple(target_sg)
        # outfile = os.path.join(output_dir, f"target-{'-'.join(map(str, sorted(sg)))}.pkl")
        # print("Target", sg, "output", outfile)

        sg = tuple(sorted(int(v) for v in target_sg))
        outfile = os.path.join(output_dir, f"target-{'-'.join(map(str, sorted(sg)))}.pkl")
        print("Target", sg, "output", outfile)

        n_batches = len(test_sets)
        noise_fracs = np.zeros(n_batches)

        assert args.start_noise + args.transitory < n_batches
        noise_fracs[args.start_noise:args.start_noise + args.transitory] = np.linspace(0, args.frac_noise,
                                                                                       args.transitory)
        noise_fracs[args.start_noise + args.transitory:] = args.frac_noise

        accuracies = []
        f1 = []
        y_trues = []
        y_preds = []
        divs = []
        altered = []

        samples = [None] * len(test_sets)

        for pos, (ts, noise_frac, matches_ts, df_ohe) in enumerate(zip(test_sets, noise_fracs, matches_ts_list, df_tests)):
            y_pred = xgb.predict(preprocessor.transform(ts.drop(columns=["target"])))
            y_true = np.copy(ts["target"].values)

            mask = df_ohe.values[:, list(target_sg)].sum(axis=1) == len(target_sg)
            mask_noise = rng.random(len(y_true)) < noise_frac
            if (mask & mask_noise).sum() > 0:
                y_true[mask & mask_noise] = 1 - y_true[mask & mask_noise]
            num_altered = (mask & mask_noise).sum()

            altered.append(mask & mask_noise)

            y_trues.append(y_true)
            y_preds.append(y_pred)

            divs.append(div_explorer(matches_ts, y_true, y_pred, [args.metric]))

            accuracies.append(accuracy_score(y_true, y_pred))
            f1.append(f1_score(y_true, y_pred))
            print("Altered", num_altered, "out of", mask.sum(), "accuracy", accuracies[-1], "f1", f1[-1])

        # store results
        with open(outfile, "wb") as f:
            pickle.dump({
                "subgroup": sg,
                "accuracies": accuracies,
                "f1": f1,
                "divs": divs,
                "y_trues": y_trues,
                "y_preds": y_preds,
                "noise_fracs": noise_fracs,
                "altered": altered,
                "matches_batches": matches_ts_list,
            }, f)
