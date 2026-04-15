"""
This file takes as input a checkpoint for which a model exists (models.py), and the subgroups (M) for the training set (precompute.py).
It produces a adult.test session (i.e. sequence of batches) where one randomly chosen subgorup is injected with noise.

This is repeated N times (n-targets) for each of a pool of supports (see "supports"). 

"""

from sklearn.metrics import f1_score, accuracy_score

import os

import argparse
import pickle

import sys
sys.path.append("../../../..")
from src.divexp import *
from src.adult.config import *

import random

def closest_odd(n):
    n = int(n)
    if n % 2 == 0:
        return n + 1
    else:
        return n

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--checkpoint", type=str)
    argParser.add_argument("--n-targets", type=int, default=10) #### changed for testing from 100 to 10
    
    # parameters for "injection"
    argParser.add_argument("--start-noise", type=int, default=10) # add noise after 10 batches
    argParser.add_argument("--transitory", type=int, default=10) # duration of the transitory
    argParser.add_argument("--frac-noise", type=float, default=0.5) # % of points to add noise to (transitory from 0 to this)d
    
    argParser.add_argument("--metric", type=str, choices=["accuracy"], default="accuracy")

    args = argParser.parse_args()

    model_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pkl")
    ds_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.dataset.pkl")
    matches_filename = os.path.join(ckpt_dir, f"matches-{args.checkpoint}.pkl")

    with open(model_filename, "rb") as f:
        xgb = pickle.load(f)
    
    with open(ds_filename, "rb") as f:
        ds = pickle.load(f)
        train_set, test_sets, num, categ, preprocessor = ds["train"], ds["test_chunks"], ds["numerical"], ds["categorical"], ds["transform"]
        df_test = pd.concat(test_sets, axis=0)
    
    with open(matches_filename, "rb") as f:
        matches_obj = pickle.load(f)
        # df_train = matches_obj["metadata_train"]
        df_tests_unsup = matches_obj["metadata_batches"]
        matches = matches_obj["matches_train"]
        df_test_unsup = pd.concat(df_tests_unsup, axis=0)
        # matches_ts_list = matches_obj["matches_batches"]
    
    # output_dir = os.path.join(ckpt_dir, f"{args.checkpoint}-{args.metric}-noise-{args.frac_noise:.2f}")
    # os.makedirs(output_dir, exist_ok=True)

    vmin = 0.01
    vmax = 1.0
    n_batches = 30

    supports = np.logspace(np.log10(vmin), np.log10(vmax), 20)

    for i in range(len(supports)-1):
        # relevant subgroups
        ######
        # valid_subgrp = matches.fi[(matches.fi.support >= supports[i]) & (matches.fi.support < supports[i+1])]
        # print(supports[i], "to", supports[i+1], "=>", len(valid_subgrp))
        #
        #
        # accuracies = []
        # f1 = []
        # y_trues = []
        # y_preds = []
        # divs = []
        # altered = []
        # # sample subgroups
        # for cnt, target_sg in enumerate(valid_subgrp.itemsets.sample(n=args.n_targets)):
        valid_subgrp = matches.fi[(matches.fi.support >= supports[i]) & (matches.fi.support < supports[i + 1])]
        print(supports[i], "to", supports[i + 1], "=>", len(valid_subgrp))

        if len(valid_subgrp) == 0:
            continue

        n_sample = min(args.n_targets, len(valid_subgrp))

        accuracies = []
        f1 = []
        y_trues = []
        y_preds = []
        divs = []
        altered = []
        # sample subgroups
        for cnt, target_sg in enumerate(valid_subgrp.itemsets.sample(n=n_sample)):
            # generate adult.test sets
            random.seed(i * args.n_targets + cnt)
            np.random.seed(i * args.n_targets + cnt)
            shuffle_ndx = np.arange(len(df_test))
            np.random.shuffle(shuffle_ndx)
            ##### df_test_chunks = np.array_split(df_test.iloc[shuffle_ndx], n_batches)
            ##### df_test_chunks_unsup = np.array_split(df_test_unsup.iloc[shuffle_ndx], n_batches)

            df_test_shuffled = df_test.iloc[shuffle_ndx].reset_index(drop=True)
            df_test_unsup_shuffled = df_test_unsup.iloc[shuffle_ndx].reset_index(drop=True)

            chunk_size = int(np.ceil(len(df_test_shuffled) / n_batches))

            df_test_chunks = [
                df_test_shuffled.iloc[i:i + chunk_size].copy()
                for i in range(0, len(df_test_shuffled), chunk_size)
            ]

            df_test_chunks_unsup = [
                df_test_unsup_shuffled.iloc[i:i + chunk_size].copy()
                for i in range(0, len(df_test_unsup_shuffled), chunk_size)
            ]
            # noise!
            noise_fracs = np.zeros(n_batches)
            assert args.start_noise + args.transitory < n_batches
            noise_fracs[args.start_noise:args.start_noise+args.transitory] = np.linspace(0, args.frac_noise, args.transitory)
            noise_fracs[args.start_noise + args.transitory:] = args.frac_noise

            for batch, batch_unsup, noise_frac in zip(df_test_chunks, df_test_chunks_unsup, noise_fracs):
                # compute matches
                matches_ts = compute_matches(batch_unsup, fi=matches.fi, n_proc=4)
                print("Target", target_sg, "batch", batch.shape, "batch-unsup", batch_unsup.shape, "matches", matches_ts.matches.mean())

                y_pred = xgb.predict(preprocessor.transform(batch.drop(columns=["target"])))
                y_true = np.copy(batch["target"].values)

                mask = batch_unsup.values[:, list(target_sg)].sum(axis=1) == len(target_sg)
                mask_noise = np.random.random(len(y_true)) < noise_frac
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
        
            ##### outfile = os.path.join(ckpt_dir, "sup-wise", f"{args.checkpoint}-noise-{args.frac_noise:.2f}-support-{supports[i]:.4f}-{supports[i+1]:.4f}-target-{'-'.join(list(map(str,target_sg)))}.pkl")
            outdir = os.path.join(ckpt_dir, "sup-wise")
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir,
                                   f"{args.checkpoint}-noise-{args.frac_noise:.2f}-support-{supports[i]:.4f}-{supports[i + 1]:.4f}-target-{'-'.join(list(map(str, target_sg)))}.pkl")
            print("Storing to", outfile)
            # store results
            with open(outfile, "wb") as f:
                pickle.dump({
                    "subgroup": target_sg,
                    "batches": df_test_chunks,
                    "batches_unsup": df_test_chunks_unsup,
                    "accuracies": accuracies,
                    "f1": f1,
                    "divs": divs,
                    "y_trues": y_trues,
                    "y_preds": y_preds,
                    "noise_fracs": noise_fracs,
                    "altered": altered
                }, f)
            

        
    # for cnt, target_sg in enumerate(matches.fi.itemsets.sample(n=args.n_targets)):
    #     sg = tuple(target_sg)
    #     if args.frac_noise == 0:
    #         outfile = os.path.join(output_dir, f"no-noise-{cnt}.pkl")
    #     else:
    #         outfile = os.path.join(output_dir, f"target-{'-'.join(map(str, sorted(sg)))}.pkl")
    #     print("Target", sg, "output", outfile)

    #     n_batches = len(test_sets)
    #     noise_fracs = np.zeros(n_batches)

    #     assert args.start_noise + args.transitory < n_batches
    #     noise_fracs[args.start_noise:args.start_noise+args.transitory] = np.linspace(0, args.frac_noise, args.transitory)
    #     noise_fracs[args.start_noise + args.transitory:] = args.frac_noise

    #     accuracies = []
    #     f1 = []
    #     y_trues = []
    #     y_preds = []
    #     divs = []
    #     altered = []

    #     samples = [None] * len(test_sets)

    #     for pos, (ts, noise_frac, matches_ts, df_ohe) in enumerate(zip(test_sets, noise_fracs, matches_ts_list, df_tests)):
            
    #         metadata = []

    #         y_pred = xgb.predict(preprocessor.transform(ts.drop(columns=["target"])))
    #         y_true = np.copy(ts["target"].values)

    #         mask = df_ohe.values[:, list(target_sg)].sum(axis=1) == len(target_sg)
    #         mask_noise = np.random.random(len(y_true)) < noise_frac
    #         if (mask & mask_noise).sum() > 0:
    #             y_true[mask & mask_noise] = 1 - y_true[mask & mask_noise]
    #         num_altered = (mask & mask_noise).sum()
            
    #         altered.append(mask & mask_noise)

    #         y_trues.append(y_true)
    #         y_preds.append(y_pred)

    #         divs.append(div_explorer(matches_ts, y_true, y_pred, [args.metric]))

    #         accuracies.append(accuracy_score(y_true, y_pred))
    #         f1.append(f1_score(y_true, y_pred))
    #         print("Altered", num_altered, "out of", mask.sum(), "accuracy", accuracies[-1], "f1", f1[-1])
        
    #     # store results
    #     with open(outfile, "wb") as f:
    #         pickle.dump({
    #             "subgroup": sg,
    #             "accuracies": accuracies,
    #             "f1": f1,
    #             "divs": divs,
    #             "y_trues": y_trues,
    #             "y_preds": y_preds,
    #             "noise_fracs": noise_fracs,
    #             "altered": altered
    #         }, f)
