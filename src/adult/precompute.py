import os
from tqdm import tqdm

import argparse
import pickle

import sys
sys.path.append("../../../..")
from src.divexp import *
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.adult.config import *

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--checkpoint", type=str)
    argParser.add_argument("--minsup", type=float, default=0.01)

    args = argParser.parse_args()

    model_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pkl")
    ds_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.dataset.pkl")

    with open(model_filename, "rb") as f:
        xgb = pickle.load(f)
    
    with open(ds_filename, "rb") as f:
        ds = pickle.load(f)
        df_train, test_sets, num, categ = ds["train"], ds["test_chunks"], ds["numerical"], ds["categorical"]


    bool_preprocessor = ColumnTransformer(
    transformers=[
        # numerical features : bins + 1-hot
        ('num', Pipeline([
            ('bins', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')),
            ]), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categ)], 
        sparse_threshold=0)

    # bool_preprocessor.fit(df_train.drop(columns=["target"]))
    df_train_unsup = pd.DataFrame(data=bool_preprocessor.fit_transform(df_train.drop(columns=["target"])), columns=bool_preprocessor.get_feature_names_out()).astype(bool)
    matches = compute_matches(df_train_unsup, minsup=args.minsup, n_proc=2) #### Change for testing/ temporary

    matches_ts_list = []
    df_tests = []
    for df_test in tqdm(test_sets):
        df_test_unsup = pd.DataFrame(bool_preprocessor.transform(df_test.drop(columns=["target"])), columns=bool_preprocessor.get_feature_names_out()).astype(bool)
        matches_ts = compute_matches(df_test_unsup, fi=matches.fi)
        matches_ts_list.append(matches_ts)
        df_tests.append(df_test_unsup)
    
    # save pickle
    # ckpt_dir = "models-ckpt" ##### commented out to make safer
    matches_filename = os.path.join(ckpt_dir, f"matches-{args.checkpoint}.pkl")

    with open(matches_filename, "wb") as f:
        pickle.dump({
            "matches_train": matches,
            "matches_batches": matches_ts_list,
            "metadata_train": df_train_unsup,
            "metadata_batches": df_tests,
        }, f)