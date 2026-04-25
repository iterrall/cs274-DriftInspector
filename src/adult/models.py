import argparse
import os
import numpy as np
import pickle
import random
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

from src.adult.config import *


def random_split(df, ratios):
    df = df.sample(frac=1).reset_index(drop=True) # shuffle
    splits = []
    start = 0
    for ratio in ratios:
        split = df.iloc[start:start+int(len(df)*ratio)].reset_index(drop=True)
        start += int(len(df)*ratio)
        splits.append(split)
    return splits
    
def load_adult_df():
    print("data_dir =", data_dir)
    print("adult.data exists =", os.path.exists(os.path.join(data_dir, "adult.data")))
    print("adult.test exists =", os.path.exists(os.path.join(data_dir, "adult.test")))


    columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","target"]
    categ = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
    num = [ c for c in columns if c not in categ and c != "target" ]
    df = pd.concat([
        pd.read_csv(os.path.join(data_dir, "adult.data"), header=None, index_col=False),
        pd.read_csv(os.path.join(data_dir, "adult.test"), header=None, index_col=False, skiprows=1)
    ], axis=0)
    df.columns = columns
    df["target"] = df["target"].apply(lambda x: 1 if x in [" >50K", ">50K."] else 0)
    return df, categ, num

if __name__ == "__main__":
    # if called, train a model

    argParser = argparse.ArgumentParser()

    argParser.add_argument("--checkpoint", type=str)
    argParser.add_argument("--use-gpu", action="store_true")
    argParser.add_argument("--n-estimators", type=int, default=50)
    argParser.add_argument("--max-depth", type=int, default=4)
    argParser.add_argument("--learning-rate", type=float, default=0.1)
    argParser.add_argument("--n-jobs", type=int, default=2)

    args = argParser.parse_args()

    print("Loading data...")
    df, categ, num = load_adult_df()
    print("Loaded data:", df.shape)

    # split `ds` into a train & adult.test -- the adult.test set is then further split into 30 chunks
    # NOTE: we should probably store these sets somewhere
    # (they should be deterministic, but still)
    random.seed(42)
    np.random.seed(42)

    test_size = 0.5
    n_chunks = 30
    print("Splitting data...")
    df_train, *df_tests = random_split(df, [ 1- test_size] + [test_size / n_chunks] * n_chunks)
    print("Train size:", df_train.shape)
    print("Num test chunks:", len(df_tests))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categ)])

    print("Preprocessing...")
    y_train = df_train["target"]
    X_train = df_train.drop(columns=["target"])
    X_train = preprocessor.fit_transform(X_train)
    print("Preprocessing done. Shape:", X_train.shape)

    print("Training XGBoost...")
    # xgb = xgb.XGBClassifier() ####****#### CHANGE to smaller for testing/temporary
    xgb_model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_jobs=args.n_jobs,
        random_state=42,
        tree_method="hist",
        device="cuda" if args.use_gpu else "cpu",
    )
    xgb_model.fit(X_train, y_train)
    print("Training done.")

    # save model
    print("Saving model...")
    model_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pkl")
    ds_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.dataset.pkl")

    os.makedirs(ckpt_dir, exist_ok=True)
    with open(model_filename, "wb") as f:
        pickle.dump(xgb_model, f)
    
    with open(ds_filename, "wb") as f:
        pickle.dump({
            "train": df_train,
            "test_chunks": df_tests,
            "numerical": num,
            "categorical": categ,
            "transform": preprocessor
        }, f)


        # python -m src.adult.models --checkpoint=adult_model