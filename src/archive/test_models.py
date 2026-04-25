import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

# Make sure project root is importable when running tests
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.adult.models as models
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb


class TestModels(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.tmpdir, "data")
        self.ckpt_dir = os.path.join(self.tmpdir, "models-ckpt")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.adult_data_path = os.path.join(self.data_dir, "adult.data")
        self.adult_test_path = os.path.join(self.data_dir, "adult.test")

        # Small synthetic Adult-like dataset
        train_rows = [
            "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K",
            "50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, >50K",
            "38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K",
            "53, Private, 234721, 11th, 7, Married-civ-spouse, Handlers-cleaners, Husband, Black, Male, 0, 0, 40, United-States, <=50K",
            "28, Private, 338409, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, Black, Female, 0, 0, 40, Cuba, >50K",
            "37, Private, 284582, Masters, 14, Married-civ-spouse, Exec-managerial, Wife, White, Female, 0, 0, 40, United-States, >50K",
        ]

        test_rows = [
            "|1x3 Cross validator",
            "52, Self-emp-not-inc, 209642, HS-grad, 9, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 45, United-States, >50K.",
            "31, Private, 45781, Masters, 14, Never-married, Prof-specialty, Not-in-family, White, Female, 14084, 0, 50, United-States, >50K.",
            "42, Private, 159449, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 5178, 0, 40, United-States, >50K.",
            "23, Private, 122272, Bachelors, 13, Never-married, Adm-clerical, Own-child, White, Female, 0, 0, 30, United-States, <=50K.",
        ]

        with open(self.adult_data_path, "w", encoding="utf-8") as f:
            f.write("\n".join(train_rows))

        with open(self.adult_test_path, "w", encoding="utf-8") as f:
            f.write("\n".join(test_rows))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_random_split_preserves_row_count(self):
        df = pd.DataFrame({"a": range(100)})
        splits = models.random_split(df, [0.5, 0.3, 0.2])

        total_rows = sum(len(s) for s in splits)
        self.assertEqual(total_rows, len(df))
        self.assertEqual(len(splits), 3)

    def test_random_split_returns_reset_indices(self):
        df = pd.DataFrame({"a": range(20)})
        splits = models.random_split(df, [0.5, 0.5])

        for split in splits:
            self.assertEqual(list(split.index), list(range(len(split))))

    def test_load_adult_df_reads_files_and_maps_target(self):
        with patch.object(models, "data_dir", self.data_dir):
            df, categ, num = models.load_adult_df()

        self.assertFalse(df.empty)
        self.assertIn("target", df.columns)

        # Check target mapped to 0/1 only
        self.assertTrue(set(df["target"].unique()).issubset({0, 1}))

        # Basic column expectations
        self.assertIn("sex", categ)
        self.assertIn("age", num)
        self.assertNotIn("target", num)

        # 6 train + 4 test rows = 10 rows total (test skips header line)
        self.assertEqual(len(df), 10)

    def test_load_adult_df_raises_if_missing_files(self):
        missing_dir = os.path.join(self.tmpdir, "missing_data")
        os.makedirs(missing_dir, exist_ok=True)

        with patch.object(models, "data_dir", missing_dir):
            with self.assertRaises(FileNotFoundError):
                models.load_adult_df()

    def test_training_smoke_test(self):
        # This tests the same core training logic as your script
        with patch.object(models, "data_dir", self.data_dir), \
             patch.object(models, "ckpt_dir", self.ckpt_dir):

            df, categ, num = models.load_adult_df()

            np.random.seed(42)
            import random
            random.seed(42)

            # Use a small split for the smoke test
            df_train, *df_tests = models.random_split(df, [0.7, 0.15, 0.15])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categ),
                ]
            )

            y_train = df_train["target"]
            X_train = df_train.drop(columns=["target"])
            X_train = preprocessor.fit_transform(X_train)

            model = xgb.XGBClassifier(
                n_estimators=5,
                max_depth=2,
                learning_rate=0.1,
                n_jobs=1,
                random_state=42,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train)

            model_filename = os.path.join(self.ckpt_dir, "adult_model.pkl")
            ds_filename = os.path.join(self.ckpt_dir, "adult_model.dataset.pkl")

            import pickle
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            with open(ds_filename, "wb") as f:
                pickle.dump(
                    {
                        "train": df_train,
                        "test_chunks": df_tests,
                        "numerical": num,
                        "categorical": categ,
                        "transform": preprocessor,
                    },
                    f,
                )

            self.assertTrue(os.path.exists(model_filename))
            self.assertTrue(os.path.exists(ds_filename))


if __name__ == "__main__":
    unittest.main()