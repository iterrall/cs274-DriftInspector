import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

data_dir = "../data"

print("Checking dataset...")

adult_data = os.path.join(data_dir, "adult.data")
adult_test = os.path.join(data_dir, "adult.test")

print("adult.data exists:", os.path.exists(adult_data))
print("adult.test exists:", os.path.exists(adult_test))

# load only first 200 rows (very fast)
df_train = pd.read_csv(adult_data, header=None, nrows=200)
df_test = pd.read_csv(adult_test, header=None, skiprows=1, nrows=200)

df = pd.concat([df_train, df_test])

columns = [
"age","workclass","fnlwgt","education","education-num","marital-status",
"occupation","relationship","race","sex","capital-gain","capital-loss",
"hours-per-week","native-country","target"
]

df.columns = columns

print("Dataset loaded:", df.shape)

# convert target
df["target"] = df["target"].apply(lambda x: 1 if ">50K" in str(x) else 0)

categorical = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
numerical = [c for c in columns if c not in categorical and c != "target"]

X = df.drop(columns=["target"])
y = df["target"]

print("Building preprocessing...")

pre = ColumnTransformer([
("num", StandardScaler(), numerical),
("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

X = pre.fit_transform(X)

print("Preprocessed shape:", X.shape)

print("Training tiny XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=5,
    max_depth=2,
    learning_rate=0.1,
    n_jobs=1
)

model.fit(X, y)

print("Training complete.")

preds = model.predict(X)

print("Predictions example:", preds[:10])
print("Test finished successfully.")