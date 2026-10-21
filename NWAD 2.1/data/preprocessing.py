import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GLOBAL_CLASS_IDS = {
    "Benign": 0,
    "Generic": 1,
    "Exploits": 2,
    "Fuzzers": 3,
    "Reconnaissance": 4,
    "DoS": 5,
    "Backdoor": 6,
    "Shellcode": 7,
    "Analysis": 8,
    "Worms": 9,
}

def build_preprocessor(X_train):

    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    categorical_columns = [
        column
        for column in X_train.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore",sparse_output=False,),),
        ])

        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable input features were found.")

    return ColumnTransformer(transformers=transformers,remainder="drop",)

def encode_global_labels(labels):
    """Map original class names to stable global IDs."""

    unknown_labels = set(labels) - set(GLOBAL_CLASS_IDS)

    if unknown_labels:
        raise ValueError(
            f"Labels missing from global mapping: {sorted(unknown_labels)}"
        )

    return np.asarray(
        [GLOBAL_CLASS_IDS[label] for label in labels],
        dtype=np.int64,
    )
