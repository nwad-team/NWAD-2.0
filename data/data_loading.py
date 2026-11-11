import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "datasets" / "raw"
PROCESSED_DIR = PROJECT_DIR / "datasets" / "processed"

# Names are normalized to lowercase because the preprocessing pipeline
# saves lowercase column names.
EXCLUDED_COLUMNS = {
    "label",
    "attack_cat",
    "srcip",
    "dstip",
    "stime",
    "ltime",
}

SHORTCUT_CHECK_COLUMNS = [
    "sport",
    "dsport",
    "proto",
    "service",
    "state",
    "is_sm_ips_ports",
]

# Main model: 37 behavioral features selected by the team.
BEHAVIORAL_COLUMNS = [
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin",
    "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "sjit", "djit", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]

# Map user-facing CLI choices to the processed dataset directories.
PROCESSED_DATASETS = {
    "A": ("config_A", "without_smote"),
    "A-Smote": ("config_A", "with_smote"),
    "C": ("config_C", "without_smote"),
    "C-Smote": ("config_C", "with_smote"),
}

SPLITS = {"train", "validation", "test"}

CONFIG_CLASSES = {
    "A": [
        "Benign",
        "Generic",
        "Exploits",
        "Fuzzers",
        "Reconnaissance",
    ],
    "A-Smote": [
        "Benign",
        "Generic",
        "Exploits",
        "Fuzzers",
        "Reconnaissance",
    ],
    "C": [
        "Benign",
        "Generic",
        "Exploits",
        "Backdoor",
        "Shellcode",
    ],
    "C-Smote": [
        "Benign",
        "Generic",
        "Exploits",
        "Backdoor",
        "Shellcode",
    ],
}


def get_processed_dataset_path(dataset: str,split: Literal["train", "validation", "test"],) -> Path:
    if dataset not in PROCESSED_DATASETS:
        valid = ", ".join(PROCESSED_DATASETS)
        raise ValueError(f"Unknown dataset {dataset!r}. Choose one of: {valid}.")

    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}. Choose one of: {sorted(SPLITS)}.")

    config_dir, variant_dir = PROCESSED_DATASETS[dataset]
    csv_path = PROCESSED_DIR / config_dir / variant_dir / f"{split}.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Processed {split} file not found: {csv_path}. "
            "Run preprocessing_data.py first."
        )

    return csv_path


def get_feature_columns(df: pd.DataFrame,feature_set: Literal["behavioral", "shortcut", "combined"] = "behavioral",) -> list[str]:
    if feature_set == "behavioral":
        requested = BEHAVIORAL_COLUMNS
    elif feature_set == "shortcut":
        requested = SHORTCUT_CHECK_COLUMNS
    elif feature_set == "combined":
        requested = BEHAVIORAL_COLUMNS + SHORTCUT_CHECK_COLUMNS
    else:
        raise ValueError("feature_set must be 'behavioral', 'shortcut', or 'combined'.")

    # Explicitly prevent targets and identifiers from entering X.
    requested = [
        col for col in requested
        if col not in EXCLUDED_COLUMNS
    ]

    missing = [col for col in requested if col not in df.columns]
    if missing:
        raise ValueError(
            "Expected feature columns are missing from the processed CSV: "
            f"{missing}. Available columns: {df.columns.tolist()}"
        )

    return requested


def load_processed_split(dataset: str,split: Literal["train", "validation", "test"],feature_set: Literal["behavioral", "shortcut", "combined"] = "behavioral",) -> tuple[pd.DataFrame, pd.Series]:

    csv_path = get_processed_dataset_path(dataset, split)
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    if "attack_cat" not in df.columns:
        raise ValueError(f"'attack_cat' is missing from {csv_path}.")

    if df["attack_cat"].isna().any():
        raise ValueError(f"Missing target labels found in {csv_path}.")

    feature_columns = get_feature_columns(df, feature_set)

    X = df[feature_columns].copy()
    y = df["attack_cat"].astype(str).copy()
    y.name = "attack_cat"

    return X, y

def load_train_validation(dataset: str,feature_set: Literal["behavioral", "shortcut", "combined"] = "behavioral") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train, y_train = load_processed_split(dataset, "train", feature_set)
    X_validation, y_validation = load_processed_split(dataset, "validation", feature_set)

    return X_train, y_train, X_validation, y_validation


CONFIG_CLASSES = {
    "A": [
        "Benign",
        "Generic",
        "Exploits",
        "Fuzzers",
        "Reconnaissance",
    ],
    "A-Smote": [
        "Benign",
        "Generic",
        "Exploits",
        "Fuzzers",
        "Reconnaissance",
    ],
    "C": [
        "Benign",
        "Generic",
        "Exploits",
        "Backdoor",
        "Shellcode",
    ],
    "C-Smote": [
        "Benign",
        "Generic",
        "Exploits",
        "Backdoor",
        "Shellcode",
    ],
}

def get_class_names(dataset: str) -> list[str]:
    if dataset not in CONFIG_CLASSES:
        valid = ", ".join(CONFIG_CLASSES)
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choose one of: {valid}."
        )

    return CONFIG_CLASSES[dataset].copy()


def encode_labels(labels: pd.Series,dataset: str,) -> np.ndarray:
    class_names = get_class_names(dataset)
    label_to_id = {
        name: index
        for index, name in enumerate(class_names)
    }

    encoded = labels.map(label_to_id)

    if encoded.isna().any():
        unexpected = labels[encoded.isna()].unique().tolist()
        raise ValueError(
            f"Labels outside the known classes for {dataset}: "
            f"{unexpected}"
        )

    return encoded.to_numpy(dtype=np.int64)