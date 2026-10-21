
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC


# ============================================================
# Configuration
# ============================================================

SEED = 42
MAX_SAMPLES_PER_CLASS = 20_000
SMOTE_TARGET_PER_CLASS = 14_000
PSEUDO_UNKNOWN_CLASS = "DoS"

ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "datasets" / "raw"
PROCESSED_DIR = ROOT_DIR / "datasets" / "processed"

RAW_FILES = [
    RAW_DIR / "UNSW-NB15_1.csv",
    RAW_DIR / "UNSW-NB15_2.csv",
    RAW_DIR / "UNSW-NB15_3.csv",
    RAW_DIR / "UNSW-NB15_4.csv",
]


UNSW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto",
    "state", "dur", "sbytes", "dbytes", "sttl",
    "dttl", "sloss", "dloss", "service", "Sload",
    "Dload", "Spkts", "Dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth",
    "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime",
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
    "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]

TARGET_COLUMN = "attack_cat"

EXCLUDED_COLUMNS = {
    "label",
    "attack_cat",
    "srcip",
    "dstip",
    "stime",
    "ltime",
}

CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "state",
    "is_sm_ips_ports",
]

CONFIG_CLASSES = {
    "A": [
        "Benign",
        "Generic",
        "Exploits",
        "Fuzzers",
        "DoS",
        "Reconnaissance",
    ],
    "C": [
        "Benign",
        "Generic",
        "Exploits",
        "DoS",
        "Backdoor",
        "Shellcode",
    ],
}

ORIGINAL_UNKNOWN_CLASSES = {
    "A": [
        "Analysis",
        "Backdoor",
        "Shellcode",
        "Worms",
    ],
    "C": [
        "Fuzzers",
        "Reconnaissance",
        "Analysis",
        "Worms",
    ],
}


# ============================================================
# Loading and cleaning
# ============================================================

def normalize_column_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name


def normalize_class_name(value) -> str:
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip()

    aliases = {
        "normal": "Benign",
        "benign": "Benign",
        "backdoors": "Backdoor",
        "backdoor": "Backdoor",
        "dos": "DoS",
        "generic": "Generic",
        "exploits": "Exploits",
        "fuzzers": "Fuzzers",
        "reconnaissance": "Reconnaissance",
        "analysis": "Analysis",
        "shellcode": "Shellcode",
        "worms": "Worms",
    }

    return aliases.get(value.lower(), value)


def load_raw_dataset() -> pd.DataFrame:
    frames = []

    for path in RAW_FILES:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw dataset file: {path}"
            )

        print(f"Loading {path.name} ...")

        frame = pd.read_csv(
            path,
            header=None,
            low_memory=False,
        )

        print(f"  Detected columns: {frame.shape[1]}")

        if frame.shape[1] != len(UNSW_COLUMNS):
            raise ValueError(
                f"{path.name} has {frame.shape[1]} columns, "
                f"but the expected UNSW-NB15 schema has "
                f"{len(UNSW_COLUMNS)} columns. "
                "Check the CSV structure before assigning headers."
            )

        frame.columns = UNSW_COLUMNS
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    df.columns = [
        normalize_column_name(str(column))
        for column in df.columns
    ]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Normalize attack categories.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(
        normalize_class_name
    )

    # In UNSW-NB15, benign traffic has an empty attack_cat
    # and Label == 0. Map those records to the Benign class.
    benign_mask = (
        df[TARGET_COLUMN].eq("Unknown")
        & pd.to_numeric(df["label"], errors="coerce").eq(0)
    )

    df.loc[benign_mask, TARGET_COLUMN] = "Benign"

    # Remove records whose attack category is still unknown.
    df = df[df[TARGET_COLUMN] != "Unknown"].copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    missing_cat = [
        column
        for column in CATEGORICAL_COLUMNS
        if column not in df.columns
    ]

    if missing_cat:
        raise ValueError(
            f"Missing expected categorical columns: {missing_cat}"
        )

    print(f"\nLoaded dataset: {len(df):,} rows")
    print("\nOriginal class distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    return df



# ============================================================
# Feature preparation and imputation
# ============================================================

def prepare_feature_types(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X = df[feature_columns].copy()

    categorical_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in X.columns
    ]

    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    # Normalize categorical values to strings so that mixed
    # integer/string values do not break categorical sampling.
    for column in categorical_columns:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("missing")
            .astype(str)
        )

    # Convert all remaining input features to numeric.
    for column in numeric_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace([np.inf, -np.inf], np.nan)

    return X, numeric_columns, categorical_columns


def fit_imputation(
    X_train: pd.DataFrame,
    X_other: list[pd.DataFrame],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Learn imputation values from training data only, then use
    those same values for train, validation, and test.
    """
    numeric_medians = {}

    for column in numeric_columns:
        median = X_train[column].median()
        numeric_medians[column] = (
            0.0 if pd.isna(median) else float(median)
        )

    categorical_modes = {}

    for column in categorical_columns:
        mode = X_train[column].mode(dropna=True)
        categorical_modes[column] = (
            str(mode.iloc[0])
            if not mode.empty
            else "missing"
        )

    def transform(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for column in numeric_columns:
            X[column] = X[column].fillna(
                numeric_medians[column]
            )

        for column in categorical_columns:
            X[column] = (
                X[column]
                .fillna(categorical_modes[column])
                .astype(str)
            )

        return X

    return transform(X_train), [
        transform(frame) for frame in X_other
    ]


# ============================================================
# Split utilities
# ============================================================

def stratified_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a class into approximately 70% train, 15% validation,
    and 15% test, using seed 42.
    """
    train_df, temporary_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        shuffle=True,
        stratify=df[TARGET_COLUMN],
    )

    val_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=SEED,
        shuffle=True,
        stratify=temporary_df[TARGET_COLUMN],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_splits(
    df: pd.DataFrame,
    config_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    known_classes = CONFIG_CLASSES[config_name]
    original_unknowns = ORIGINAL_UNKNOWN_CLASSES[config_name]

    allowed_classes = set(known_classes) | set(original_unknowns)
    observed_classes = set(df[TARGET_COLUMN].unique())

    missing_classes = allowed_classes - observed_classes

    if missing_classes:
        raise ValueError(
            f"Configuration {config_name} is missing expected "
            f"classes: {sorted(missing_classes)}"
        )

    if PSEUDO_UNKNOWN_CLASS not in known_classes:
        raise ValueError(
            f"Pseudo-unknown class {PSEUDO_UNKNOWN_CLASS} "
            f"is not defined in configuration {config_name}."
        )

    # Keep only the classes assigned to this configuration.
    relevant_df = df[
        df[TARGET_COLUMN].isin(allowed_classes)
    ].copy()

    # Cap each originally known class BEFORE splitting.
    capped_frames = []

    for class_name in known_classes:
        class_df = relevant_df[
            relevant_df[TARGET_COLUMN] == class_name
        ].copy()

        if len(class_df) > MAX_SAMPLES_PER_CLASS:
            class_df = class_df.sample(
                n=MAX_SAMPLES_PER_CLASS,
                random_state=SEED,
            )

        capped_frames.append(class_df)

    capped_known_df = pd.concat(
        capped_frames,
        ignore_index=True,
    )

    # Split the five actual training classes using 70/15/15.
    # Split DoS 50/50 between validation and test.
    # DoS must never appear in training.
    train_parts = []
    val_parts = []
    test_parts = []

    dos_validation_count = 0
    dos_test_count = 0

    for class_name in known_classes:
        class_df = capped_known_df[
            capped_known_df[TARGET_COLUMN] == class_name
        ].copy()

        if class_name == PSEUDO_UNKNOWN_CLASS:
            dos_val, dos_test = train_test_split(
                class_df,
                test_size=0.50,
                random_state=SEED,
                shuffle=True,
            )

            val_parts.append(dos_val)
            test_parts.append(dos_test)

            dos_validation_count = len(dos_val)
            dos_test_count = len(dos_test)

        else:
            class_train, class_val, class_test = stratified_split(
                class_df
            )

            train_parts.append(class_train)
            val_parts.append(class_val)
            test_parts.append(class_test)

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    val_df = pd.concat(
        val_parts,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True,
    )

    # Original unknown classes appear ONLY in test.
    original_unknown_df = relevant_df[
        relevant_df[TARGET_COLUMN].isin(original_unknowns)
    ].copy()

    test_df = pd.concat(
        [test_df, original_unknown_df],
        ignore_index=True,
    )

    # Shuffle each split reproducibly.
    train_df = train_df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    val_df = val_df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    test_df = test_df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    # Validate the intended class protocol.
    train_classes = set(train_df[TARGET_COLUMN].unique())
    val_classes = set(val_df[TARGET_COLUMN].unique())
    test_classes = set(test_df[TARGET_COLUMN].unique())

    expected_train_classes = (
        set(known_classes) - {PSEUDO_UNKNOWN_CLASS}
    )

    if train_classes != expected_train_classes:
        raise RuntimeError(
            f"Unexpected train classes for config {config_name}: "
            f"{sorted(train_classes)}"
        )

    if PSEUDO_UNKNOWN_CLASS in train_classes:
        raise RuntimeError("DoS must not appear in training.")

    if PSEUDO_UNKNOWN_CLASS not in val_classes:
        raise RuntimeError("DoS is missing from validation.")

    if PSEUDO_UNKNOWN_CLASS not in test_classes:
        raise RuntimeError("DoS is missing from test.")

    if val_classes & set(original_unknowns):
        raise RuntimeError(
            "Original unknown classes leaked into validation."
        )

    if train_classes & set(original_unknowns):
        raise RuntimeError(
            "Original unknown classes leaked into training."
        )

    if not set(original_unknowns).issubset(test_classes):
        raise RuntimeError(
            "One or more original unknown classes are missing "
            "from test."
        )

    # Record split information for reproducibility.
    metadata = {
        "config": config_name,
        "seed": SEED,
        "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
        "pseudo_unknown_class": PSEUDO_UNKNOWN_CLASS,
        "original_known_classes": known_classes,
        "train_known_classes": sorted(expected_train_classes),
        "original_unknown_classes": original_unknowns,
        "dos_samples_in_train": 0,
        "dos_samples_in_validation": dos_validation_count,
        "dos_samples_in_test": dos_test_count,
    }

    return train_df, val_df, test_df, metadata


# ============================================================
# SMOTENC
# ============================================================

def apply_smote(
    train_df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """
    Balance the five training classes to 14,000 samples each.
    DoS is not present in train and is never synthesized.
    """
    X = train_df.drop(columns=[TARGET_COLUMN]).copy()
    y = train_df[TARGET_COLUMN].copy()

    if PSEUDO_UNKNOWN_CLASS in set(y.unique()):
        raise RuntimeError(
            "DoS must not be present before applying SMOTENC."
        )

    sampling_strategy = {
        class_name: SMOTE_TARGET_PER_CLASS
        for class_name, count in y.value_counts().items()
        if count < SMOTE_TARGET_PER_CLASS
    }

    if not sampling_strategy:
        return train_df.copy()

    categorical_indices = [
        X.columns.get_loc(column)
        for column in categorical_columns
    ]

    sampler = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=sampling_strategy,
        random_state=SEED,
        k_neighbors=5,
    )

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    result = pd.DataFrame(
        X_resampled,
        columns=X.columns,
    )
    result[TARGET_COLUMN] = np.asarray(y_resampled)

    # Restore the original column order.
    result = result[train_df.columns]

    return result.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)


# ============================================================
# Saving and reporting
# ============================================================

def save_variant(
    config_name: str,
    variant_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: dict,
) -> None:
    output_dir = (
        PROCESSED_DIR / f"config_{config_name}" / variant_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    metadata = {
        **metadata,
        "variant": variant_name,
        "split_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "class_counts": {
            "train": {
                str(k): int(v)
                for k, v in train_df[TARGET_COLUMN]
                .value_counts().items()
            },
            "validation": {
                str(k): int(v)
                for k, v in val_df[TARGET_COLUMN]
                .value_counts().items()
            },
            "test": {
                str(k): int(v)
                for k, v in test_df[TARGET_COLUMN]
                .value_counts().items()
            },
        },
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print(f"\nSaved: {output_dir}")
    print(
        f"  Train={len(train_df):,} | "
        f"Validation={len(val_df):,} | "
        f"Test={len(test_df):,}"
    )

    for split_name, split_df in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:
        print(f"\n{split_name} class distribution:")
        print(
            split_df[TARGET_COLUMN]
            .value_counts()
            .sort_index()
            .to_string()
        )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataset()
    X_all, numeric_columns, categorical_columns = (
        prepare_feature_types(raw_df)
    )

    # Work on features and target in the same row order.
    raw_df = raw_df.reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)

    for config_name in ("A", "C"):
        print("\n" + "=" * 65)
        print(f"CONFIGURATION {config_name}")
        print("=" * 65)

        train_df, val_df, test_df, metadata = build_splits(
            raw_df,
            config_name,
        )

        # Select features using the split row indices after reset.
        # build_splits resets indices, so use the row IDs to preserve
        # the exact corresponding feature rows.
        #
        # Attach temporary row IDs before rebuilding splits below.
        # (The split helper operates on this same indexed dataframe.)

        # Rebuild with explicit source-row IDs to avoid any mismatch.
        indexed_df = raw_df.copy()
        indexed_df["_source_row_id"] = np.arange(len(indexed_df))

        train_df, val_df, test_df, metadata = build_splits(
            indexed_df,
            config_name,
        )

        # _source_row_id is bookkeeping only; never a model feature.
        train_ids = train_df["_source_row_id"].to_numpy()
        val_ids = val_df["_source_row_id"].to_numpy()
        test_ids = test_df["_source_row_id"].to_numpy()

        y_train = train_df[TARGET_COLUMN].reset_index(drop=True)
        y_val = val_df[TARGET_COLUMN].reset_index(drop=True)
        y_test = test_df[TARGET_COLUMN].reset_index(drop=True)

        X_train = X_all.iloc[train_ids].reset_index(drop=True)
        X_val = X_all.iloc[val_ids].reset_index(drop=True)
        X_test = X_all.iloc[test_ids].reset_index(drop=True)

        X_train, [X_val, X_test] = fit_imputation(
            X_train,
            [X_val, X_test],
            numeric_columns,
            categorical_columns,
        )

        train_clean = X_train.copy()
        train_clean[TARGET_COLUMN] = y_train

        val_clean = X_val.copy()
        val_clean[TARGET_COLUMN] = y_val

        test_clean = X_test.copy()
        test_clean[TARGET_COLUMN] = y_test

        # Drop bookkeeping fields from the split data.
        for split_df in (train_clean, val_clean, test_clean):
            split_df.drop(
                columns=["_source_row_id"],
                errors="ignore",
                inplace=True,
            )

        # Save without SMOTE.
        save_variant(
            config_name,
            "without_smote",
            train_clean,
            val_clean,
            test_clean,
            metadata,
        )

        # SMOTE only on training data.
        train_smote = apply_smote(
            train_clean,
            numeric_columns,
            categorical_columns,
        )

        if PSEUDO_UNKNOWN_CLASS in set(
            train_smote[TARGET_COLUMN].unique()
        ):
            raise RuntimeError(
                "DoS unexpectedly appeared in SMOTE training data."
            )

        save_variant(
            config_name,
            "with_smote",
            train_smote,
            val_clean,
            test_clean,
            metadata,
        )

    print("\nPreprocessing completed successfully.")
    print(f"Output directory: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
