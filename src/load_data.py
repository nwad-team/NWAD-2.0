# Data loading for NWAD-2.0
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_FILES = ["UNSW-NB15_1.csv", "UNSW-NB15_2.csv", "UNSW-NB15_3.csv", "UNSW-NB15_4.csv"]
FEATURES_FILE = "NUSW-NB15_features.csv"

def load_feature_names():
    feat_path = DATA_DIR / FEATURES_FILE
    feat_df = pd.read_csv(feat_path, encoding="latin1")
    feat_df.columns = feat_df.columns.str.strip()
    return feat_df["Name"].str.strip().tolist()

def load_dataset():
    col_names = load_feature_names()
    frames = []
    for fname in DATA_FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"Missing: {fname}")
            continue
        print(f"Loading: {fname}")
        df = pd.read_csv(fpath, header=None, names=col_names, low_memory=False)
        frames.append(df)
    full_df = pd.concat(frames, ignore_index=True)
    full_df = clean_attack_cat(full_df)
    return full_df

def clean_attack_cat(df):
    df["attack_cat"]=df["attack_cat"].astype(str).str.strip()
    df["attack_cat"]=df["attack_cat"].replace({"Backdoors": "Backdoor", "nan": pd.NA})
    df = df.rename(columns={"ct_src_ ltm": "ct_src_ltm"})
    return df

def summarize(df):
    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: \n{list(df.columns)}")

    for col_name in ["label", "attack_cat"]:
        col = next((c for c in df.columns if c.lower() == col_name), None)
        if col:
            print(f"\nClass distribution ('{col}'):")
            print(df[col].value_counts(dropna=False))
        else:
            print(f"\nNo '{col_name}' column found")

    n_missing = df.isnull().sum().sum()
    n_inf = df.select_dtypes(include="number").apply(lambda c: (c == float("inf")).sum()).sum()
    print(f"\nMissing Values: {n_missing}")
    print(f"Infinite values: {n_inf}")

if __name__ == "__main__":
    df = load_dataset()
    summarize(df)