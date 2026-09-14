# Basic data loading skeleton for the NWAD project
import pandas as pd
from pathlib import Path 

DATA_DIR = Path(__file__).parent.parent / "data"

def find_csv_files():
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}/")
        print("Download a dataset subset and place it there before running this")
    return csv_files

def load_dataset(filepath):
    print(f"Loading: {filepath.name}")
    df = pd.read_csv(filepath, low_memory=False)

    df.columns = df.columns.str.strip()

    return df

def summarize(df):
    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: \n{list(df.columns)}")

    label_col = next((c for c in df.columns if c.lower == "label"), None)
    if label_col:
        print(f"\nClass distribution ('{label_col}'):")
        print(df[label_col].val_counts())
    else: 
        print("\nNo 'Label' column found")


    n_missing = df.isnull().sum().sum()
    n_inf = df.select_dtypes(include="number").apply(lambda col: (col==float("inf")).sum()).sim()
    print(f"\nMissing Values: {n_missing}")
    print(f"Infinite values: {n_inf}")

if __name__ == "__main__":
    files = find_csv_files()
    for f in files:
        df = load_dataset(f)
        summarize(df)
        print("\n" + "=" * 60 + "\n")
