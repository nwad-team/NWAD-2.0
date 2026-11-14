"""
Data only:
    python overlap_analysis.py --dataset C
With the model:
    python overlap_analysis.py --dataset C --experiment checkpoints/C_behavioral_ce --metric mahalanobis
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_loading import CONFIG_CLASSES, load_processed_split


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze class overlap in UNSW-NB15 features and MLP embeddings.")
    parser.add_argument("--dataset", choices=[k for k in CONFIG_CLASSES if "-" not in k], required=True)
    parser.add_argument("--feature-set", choices=["behavioral", "shortcut", "combined"], default="behavioral")
    parser.add_argument("--experiment", type=Path, default=None, help="Optional trained experiment for Part 2.")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--metric", choices=["euclidean", "mahalanobis"], default="mahalanobis")
    parser.add_argument("--top-pairs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results") / "overlap")
    return parser.parse_args()


def duplicate_conflicts(X: pd.DataFrame, y: pd.Series):
    """Rows whose full feature vector appears under more than one label."""
    df = X.copy()
    df["_label"] = y.values
    feature_cols = list(X.columns)
    key = pd.util.hash_pandas_object(df[feature_cols], index=False)
    df["_key"] = key.values

    labels_per_key = df.groupby("_key")["_label"].nunique()
    conflicting_keys = labels_per_key[labels_per_key > 1].index
    df["_conflict"] = df["_key"].isin(conflicting_keys)

    per_class = (
        df.groupby("_label")
        .agg(rows=("_label", "size"), conflicting_rows=("_conflict", "sum"))
        .assign(conflicting_share=lambda t: t["conflicting_rows"] / t["rows"])
        .sort_values("conflicting_share", ascending=False)
    )

    # Pair counts: for each conflicting vector, every pair of labels it carries.
    pairs = {}
    for _, labels in df[df["_conflict"]].groupby("_key")["_label"]:
        counts = labels.value_counts()
        names = sorted(counts.index)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs[(a, b)] = pairs.get((a, b), 0) + int(counts[a] + counts[b])
    pair_df = (
        pd.DataFrame([{"class_a": a, "class_b": b, "rows_involved": n} for (a, b), n in pairs.items()])
        .sort_values("rows_involved", ascending=False)
        if pairs else pd.DataFrame(columns=["class_a", "class_b", "rows_involved"])
    )
    return per_class, pair_df


def cohens_d(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2).replace(0, np.nan)
    return ((a.mean() - b.mean()) / pooled).dropna()


def embedding_overlap(args, out_dir: Path):
    from utilis.experiment import compute_outputs, encode_known, load_experiment, load_split
    from utilis.rejection import CentroidModel, softmax

    experiment = load_experiment(args.experiment, args.checkpoint)
    class_names = experiment["class_names"]

    X_train, y_train = load_split(experiment, "train", real_rows_only=True)
    train_labels = encode_known(y_train, class_names)
    _, train_emb = compute_outputs(experiment, X_train)
    centroids = CentroidModel(args.metric).fit(train_emb, train_labels, len(class_names))

    X_val, y_val = load_split(experiment, "validation")
    y_val = np.asarray(y_val, dtype=object)
    known = np.isin(y_val, class_names)
    X_known, y_known = X_val[known].reset_index(drop=True), y_val[known]

    logits, emb = compute_outputs(experiment, X_known)
    names = np.asarray(class_names, dtype=object)
    nearest = names[centroids.distances(emb).argmin(axis=1)]
    pred = names[softmax(logits).argmax(axis=1)]

    confusion = pd.crosstab(pd.Series(y_known, name="true"), pd.Series(nearest, name="nearest_centroid"), normalize="index")
    print("\nNearest-centroid confusion on validation known classes, row = true class")
    print(confusion.round(3).to_string())

    misplaced = nearest != y_known
    samples = X_known[misplaced].copy()
    samples.insert(0, "predicted", pred[misplaced])
    samples.insert(0, "nearest_centroid", nearest[misplaced])
    samples.insert(0, "true", y_known[misplaced])
    samples_path = out_dir / f"misplaced_validation_{args.metric}.csv"
    samples.to_csv(samples_path, index=False)
    print(f"\n{int(misplaced.sum())} of {len(y_known)} validation flows sit closest to another class centroid. Saved to {samples_path}")

    pair_counts = (
        pd.DataFrame({"true": y_known[misplaced], "nearest": nearest[misplaced]})
        .value_counts().reset_index(name="flows").head(args.top_pairs)
    )
    numeric = X_known.select_dtypes(include="number")
    for _, row in pair_counts.iterrows():
        a, b, n = row["true"], row["nearest"], row["flows"]
        moved = numeric[(y_known == a) & (nearest == b)]
        stayed = numeric[(y_known == a) & (nearest == a)]
        target = numeric[(y_known == b) & (nearest == b)]
        if len(moved) < 2 or len(stayed) < 2 or len(target) < 2:
            continue
        d_own = cohens_d(moved, stayed).abs().sort_values(ascending=False).head(10)
        d_target = cohens_d(moved, target).abs().reindex(d_own.index)
        table = pd.DataFrame({"d_vs_own_class": d_own, "d_vs_target_class": d_target})
        print(f"\n{a} flows placed near {b}: {n}")
        print("Features where they differ most from correctly placed " + a + ", and how close they are to " + b)
        print(table.round(2).to_string())
        table.to_csv(out_dir / f"pair_{a}_to_{b}_{args.metric}.csv")


def main():
    args = parse_args()
    out_dir = args.out / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Part 1: every split of this config, so unknown classes are included too.
    frames = [load_processed_split(args.dataset, split, args.feature_set) for split in ("train", "validation", "test")]
    X = pd.concat([f[0] for f in frames], ignore_index=True)
    y = pd.concat([f[1] for f in frames], ignore_index=True)

    per_class, pairs = duplicate_conflicts(X, y)
    print(f"Config {args.dataset} | {args.feature_set} features | {len(X)} rows")
    print("\nRows whose exact feature vector also appears under another label")
    print(per_class.round(4).to_string())
    print("\nLabel pairs sharing identical feature vectors")
    print(pairs.head(15).to_string(index=False))
    per_class.to_csv(out_dir / f"duplicate_conflicts_{args.feature_set}.csv")
    pairs.to_csv(out_dir / f"duplicate_pairs_{args.feature_set}.csv", index=False)

    if args.experiment is not None:
        embedding_overlap(args, out_dir)

    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()