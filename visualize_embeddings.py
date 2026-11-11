import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utilis.experiment import PSEUDO_UNKNOWN_CLASS, compute_outputs, encode_known, load_experiment, load_split

# One fixed color per class across every configuration and figure.
CLASS_COLORS = {
    "Benign": "#4C72B0",
    "Generic": "#DD8452",
    "Exploits": "#55A868",
    "Fuzzers": "#C44E52",
    "Reconnaissance": "#8172B3",
    "Backdoor": "#937860",
    "Shellcode": "#DA8BC3",
    "DoS": "#222222",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Project embeddings to 2-D with PCA and/or UMAP.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--method", choices=["pca", "umap", "both"], default="both")
    parser.add_argument("--max-per-class", type=int, default=2000, help="Random points per class to fit and plot (default: 2000).")
    parser.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors (default: 30).")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def sample_per_class(labels, max_per_class, rng):
    keep = []
    for name in sorted(set(labels)):
        idx = np.flatnonzero(labels == name)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def compactness(embeddings, labels, class_names):
    
    centroids = {c: embeddings[labels == c].mean(axis=0) for c in class_names}
    rows = {}
    for c in class_names:
        points = embeddings[labels == c]
        spread = float(np.linalg.norm(points - centroids[c], axis=1).mean())
        nearest_name, separation = min(
            ((o, float(np.linalg.norm(centroids[c] - centroids[o]))) for o in class_names if o != c),
            key=lambda item: item[1],
        )
        rows[c] = {"spread": spread, "separation": separation, "nearest_class": nearest_name, "ratio": spread / separation}
    return rows, centroids


def plot_projection(train_2d, train_labels, val_2d, val_labels, centroids_2d, class_names, title, save_path, show):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=True, sharey=True)
    panels = [
        (axes[0], train_2d, train_labels, "Training embeddings (real rows, known classes)"),
        (axes[1], val_2d, val_labels, "Validation embeddings (known + DoS pseudo-unknown)"),
    ]

    order = class_names + [PSEUDO_UNKNOWN_CLASS]
    for ax, points, labels, subtitle in panels:
        for name in order:
            mask = labels == name
            if not mask.any():
                continue
            is_pseudo = name == PSEUDO_UNKNOWN_CLASS
            ax.scatter(
                points[mask, 0], points[mask, 1],
                s=10 if is_pseudo else 4,
                marker="x" if is_pseudo else "o",
                alpha=0.6 if is_pseudo else 0.35,
                linewidths=0.6 if is_pseudo else 0,
                color=CLASS_COLORS.get(name, "gray"),
                label=f"{name} (unseen)" if is_pseudo else name,
                rasterized=True,
            )
        for name in class_names:
            ax.scatter(*centroids_2d[name], marker="*", s=260, color=CLASS_COLORS.get(name, "gray"), edgecolor="black", linewidth=1.2, zorder=5)
        ax.set_title(subtitle)
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=len(labels), markerscale=3, frameon=False)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    fig.suptitle(title + "\nStars = class centroids (training embeddings, projected)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    experiment = load_experiment(args.experiment, args.checkpoint)
    class_names = experiment["class_names"]
    tag = f"Config {experiment['dataset']} | {experiment['feature_set']} | loss: {experiment['loss_type']}"

    X_train, y_train = load_split(experiment, "train", real_rows_only=True)
    X_val, y_val = load_split(experiment, "validation")
    y_train = np.asarray(y_train, dtype=object)
    y_val = np.asarray(y_val, dtype=object)

    _, train_emb = compute_outputs(experiment, X_train)
    _, val_emb = compute_outputs(experiment, X_val)

    # Compactness on the full 32-d embedding, using all rows.
    stats, centroids = compactness(train_emb, y_train, class_names)
    known_val = np.isin(y_val, class_names)
    sil_idx = sample_per_class(y_val[known_val], 1000, rng)
    silhouette = float(silhouette_score(val_emb[known_val][sil_idx], encode_known(y_val[known_val][sil_idx], class_names)))

    print(f"{tag}\n")
    print(f"{'class':<16}{'spread':>9}{'separation':>12}{'ratio':>8}  nearest")
    for name, row in stats.items():
        print(f"{name:<16}{row['spread']:>9.3f}{row['separation']:>12.3f}{row['ratio']:>8.3f}  {row['nearest_class']}")
    print(f"\nSilhouette (validation, known classes, Euclidean): {silhouette:.4f}")

    # Fit projections on a class-balanced sample of training rows only.
    train_idx = sample_per_class(y_train, args.max_per_class, rng)
    val_idx = sample_per_class(y_val, args.max_per_class, rng)
    centroid_matrix = np.stack([centroids[c] for c in class_names])

    methods = ["pca", "umap"] if args.method == "both" else [args.method]
    summary = {"compactness": stats, "silhouette_validation_known": silhouette}

    for method in methods:
        if method == "pca":
            projector = PCA(n_components=2, random_state=args.seed).fit(train_emb[train_idx])
            explained = projector.explained_variance_ratio_
            subtitle = f"PCA (explained variance {explained[0]:.1%} + {explained[1]:.1%})"
            summary["pca_explained_variance"] = explained.tolist()
        else:
            import umap

            projector = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=args.seed).fit(train_emb[train_idx])
            subtitle = f"UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})"

        train_2d = projector.transform(train_emb[train_idx])
        val_2d = projector.transform(val_emb[val_idx])
        centroid_2d = dict(zip(class_names, projector.transform(centroid_matrix)))

        save_path = args.experiment / f"embedding_{method}.png"
        plot_projection(train_2d, y_train[train_idx], val_2d, y_val[val_idx], centroid_2d, class_names,
                        f"{tag} | {subtitle}", save_path, show=not args.no_show)
        print(f"Saved {save_path}")

    with open(args.experiment / "embedding_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
