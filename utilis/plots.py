from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true,y_pred,class_names: list[str],title: str,save_path: Path | None = None,show: bool = True,):

    # Import here so training does not require a display backend unless used.
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    labels = list(range(len(class_names)))
    counts = confusion_matrix(y_true, y_pred, labels=labels)

    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals > 0,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")

    ax.set_xticks(labels, labels=class_names, rotation=35, ha="right")
    ax.set_yticks(labels, labels=class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    for i in labels:
        for j in labels:
            value = normalized[i, j]
            ax.text(
                j,
                i,
                f"{counts[i, j]}\n({value:.2f})",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > 0.5 else "black",
            )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    return counts


def _pyplot(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def _finish(fig, plt, save_path, show):
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_tradeoff_curves(curves: dict, chosen: dict, max_frr: float, title: str, save_path=None, show: bool = True):

    plt = _pyplot(show)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {"confidence": "tab:orange", "distance": "tab:blue", "combined": "tab:green"}

    for method, points in curves.items():
        ax.plot(points[:, 0], points[:, 1], label=method, color=colors.get(method), linewidth=1.8)
        if method in chosen:
            ax.scatter(*chosen[method], color=colors.get(method), edgecolor="black", zorder=5, s=60)

    ax.axvline(max_frr, color="gray", linestyle="--", linewidth=1, label=f"FRR budget {max_frr:.0%}")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Known-class false rejection rate (validation)")
    ax.set_ylabel("Pseudo-unknown (DoS) rejection rate")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    _finish(fig, plt, save_path, show)


def plot_score_histograms(known: dict, unknown: dict, thresholds: dict, title: str, save_path=None, show: bool = True):

    plt = _pyplot(show)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    panels = [
        ("conf", "Max softmax confidence", thresholds.get("tau_c"), np.linspace(0, 1, 51)),
        ("dist_pred", "Normalized distance to predicted-class centroid", thresholds.get("tau_d"), None),
    ]

    for ax, (key, label, tau, bins) in zip(axes, panels):
        if bins is None:
            upper = np.quantile(np.concatenate([known[key], unknown[key]]), 0.99)
            bins = np.linspace(0, upper, 51)
        ax.hist(np.clip(known[key], bins[0], bins[-1]), bins=bins, alpha=0.6, density=True, label="known classes", color="tab:blue")
        ax.hist(np.clip(unknown[key], bins[0], bins[-1]), bins=bins, alpha=0.6, density=True, label="DoS (pseudo-unknown)", color="tab:red")
        if tau is not None and np.isfinite(tau) and tau > 0:
            ax.axvline(tau, color="black", linestyle="--", linewidth=1.2, label=f"threshold {tau:.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle(title)
    _finish(fig, plt, save_path, show)


def plot_outcome_table(table, title: str, save_path=None, show: bool = True):

    plt = _pyplot(show)

    counts = table.to_numpy()
    totals = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)

    fig, ax = plt.subplots(figsize=(1.3 * counts.shape[1] + 2.5, 0.75 * counts.shape[0] + 2))
    image = ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")

    ax.set_xticks(range(counts.shape[1]), labels=table.columns, rotation=35, ha="right")
    ax.set_yticks(range(counts.shape[0]), labels=table.index)
    ax.set_xlabel("Output")
    ax.set_ylabel("True class")
    ax.set_title(title)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, f"{counts[i, j]}\n({fractions[i, j]:.2f})", ha="center", va="center", fontsize=8,
                    color="white" if fractions[i, j] > 0.5 else "black")

    _finish(fig, plt, save_path, show)
