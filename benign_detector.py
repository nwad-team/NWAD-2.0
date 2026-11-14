
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utilis.experiment import PSEUDO_UNKNOWN_CLASS, compute_outputs, encode_known, load_experiment, load_split
from utilis.rejection import METHODS, METRICS, CentroidModel, apply_rule, compute_scores

BENIGN = "Benign"


def parse_args():
    parser = argparse.ArgumentParser(description="Benign-centered anomaly detection vs per-class centroid rejection.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--metric", choices=METRICS, default="mahalanobis")
    parser.add_argument("--max-fpr", type=float, default=0.05, help="Share of validation Benign flows flagged (default: 0.05).")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--final", action="store_true", help="Required for --split test.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("--split test requires --final. Compare on validation first.")
    if not 0.0 < args.max_fpr < 0.5:
        parser.error("--max-fpr must be in (0, 0.5)")
    return args


def binary_table(y_true, flagged, class_names):
    rows = []
    for cls in pd.unique(y_true):
        mask = y_true == cls
        group = "Benign" if cls == BENIGN else ("known attack" if cls in class_names else "unknown attack")
        rows.append({
            "class": cls,
            "group": group,
            "n": int(mask.sum()),
            "flagged": float(flagged[mask].mean()),
            "accepted_as_benign": float(1 - flagged[mask].mean()),
        })
    order = {"Benign": 0, "known attack": 1, "unknown attack": 2}
    df = pd.DataFrame(rows)
    df["_order"] = df["group"].map(order)
    return df.sort_values(["_order", "class"]).drop(columns="_order").reset_index(drop=True)


def summary(y_true, flagged, class_names):
    benign = y_true == BENIGN
    known_attack = np.isin(y_true, class_names) & ~benign
    unknown = ~np.isin(y_true, class_names)
    out = {"benign_fpr": float(flagged[benign].mean())}
    if known_attack.any():
        out["known_attack_detection"] = float(flagged[known_attack].mean())
    if unknown.any():
        out["unknown_detection"] = float(flagged[unknown].mean())
        out["unknown_accepted_as_benign"] = float(1 - flagged[unknown].mean())
    return out


def main():
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiment = load_experiment(args.experiment, args.checkpoint)
    class_names = experiment["class_names"]
    if BENIGN not in class_names:
        raise ValueError(f"{BENIGN} is not a known class in this experiment: {class_names}")
    benign_idx = class_names.index(BENIGN)

    # 1. Benign-only centroid from real training rows.
    X_train, y_train = load_split(experiment, "train", real_rows_only=True)
    train_labels = encode_known(y_train, class_names)
    _, train_emb = compute_outputs(experiment, X_train, args.batch_size)
    benign_train = train_emb[train_labels == benign_idx]
    benign_model = CentroidModel(args.metric).fit(benign_train, np.zeros(len(benign_train), dtype=np.int64), 1)
    print(f"Benign centroid fit on {len(benign_train)} training rows ({args.metric}).")

    # 2. Threshold from validation Benign only.
    X_val, y_val = load_split(experiment, "validation")
    y_val = np.asarray(y_val, dtype=object)
    _, val_emb = compute_outputs(experiment, X_val, args.batch_size)
    val_dist = benign_model.distances(val_emb)[:, 0]
    tau_benign = float(np.quantile(val_dist[y_val == BENIGN], 1 - args.max_fpr))
    print(f"tau_benign = {tau_benign:.4f} flags {args.max_fpr:.0%} of validation Benign.")

    # 3. Scores on the evaluation split.
    if args.split == "validation":
        X, y, emb = X_val, y_val, val_emb
        logits, _ = compute_outputs(experiment, X, args.batch_size)
    else:
        X, y = load_split(experiment, "test")
        y = np.asarray(y, dtype=object)
        logits, emb = compute_outputs(experiment, X, args.batch_size)

    dist = benign_model.distances(emb)[:, 0]
    flagged_benign_model = dist > tau_benign

    comparisons = {"benign_centered": {"tau_benign": tau_benign, "summary": summary(y, flagged_benign_model, class_names)}}
    tables = {"benign_centered": binary_table(y, flagged_benign_model, class_names)}

    is_attack = y != BENIGN
    comparisons["benign_centered"]["auroc_benign_vs_attack"] = float(roc_auc_score(is_attack, dist))

    # 4. Per-class rule from calibrate.py, scored the same binary way.
    calib_path = args.experiment / f"rejection_{args.metric}.json"
    if calib_path.is_file():
        with open(calib_path, encoding="utf-8") as f:
            calib = json.load(f)
        if calib["checkpoint_epoch"] != experiment["checkpoint_epoch"]:
            raise RuntimeError(f"{calib_path.name} was calibrated on a different checkpoint epoch.")
        centroids = CentroidModel.from_dict(calib["centroids"])
        scores = compute_scores(logits, emb, centroids)
        for method in METHODS:
            sel = calib["methods"].get(method, {})
            if not sel.get("feasible", False):
                continue
            labels = apply_rule(scores, class_names, method, sel["tau_c"], sel["tau_d"], calib["require_agreement"])
            flagged = labels != BENIGN
            key = f"per_class_{method}"
            comparisons[key] = {"tau_c": sel["tau_c"], "tau_d": sel["tau_d"], "summary": summary(y, flagged, class_names)}
            tables[key] = binary_table(y, flagged, class_names)
    else:
        print(f"\n{calib_path.name} not found, run calibrate.py --metric {args.metric} first to include the per-class rule.")

    # 5. Report.
    print(f"\nConfig {experiment['dataset']} | {experiment['loss_type']} | {args.metric} | {args.split}")
    summary_df = pd.DataFrame({k: v["summary"] for k, v in comparisons.items()}).T
    print("\nBinary IDS view, flagged = not let through as Benign")
    print(summary_df.round(4).to_string())
    for key, table in tables.items():
        print(f"\n[{key}]")
        print(table.round(4).to_string(index=False))

    # 6. Figure: distance to the Benign centroid per true class.
    classes = list(pd.unique(y))
    classes.sort(key=lambda c: (c != BENIGN, c not in class_names, c))
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(classes) + 1.5))
    data = [np.log10(dist[y == c] + 1e-6) for c in classes]
    ax.boxplot(data, vert=False, showfliers=False)
    ax.set_yticks(range(1, len(classes) + 1))
    ax.set_yticklabels(classes)
    ax.axvline(np.log10(tau_benign), color="#D9752B", linestyle="--", label=f"threshold, {args.max_fpr:.0%} Benign FPR")
    ax.invert_yaxis()
    ax.set_xlabel("log10 scaled distance to Benign centroid")
    ax.set_title(f"Distance to the Benign centroid | Config {experiment['dataset']} | {args.split}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig_path = args.experiment / f"benign_distance_{args.metric}_{args.split}.png"
    fig.savefig(fig_path, dpi=150)
    if not args.no_show:
        plt.show()
    plt.close(fig)

    out_path = args.experiment / f"benign_vs_perclass_{args.metric}_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "created": datetime.now().isoformat(timespec="seconds"),
            "dataset": experiment["dataset"],
            "loss_type": experiment["loss_type"],
            "metric": args.metric,
            "split": args.split,
            "max_fpr": args.max_fpr,
            "comparisons": comparisons,
            "per_class_tables": {k: t.to_dict(orient="records") for k, t in tables.items()},
        }, f, indent=2)
    print(f"\nSaved {out_path.name} and {fig_path.name} to {args.experiment}")


if __name__ == "__main__":
    main()