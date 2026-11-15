import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from utilis.experiment import ANOMALY_LABEL, UNKNOWN_LABEL, compute_outputs, load_experiment, load_split
from utilis.plots import plot_confusion_matrix, plot_outcome_table
from utilis.rejection import (
    METHODS,
    METRICS,
    BenignReference,
    CentroidModel,
    apply_anomaly_rule,
    apply_rule,
    compute_anomaly_scores,
    compute_scores,
    format_report,
    open_set_report,
    outcome_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained and calibrated model with the open-set rejection rule.")
    parser.add_argument("--experiment", type=Path, required=True, help="Experiment directory written by train.py --experiment-name.")
    parser.add_argument("--metric", choices=METRICS, default="euclidean", help="Which calibration file to use: rejection_<metric>.json (or anomaly_<metric>.json).")
    parser.add_argument("--anomaly", action="store_true", help="Use the ANOMALY-mode calibration (Benign centroid + entropy).")
    parser.add_argument("--split", choices=["validation", "test"], default="validation", help="Data split to evaluate (default: validation).")
    parser.add_argument("--final", action="store_true", help="Required for --split test: confirms the protocol is frozen.")
    parser.add_argument("--batch-size", "-b", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str.upper, choices=["FP32", "FP16", "BF16"], default="FP32", help="Use automatic mixed precision during inference when supported.")
    parser.add_argument("--cast", type=str.upper, choices=["FP32", "FP16", "BF16"], default=None, help="Cast model weights and inputs to this dtype instead of using autocast.")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows.")

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.cast is not None and args.precision != "FP32":
        parser.error("--cast and --precision (autocast) cannot be used together")
    if args.split == "test" and not args.final:
        parser.error("--split test requires --final. Calibrate first and only run the test split once the protocol is frozen.")

    return args


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    prefix = "anomaly" if args.anomaly else "rejection"
    calibration_path = args.experiment / f"{prefix}_{args.metric}.json"
    if not calibration_path.is_file():
        flag = " --anomaly" if args.anomaly else ""
        raise FileNotFoundError(f"{calibration_path} not found. Run calibrate.py --metric {args.metric}{flag} first.")

    with open(calibration_path, encoding="utf-8") as f:
        calibration = json.load(f)

    experiment = load_experiment(args.experiment, calibration["checkpoint"])
    if experiment["checkpoint_epoch"] != calibration["checkpoint_epoch"]:
        raise RuntimeError(
            f"{calibration['checkpoint']}.pt is from epoch {experiment['checkpoint_epoch']}, but the calibration "
            f"was made with epoch {calibration['checkpoint_epoch']}. Re-run calibrate.py."
        )

    class_names = experiment["class_names"]

    X, y = load_split(experiment, args.split)
    y = np.asarray(y, dtype=object)

    logits, embeddings = compute_outputs(experiment, X, args.batch_size, args.precision, args.cast)
    if args.anomaly:
        scores = compute_anomaly_scores(logits, embeddings, BenignReference.from_dict(calibration["benign_reference"]))
        reject_label = ANOMALY_LABEL
    else:
        scores = compute_scores(logits, embeddings, CentroidModel.from_dict(calibration["centroids"]))
        reject_label = UNKNOWN_LABEL
    closed_labels = np.asarray(class_names, dtype=object)[scores["pred"]]

    tag = f"Config {experiment['dataset']} | {experiment['loss_type']} | {args.metric}" + (" | ANOMALY mode" if args.anomaly else "")

    print(f"Experiment: {args.experiment} ({calibration['checkpoint']}.pt, epoch {experiment['checkpoint_epoch']})")
    print(f"Split: {args.split} | Samples: {len(y)} | Device: {experiment['device']}")
    print("\nClass counts in this split:")
    print(pd.Series(y).value_counts().to_string())

    # Closed-set view: known classes only, before any rejection.
    known_mask = np.isin(y, class_names)
    print("\nClosed-set report on known classes (no rejection):")
    print(classification_report(y[known_mask], closed_labels[known_mask], labels=class_names, digits=4, zero_division=0))

    index = {name: i for i, name in enumerate(class_names)}
    plot_confusion_matrix(
        y_true=[index[c] for c in y[known_mask]],
        y_pred=scores["pred"][known_mask],
        class_names=class_names,
        title=f"{tag}\n{args.split}, known classes, no rejection",
        save_path=args.experiment / f"{args.split}_closed_set_confusion.png",
        show=not args.no_show,
    )

    # Open-set view: each rejection method with its frozen thresholds.
    results = {}
    methods = ["anomaly"] if args.anomaly else list(METHODS)
    for method in methods:
        selection = calibration["methods"][method]
        if not selection.get("feasible"):
            print(f"\n[{method}] skipped: not feasible at calibration time.")
            continue

        if args.anomaly:
            thresholds = {"tau_h": selection["tau_h"], "tau_b": selection["tau_b"]}
            final_labels = apply_anomaly_rule(scores, class_names, **thresholds)
        else:
            thresholds = {"tau_c": selection["tau_c"], "tau_d": selection["tau_d"]}
            final_labels = apply_rule(scores, class_names, method, **thresholds, require_agreement=calibration["require_agreement"])

        report = open_set_report(y, final_labels, closed_labels, class_names, reject_label=reject_label)
        table = outcome_table(y, final_labels, class_names, reject_label=reject_label)
        results[method] = {**thresholds, "report": report, "outcomes": table.to_dict()}

        print(f"\n[{method}] " + " | ".join(f"{k} = {v:.4f}" for k, v in thresholds.items()))
        print(format_report(report))
        print(table.to_string())

        plot_outcome_table(
            table,
            title=f"{tag} | {args.split}, {method} rule",
            save_path=args.experiment / f"{args.split}_outcomes_{'anomaly' if args.anomaly else method}_{args.metric}.png",
            show=(not args.no_show) and method in ("combined", "anomaly"),
        )

    out_path = args.experiment / f"results_{args.split}_{prefix}_{args.metric}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "metric": args.metric,
                "calibration_file": calibration_path.name,
                "reject_label": reject_label,
                "precision": args.precision,
                "cast": args.cast,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved {out_path.name} and figures to {args.experiment}")


if __name__ == "__main__":
    main()
