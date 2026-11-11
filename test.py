import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from utilis.experiment import UNKNOWN_LABEL, compute_outputs, load_experiment, load_split
from utilis.plots import plot_confusion_matrix, plot_outcome_table
from utilis.rejection import (
    METHODS,
    METRICS,
    CentroidModel,
    apply_rule,
    compute_scores,
    format_report,
    open_set_report,
    outcome_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained and calibrated model with the open-set rejection rule.")
    parser.add_argument("--experiment", type=Path, required=True, help="Experiment directory written by train.py --experiment-name.")
    parser.add_argument("--metric", choices=METRICS, default="euclidean", help="Which calibration file to use: rejection_<metric>.json.")
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

    calibration_path = args.experiment / f"rejection_{args.metric}.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"{calibration_path} not found. Run calibrate.py --metric {args.metric} first.")

    with open(calibration_path, encoding="utf-8") as f:
        calibration = json.load(f)

    experiment = load_experiment(args.experiment, calibration["checkpoint"])
    if experiment["checkpoint_epoch"] != calibration["checkpoint_epoch"]:
        raise RuntimeError(
            f"{calibration['checkpoint']}.pt is from epoch {experiment['checkpoint_epoch']}, but the calibration "
            f"was made with epoch {calibration['checkpoint_epoch']}. Re-run calibrate.py."
        )

    class_names = experiment["class_names"]
    centroids = CentroidModel.from_dict(calibration["centroids"])
    require_agreement = calibration["require_agreement"]

    X, y = load_split(experiment, args.split)
    y = np.asarray(y, dtype=object)

    logits, embeddings = compute_outputs(experiment, X, args.batch_size, args.precision, args.cast)
    scores = compute_scores(logits, embeddings, centroids)
    closed_labels = np.asarray(class_names, dtype=object)[scores["pred"]]

    tag = f"Config {experiment['dataset']} | {experiment['loss_type']} | {args.metric}"

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
    for method in METHODS:
        selection = calibration["methods"][method]
        if not selection.get("feasible"):
            print(f"\n[{method}] skipped: not feasible at calibration time.")
            continue

        final_labels = apply_rule(scores, class_names, method, selection["tau_c"], selection["tau_d"], require_agreement)
        report = open_set_report(y, final_labels, closed_labels, class_names)
        table = outcome_table(y, final_labels, class_names)
        results[method] = {"tau_c": selection["tau_c"], "tau_d": selection["tau_d"], "report": report, "outcomes": table.to_dict()}

        print(f"\n[{method}] tau_c = {selection['tau_c']:.4f} | tau_d = {selection['tau_d']:.4f}")
        print(format_report(report))
        print(table.to_string())

        plot_outcome_table(
            table,
            title=f"{tag} | {args.split}, {method} rule",
            save_path=args.experiment / f"{args.split}_outcomes_{method}_{args.metric}.png",
            show=(not args.no_show) and method == "combined",
        )

    out_path = args.experiment / f"results_{args.split}_{args.metric}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "metric": args.metric,
                "calibration_file": calibration_path.name,
                "unknown_label": UNKNOWN_LABEL,
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
