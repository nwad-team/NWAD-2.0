import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data.data_loading import CONFIG_CLASSES, load_processed_split
from utilis.experiment import PSEUDO_UNKNOWN_CLASS, UNKNOWN_LABEL
from utilis.plots import plot_outcome_table
from utilis.rejection import (
    apply_rule,
    format_report,
    open_set_report,
    outcome_table,
    search_thresholds,
    subset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Open-set rejection for classical baselines.")
    parser.add_argument("--dataset", choices=list(CONFIG_CLASSES), required=True)
    parser.add_argument("--feature-set", choices=["behavioral", "shortcut", "combined"], default="behavioral")
    parser.add_argument("--model", choices=["random-forest", "logistic-regression"], required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--final", action="store_true", help="Required for --split test.")
    parser.add_argument("--max-frr", type=float, default=0.05, help="Max false rejection of known validation samples (default: 0.05).")
    parser.add_argument("--root", type=Path, default=Path("results") / "baselines")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("--split test requires --final. Calibrate on validation first.")
    return args


def confidence_scores(model, preprocessor, X):
    X_processed = np.asarray(preprocessor.transform(X), dtype=np.float32)
    probs = model.predict_proba(X_processed)
    pred = probs.argmax(axis=1)
    n = len(pred)
    return {
        "pred": pred,
        "conf": probs.max(axis=1),
        "dist_pred": np.zeros(n),
        "nearest": pred.copy(),
        "agree": np.ones(n, dtype=bool),
    }


def main():
    args = parse_args()
    class_names = CONFIG_CLASSES[args.dataset]
    model_dir = args.root / args.dataset / args.feature_set / args.model

    model = joblib.load(model_dir / "model.joblib")
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")

    if list(model.classes_) != list(range(len(class_names))):
        raise RuntimeError(f"Model classes {model.classes_} do not match {class_names}.")

    calibration_path = model_dir / "rejection_confidence.json"

    if args.split == "validation":
        X_val, y_val = load_processed_split(args.dataset, "validation", args.feature_set)
        y_val = np.asarray(y_val, dtype=object)
        scores_val = confidence_scores(model, preprocessor, X_val)

        known = np.isin(y_val, class_names)
        pseudo = y_val == PSEUDO_UNKNOWN_CLASS
        if not pseudo.any():
            raise ValueError(f"No {PSEUDO_UNKNOWN_CLASS} rows in validation, cannot calibrate.")

        selection = search_thresholds(
            subset(scores_val, known), subset(scores_val, pseudo), "confidence", args.max_frr
        )
        if not selection["feasible"]:
            raise RuntimeError("No feasible confidence threshold at this --max-frr.")

        with open(calibration_path, "w", encoding="utf-8") as f:
            json.dump({"dataset": args.dataset, "feature_set": args.feature_set, "model": args.model,
                       "max_frr": args.max_frr, "selection": selection}, f, indent=2)
        print(f"Calibrated tau_c = {selection['tau_c']:.4f} | known FRR = {selection['known_false_rejection']:.4f} "
              f"| DoS rejection = {selection['pseudo_unknown_rejection']:.4f}")
        X, y, scores = X_val, y_val, scores_val
    else:
        if not calibration_path.is_file():
            raise FileNotFoundError(f"{calibration_path} not found. Run with --split validation first.")
        with open(calibration_path, encoding="utf-8") as f:
            selection = json.load(f)["selection"]
        X, y = load_processed_split(args.dataset, "test", args.feature_set)
        y = np.asarray(y, dtype=object)
        scores = confidence_scores(model, preprocessor, X)

    closed_labels = np.asarray(class_names, dtype=object)[scores["pred"]]
    final_labels = apply_rule(scores, class_names, "confidence", selection["tau_c"], np.inf)
    report = open_set_report(y, final_labels, closed_labels, class_names)
    table = outcome_table(y, final_labels, class_names)

    print(f"\n{args.model} | Config {args.dataset} | {args.feature_set} | {args.split}")
    print(pd.Series(y).value_counts().to_string())
    print(format_report(report))
    print(table.to_string())

    plot_outcome_table(
        table,
        title=f"{args.model} | Config {args.dataset} | {args.split}, confidence rule",
        save_path=model_dir / f"{args.split}_outcomes_confidence.png",
        show=not args.no_show,
    )

    out_path = model_dir / f"results_{args.split}_confidence.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "unknown_label": UNKNOWN_LABEL, "tau_c": selection["tau_c"],
                   "report": report, "outcomes": table.to_dict()}, f, indent=2)
    print(f"\nSaved {out_path.name} to {model_dir}")


if __name__ == "__main__":
    main()