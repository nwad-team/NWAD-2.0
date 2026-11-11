import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from utilis.experiment import PSEUDO_UNKNOWN_CLASS, compute_outputs, encode_known, load_experiment, load_split
from utilis.plots import plot_outcome_table, plot_score_histograms, plot_tradeoff_curves
from utilis.rejection import (
    METHODS,
    METRICS,
    CentroidModel,
    apply_rule,
    auroc,
    compute_scores,
    format_report,
    open_set_report,
    outcome_table,
    search_thresholds,
    subset,
    tradeoff_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fit centroids and calibrate rejection thresholds on validation data.")
    parser.add_argument("--experiment", type=Path, required=True, help="Experiment directory written by train.py --experiment-name.")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--metric", choices=METRICS, default="euclidean", help="Distance in embedding space (default: euclidean).")
    parser.add_argument("--max-frr", type=float, default=0.05, help="Maximum false rejection of known validation samples (default: 0.05).")
    parser.add_argument("--scale-quantile", type=float, default=0.95, help="Per-class distance scale: this quantile of training distances to the own centroid (default: 0.95).")
    parser.add_argument("--require-agreement", action="store_true", help="Also reject when the nearest centroid is not the predicted class (ablation; off by default).")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--precision", type=str.upper, choices=["FP32", "FP16", "BF16"], default="FP32")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows.")

    args = parser.parse_args()
    if not 0.0 < args.max_frr < 0.5:
        parser.error("--max-frr must be in (0, 0.5)")
    if not 0.5 <= args.scale_quantile < 1.0:
        parser.error("--scale-quantile must be in [0.5, 1)")
    return args


def main():
    args = parse_args()
    require_agreement = args.require_agreement

    experiment = load_experiment(args.experiment, args.checkpoint)
    class_names = experiment["class_names"]
    num_classes = len(class_names)

    print(f"Experiment: {args.experiment} ({args.checkpoint}.pt, epoch {experiment['checkpoint_epoch']})")
    print(f"Dataset: {experiment['dataset']} | Features: {experiment['feature_set']} | Loss: {experiment['loss_type']}")
    print(f"Metric: {args.metric} | FRR budget: {args.max_frr:.0%} | Agreement required: {require_agreement}")

    # 1. Centroids from real training rows only.
    X_train, y_train = load_split(experiment, "train", real_rows_only=True)
    train_labels = encode_known(y_train, class_names)
    if np.any(train_labels < 0):
        raise ValueError("Training split contains classes outside the known set.")

    _, train_embeddings = compute_outputs(experiment, X_train, args.batch_size, args.precision)
    centroids = CentroidModel(args.metric, quantile=args.scale_quantile).fit(train_embeddings, train_labels, num_classes)
    print(f"Centroids fit on {len(X_train)} real training rows.")

    # 2. Validation scores: known classes + DoS pseudo-unknown.
    X_val, y_val = load_split(experiment, "validation")
    y_val = np.asarray(y_val, dtype=object)

    val_logits, val_embeddings = compute_outputs(experiment, X_val, args.batch_size, args.precision)
    scores = compute_scores(val_logits, val_embeddings, centroids)

    known_mask = np.isin(y_val, class_names)
    pseudo_mask = y_val == PSEUDO_UNKNOWN_CLASS
    unexpected = sorted(set(y_val[~known_mask & ~pseudo_mask]))
    if unexpected:
        raise ValueError(f"Validation contains classes that are neither known nor DoS: {unexpected}")

    known_scores = subset(scores, known_mask)
    pseudo_scores = subset(scores, pseudo_mask)
    print(f"Validation: {known_mask.sum()} known, {pseudo_mask.sum()} DoS")
    print(f"Known samples whose nearest centroid disagrees with the prediction: {1 - known_scores['agree'].mean():.2%}")
    print(f"DoS samples whose nearest centroid disagrees with the prediction:   {1 - pseudo_scores['agree'].mean():.2%}")

    aurocs = auroc(known_scores, pseudo_scores)
    print(f"\nAUROC known vs DoS | confidence: {aurocs['confidence']:.4f} | distance: {aurocs['distance']:.4f}")

    # 3. Threshold search per method.
    closed_labels = np.asarray(class_names, dtype=object)[scores["pred"]]
    results = {}
    curves = {}
    chosen = {}

    for method in METHODS:
        selection = search_thresholds(known_scores, pseudo_scores, method, args.max_frr, require_agreement)
        curves[method] = tradeoff_curve(known_scores, pseudo_scores, method, require_agreement)

        if not selection["feasible"]:
            print(f"\n[{method}] no thresholds satisfy FRR <= {args.max_frr:.0%} (agreement alone rejects too many knowns).")
            results[method] = selection
            continue

        final_labels = apply_rule(scores, class_names, method, selection["tau_c"], selection["tau_d"], require_agreement)
        report = open_set_report(y_val, final_labels, closed_labels, class_names)
        selection["validation_report"] = report
        results[method] = selection
        chosen[method] = (selection["known_false_rejection"], selection["pseudo_unknown_rejection"])

        print(f"\n[{method}] tau_c = {selection['tau_c']:.4f} | tau_d = {selection['tau_d']:.4f}")
        print(format_report(report))

    # 4. Save the frozen calibration and figures.
    out_json = args.experiment / f"rejection_{args.metric}.json"
    tag = f"Config {experiment['dataset']} | {experiment['loss_type']} | {args.metric}"

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "experiment": str(args.experiment),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": experiment["checkpoint_epoch"],
        "dataset": experiment["dataset"],
        "feature_set": experiment["feature_set"],
        "loss_type": experiment["loss_type"],
        "class_names": class_names,
        "pseudo_unknown_class": PSEUDO_UNKNOWN_CLASS,
        "max_frr": args.max_frr,
        "require_agreement": require_agreement,
        "auroc_known_vs_pseudo": aurocs,
        "centroids": centroids.to_dict(),
        "methods": results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        # tau_d = inf for confidence-only is written as the JSON token Infinity.
        json.dump(payload, f, indent=2)

    show = not args.no_show
    plot_tradeoff_curves(
        curves, chosen, args.max_frr,
        title=f"{tag}\nValidation trade-off (AUROC conf {aurocs['confidence']:.3f}, dist {aurocs['distance']:.3f})",
        save_path=args.experiment / f"calibration_curves_{args.metric}.png", show=show,
    )

    combined = results.get("combined", {})
    plot_score_histograms(
        known_scores, pseudo_scores,
        {"tau_c": combined.get("tau_c"), "tau_d": combined.get("tau_d")},
        title=f"{tag} | validation scores (combined thresholds)",
        save_path=args.experiment / f"calibration_scores_{args.metric}.png", show=show,
    )

    if combined.get("feasible"):
        final_labels = apply_rule(scores, class_names, "combined", combined["tau_c"], combined["tau_d"], require_agreement)
        plot_outcome_table(
            outcome_table(y_val, final_labels, class_names),
            title=f"{tag} | validation, combined rule",
            save_path=args.experiment / f"calibration_outcomes_{args.metric}.png", show=show,
        )

    print(f"\nSaved {out_json.name} and calibration figures to {args.experiment}")


if __name__ == "__main__":
    main()
