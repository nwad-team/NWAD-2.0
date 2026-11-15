"""Fit the reference centroids and set the rejection thresholds on validation data.

Two modes (see utilis/rejection.py for the exact rules):

  UNKNOWN mode (default): one centroid per known class, output UNKNOWN.
      python calibrate.py --experiment checkpoints/A_ce                      # budget search
      python calibrate.py --experiment checkpoints/A_ce --tau-c 0.75 --tau-d 1.5   # manual
      python calibrate.py --experiment checkpoints/A_ce --tau-d 1.5                # tau_d fixed, tau_c searched

  ANOMALY mode (--anomaly): only the Benign centroid, output ANOMALY.
      python calibrate.py --experiment checkpoints/A_ce --anomaly            # budget search
      python calibrate.py --experiment checkpoints/A_ce --anomaly --tau-h 0.4 --tau-b 2.0
      python calibrate.py --experiment checkpoints/A_ce --anomaly --tau-b 2.0      # tau_b fixed, tau_h searched (recommended)

Each threshold is either given by hand or, when omitted, searched so that at most
--max-frr of the known validation samples are rejected/flagged while as much
DoS as possible is caught. Either way they are saved to a JSON file that
test.py reads without changing anything:
  rejection_<metric>.json  (UNKNOWN mode)
  anomaly_<metric>.json    (ANOMALY mode)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from utilis.experiment import (
    ANOMALY_LABEL,
    BENIGN_CLASS,
    PSEUDO_UNKNOWN_CLASS,
    UNKNOWN_LABEL,
    compute_outputs,
    encode_known,
    load_experiment,
    load_split,
)
from utilis.plots import plot_anomaly_histograms, plot_outcome_table, plot_score_histograms, plot_tradeoff_curves
from utilis.rejection import (
    METHODS,
    METRICS,
    BenignReference,
    CentroidModel,
    accept_mask,
    anomaly_mask,
    anomaly_tradeoff_curve,
    apply_anomaly_rule,
    apply_rule,
    auroc,
    compute_anomaly_scores,
    compute_scores,
    format_report,
    open_set_report,
    outcome_table,
    search_anomaly_thresholds,
    search_thresholds,
    subset,
    tradeoff_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fit centroids and set rejection thresholds on validation data.")
    parser.add_argument("--experiment", type=Path, required=True, help="Experiment directory written by train.py --experiment-name.")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--metric", choices=METRICS, default="euclidean", help="Distance in embedding space (default: euclidean).")
    parser.add_argument("--anomaly", action="store_true", help="ANOMALY mode: use only the Benign centroid plus prediction entropy.")

    # UNKNOWN-mode thresholds. A given value is fixed; an omitted one is searched under --max-frr.
    parser.add_argument("--tau-c", type=float, default=None, help="UNKNOWN mode: minimum max-softmax confidence to accept (0-1).")
    parser.add_argument("--tau-d", type=float, default=None, help="UNKNOWN mode: maximum normalized distance to the predicted-class centroid.")

    # ANOMALY-mode thresholds. A given value is fixed; an omitted one is searched under --max-frr.
    parser.add_argument("--tau-h", type=float, default=None, help="ANOMALY mode: minimum normalized entropy to count as uncertain (0-1).")
    parser.add_argument("--tau-b", type=float, default=None, help="ANOMALY mode: minimum distance to the Benign centroid (Benign radii) to count as far.")

    parser.add_argument("--max-frr", type=float, default=0.05, help="Budget for the threshold search: max share of known validation samples rejected/flagged (default: 0.05).")
    parser.add_argument("--scale-quantile", type=float, default=0.95, help="Distance unit: this quantile of a class's training distances to its centroid (default: 0.95).")
    parser.add_argument("--require-agreement", action="store_true", help="UNKNOWN mode ablation: also reject when the nearest centroid is not the predicted class.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--precision", type=str.upper, choices=["FP32", "FP16", "BF16"], default="FP32")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows.")

    args = parser.parse_args()

    if args.anomaly and (args.tau_c is not None or args.tau_d is not None):
        parser.error("--tau-c/--tau-d belong to UNKNOWN mode; with --anomaly use --tau-h/--tau-b.")
    if not args.anomaly and (args.tau_h is not None or args.tau_b is not None):
        parser.error("--tau-h/--tau-b need --anomaly.")
    for name in ("tau_c", "tau_h"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    for name in ("tau_d", "tau_b"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0.0):
            parser.error(f"--{name.replace('_', '-')} must be a finite non-negative number")
    if not 0.0 < args.max_frr < 1.0:
        parser.error("--max-frr must be in (0, 1)")
    if not 0.5 <= args.scale_quantile < 1.0:
        parser.error("--scale-quantile must be in [0.5, 1)")

    return args


# ---------------------------------------------------------------------------
# UNKNOWN mode
# ---------------------------------------------------------------------------

def calibrate_unknown(args, experiment, train_embeddings, train_labels, val_logits, val_embeddings, y_val, masks, show):
    class_names = experiment["class_names"]
    known_mask, pseudo_mask = masks
    needs = {"confidence": ("tau_c",), "distance": ("tau_d",), "combined": ("tau_c", "tau_d")}

    centroids = CentroidModel(args.metric, quantile=args.scale_quantile).fit(train_embeddings, train_labels, len(class_names))
    scores = compute_scores(val_logits, val_embeddings, centroids)
    known, pseudo = subset(scores, known_mask), subset(scores, pseudo_mask)

    print(f"Distance unit: 1.0 = {args.scale_quantile:.0%} quantile of each class's training distances to its centroid")
    print("Class scales (raw distance that equals 1.0): " + ", ".join(f"{c} {s:.2f}" for c, s in zip(class_names, centroids.scales)))
    aurocs = auroc(known, pseudo)
    print(f"AUROC known vs DoS | confidence {aurocs['confidence']:.4f} | distance {aurocs['distance']:.4f}")

    closed_labels = np.asarray(class_names, dtype=object)[scores["pred"]]
    results, curves, chosen = {}, {}, {}

    for method in METHODS:
        curves[method] = tradeoff_curve(known, pseudo, method, args.require_agreement)

        # Every threshold this method uses was given -> evaluate as is.
        # Otherwise keep the given ones fixed and search the rest under the budget.
        if all(getattr(args, name) is not None for name in needs[method]):
            tau_c = args.tau_c if method in ("confidence", "combined") else 0.0
            tau_d = args.tau_d if method in ("distance", "combined") else np.inf
            selection = {
                "method": method,
                "feasible": True,
                "selection": "manual",
                "tau_c": float(tau_c),
                "tau_d": float(tau_d),
                "known_false_rejection": float(1.0 - accept_mask(known, method, tau_c, tau_d, args.require_agreement).mean()),
                "pseudo_unknown_rejection": float(1.0 - accept_mask(pseudo, method, tau_c, tau_d, args.require_agreement).mean()),
            }
        else:
            selection = search_thresholds(known, pseudo, method, args.max_frr, args.require_agreement, tau_c=args.tau_c, tau_d=args.tau_d)
            selection["selection"] = f"search (budget {args.max_frr:.0%})"
            if not selection["feasible"]:
                print(f"\n[{method}] no thresholds keep known rejection <= {args.max_frr:.0%}.")
                results[method] = selection
                continue

        final_labels = apply_rule(scores, class_names, method, selection["tau_c"], selection["tau_d"], args.require_agreement)
        selection["validation_report"] = open_set_report(y_val, final_labels, closed_labels, class_names)
        results[method] = selection
        chosen[method] = (selection["known_false_rejection"], selection["pseudo_unknown_rejection"])

        print(f"\n[{method}] tau_c = {selection['tau_c']:.4f} | tau_d = {selection['tau_d']:.4f} | {selection['selection']}")
        print(format_report(selection["validation_report"]))

    any_search = any(r.get("selection", "").startswith("search") for r in results.values())
    tag = f"Config {experiment['dataset']} | {experiment['loss_type']} | {args.metric}"
    plot_tradeoff_curves(
        curves, chosen,
        title=f"{tag}\nValidation trade-off (AUROC conf {aurocs['confidence']:.3f}, dist {aurocs['distance']:.3f})",
        save_path=args.experiment / f"calibration_curves_{args.metric}.png", show=show,
        max_frr=args.max_frr if any_search else None,
    )

    combined = results.get("combined", {})
    plot_score_histograms(
        known, pseudo, {"tau_c": combined.get("tau_c"), "tau_d": combined.get("tau_d")},
        title=f"{tag} | validation scores (combined thresholds)",
        save_path=args.experiment / f"calibration_scores_{args.metric}.png", show=show,
    )
    if combined.get("feasible"):
        final_labels = apply_rule(scores, class_names, "combined", combined["tau_c"], combined["tau_d"], args.require_agreement)
        plot_outcome_table(
            outcome_table(y_val, final_labels, class_names),
            title=f"{tag} | validation, combined rule",
            save_path=args.experiment / f"calibration_outcomes_{args.metric}.png", show=show,
        )

    payload = {
        "mode": "unknown",
        "reject_label": UNKNOWN_LABEL,
        "require_agreement": args.require_agreement,
        "auroc_known_vs_pseudo": aurocs,
        "centroids": centroids.to_dict(),
        "methods": results,
    }
    return payload, args.experiment / f"rejection_{args.metric}.json"


# ---------------------------------------------------------------------------
# ANOMALY mode
# ---------------------------------------------------------------------------

def calibrate_anomaly(args, experiment, train_embeddings, train_labels, val_logits, val_embeddings, y_val, masks, show):
    class_names = experiment["class_names"]
    known_mask, pseudo_mask = masks
    manual = args.tau_h is not None and args.tau_b is not None

    if BENIGN_CLASS not in class_names:
        raise ValueError(f"ANOMALY mode needs {BENIGN_CLASS} among the known classes.")
    benign_index = class_names.index(BENIGN_CLASS)

    benign = BenignReference(args.metric, quantile=args.scale_quantile).fit(train_embeddings[train_labels == benign_index])
    scores = compute_anomaly_scores(val_logits, val_embeddings, benign)
    known, pseudo = subset(scores, known_mask), subset(scores, pseudo_mask)

    print(f"Distance unit: 1.0 = {args.scale_quantile:.0%} quantile of Benign training distances to the Benign centroid (raw {benign.scale:.3f})")
    aurocs = auroc(known, pseudo)
    print(
        f"AUROC known vs DoS | confidence {aurocs['confidence']:.4f} | entropy {aurocs['entropy']:.4f} | "
        f"distance to Benign {aurocs['benign_distance']:.4f}"
    )

    y_known = y_val[known_mask]
    for name, group in (("Benign (val)", y_known == BENIGN_CLASS), ("known attacks (val)", y_known != BENIGN_CLASS)):
        d = known["dist_benign"][group]
        print(f"  distance to Benign, {name:<20} median {np.median(d):6.2f} | 5th pct {np.quantile(d, 0.05):6.2f}")
    print(f"  distance to Benign, {'DoS (val)':<20} median {np.median(pseudo['dist_benign']):6.2f} | 5th pct {np.quantile(pseudo['dist_benign'], 0.05):6.2f}")

    # Both given -> evaluate as is. Otherwise keep the given one fixed and search the other.
    if manual:
        selection = {
            "method": "anomaly",
            "feasible": True,
            "selection": "manual",
            "tau_h": float(args.tau_h),
            "tau_b": float(args.tau_b),
            "known_false_rejection": float(anomaly_mask(known, args.tau_h, args.tau_b).mean()),
            "pseudo_unknown_rejection": float(anomaly_mask(pseudo, args.tau_h, args.tau_b).mean()),
        }
    else:
        selection = search_anomaly_thresholds(known, pseudo, args.max_frr, tau_h=args.tau_h, tau_b=args.tau_b)
        selection["selection"] = f"search (budget {args.max_frr:.0%})"
        if not selection["feasible"]:
            raise ValueError(f"The fixed threshold already flags more than {args.max_frr:.0%} of the known samples.")

    closed_labels = np.asarray(class_names, dtype=object)[scores["pred"]]
    final_labels = apply_anomaly_rule(scores, class_names, selection["tau_h"], selection["tau_b"])
    selection["validation_report"] = open_set_report(y_val, final_labels, closed_labels, class_names, reject_label=ANOMALY_LABEL)

    print(f"\n[anomaly] tau_h = {selection['tau_h']:.4f} | tau_b = {selection['tau_b']:.4f} | {selection['selection']}")
    print(format_report(selection["validation_report"]))

    tag = f"Config {experiment['dataset']} | {experiment['loss_type']} | {args.metric} | ANOMALY mode"
    plot_tradeoff_curves(
        {"anomaly": anomaly_tradeoff_curve(known, pseudo, selection["tau_b"])},
        {"anomaly": (selection["known_false_rejection"], selection["pseudo_unknown_rejection"])},
        title=f"{tag}\nEntropy sweep at tau_b = {selection['tau_b']:.2f} (AUROC entropy {aurocs['entropy']:.3f})",
        save_path=args.experiment / f"anomaly_curve_{args.metric}.png", show=show,
        max_frr=None if manual else args.max_frr,
    )
    plot_anomaly_histograms(
        {
            "Benign": subset(known, y_known == BENIGN_CLASS),
            "known attacks": subset(known, y_known != BENIGN_CLASS),
            "DoS (pseudo-unknown)": pseudo,
        },
        {"tau_h": selection["tau_h"], "tau_b": selection["tau_b"]},
        title=f"{tag} | validation scores",
        save_path=args.experiment / f"anomaly_scores_{args.metric}.png", show=show,
    )
    plot_outcome_table(
        outcome_table(y_val, final_labels, class_names, reject_label=ANOMALY_LABEL),
        title=f"{tag} | validation",
        save_path=args.experiment / f"anomaly_outcomes_{args.metric}.png", show=show,
    )

    payload = {
        "mode": "anomaly",
        "reject_label": ANOMALY_LABEL,
        "auroc_known_vs_pseudo": aurocs,
        "benign_reference": benign.to_dict(),
        "methods": {"anomaly": selection},
    }
    return payload, args.experiment / f"anomaly_{args.metric}.json"


# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    experiment = load_experiment(args.experiment, args.checkpoint)
    class_names = experiment["class_names"]
    given = [f"--{n.replace('_', '-')} {getattr(args, n)}" for n in ("tau_c", "tau_d", "tau_h", "tau_b") if getattr(args, n) is not None]
    threshold_text = (f"fixed {', '.join(given)}; " if given else "") + f"others searched with budget {args.max_frr:.0%}"

    print(f"Experiment: {args.experiment} ({args.checkpoint}.pt, epoch {experiment['checkpoint_epoch']})")
    print(f"Dataset: {experiment['dataset']} | Features: {experiment['feature_set']} | Loss: {experiment['loss_type']}")
    print(f"Mode: {'ANOMALY' if args.anomaly else 'UNKNOWN'} | Metric: {args.metric} | Thresholds: {threshold_text}")

    # Reference centroids come from REAL training rows only (no SMOTE samples).
    X_train, y_train = load_split(experiment, "train", real_rows_only=True)
    train_labels = encode_known(y_train, class_names)
    if np.any(train_labels < 0):
        raise ValueError("Training split contains classes outside the known set.")
    _, train_embeddings = compute_outputs(experiment, X_train, args.batch_size, args.precision)

    # Validation: known classes + DoS pseudo-unknown.
    X_val, y_val = load_split(experiment, "validation")
    y_val = np.asarray(y_val, dtype=object)
    val_logits, val_embeddings = compute_outputs(experiment, X_val, args.batch_size, args.precision)

    known_mask = np.isin(y_val, class_names)
    pseudo_mask = y_val == PSEUDO_UNKNOWN_CLASS
    unexpected = sorted(set(y_val[~known_mask & ~pseudo_mask]))
    if unexpected:
        raise ValueError(f"Validation contains classes that are neither known nor DoS: {unexpected}")
    print(f"Centroids from {len(X_train)} real training rows | Validation: {known_mask.sum()} known, {pseudo_mask.sum()} DoS\n")

    calibrate = calibrate_anomaly if args.anomaly else calibrate_unknown
    payload, out_json = calibrate(
        args, experiment, train_embeddings, train_labels, val_logits, val_embeddings, y_val,
        (known_mask, pseudo_mask), show=not args.no_show,
    )

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
        "metric": args.metric,
        "scale_quantile": args.scale_quantile,
        "fixed_thresholds": {n: getattr(args, n) for n in ("tau_c", "tau_d", "tau_h", "tau_b") if getattr(args, n) is not None},
        "max_frr": args.max_frr,
        **payload,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        # tau_d = inf (confidence-only) or tau_h = inf (nothing flagged) is written as Infinity.
        json.dump(payload, f, indent=2)

    print(f"\nSaved {out_json.name} and figures to {args.experiment}")


if __name__ == "__main__":
    main()
