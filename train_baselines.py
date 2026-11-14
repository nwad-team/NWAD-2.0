import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, f1_score

from data.data_loading import CONFIG_CLASSES, encode_labels, load_train_validation
from data.preprocessing import build_preprocessor
from utilis.plots import plot_confusion_matrix


DATASETS = ["A", "A-Smote", "C", "C-Smote"]

MODEL_NAMES = {
    "random-forest": "Random Forest",
    "logistic-regression": "Logistic Regression",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train classical baselines on known classes.")
    parser.add_argument("--dataset",choices=[*DATASETS, "all"],required=True,help="Dataset configuration to load, or 'all' for the four configurations.",)
    parser.add_argument("--feature-set",choices=["behavioral", "shortcut", "combined"],default="behavioral",help="Feature group to load (default: behavioral).",)
    parser.add_argument("--model",choices=["both", *MODEL_NAMES],default="both",help="Baseline to train (default: both).",)
    parser.add_argument("--n-estimators", type=int, default=250, help="Random Forest: number of trees (default: 250).")
    parser.add_argument("--max-iter", type=int, default=250, help="Logistic Regression: maximum solver iterations (default: 250).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output",type=Path,default=Path("results") / "baselines",help="Root directory for reports and model artifacts.",)
    parser.add_argument("--no-show",action="store_true",help="Do not open confusion-matrix windows.",)
    args = parser.parse_args()
    if args.n_estimators < 1 or args.max_iter < 1:
        parser.error("--n-estimators and --max-iter must be positive")
    return args


def build_models(args):
    return {
        "random-forest": RandomForestClassifier(n_estimators=args.n_estimators,n_jobs=-1,random_state=args.seed,),
        "logistic-regression": LogisticRegression(max_iter=args.max_iter,random_state=args.seed,),
    }


def main():
    args = parse_args()
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    all_rows = []
    for dataset in datasets:
        all_rows.extend(run_dataset(args, dataset))

    if len(datasets) > 1:
        summary = pd.DataFrame(all_rows)
        summary_path = args.output / f"summary_all_{args.feature_set}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)

        print(f"\n{'=' * 72}\nValidation Macro-F1 (known classes)")
        print(summary.pivot(index="dataset", columns="model", values="validation_macro_f1").reindex(datasets).round(4).to_string())
        print(f"\nSaved {summary_path}")


def run_dataset(args, dataset: str) -> list[dict]:
    """Train the requested baselines on one dataset configuration."""
    class_names = CONFIG_CLASSES[dataset]
    num_classes = len(class_names)

    X_train, y_train, X_validation, y_validation = load_train_validation(
        dataset=dataset,
        feature_set=args.feature_set,
    )

    train_mask = y_train.isin(class_names)
    validation_mask = y_validation.isin(class_names)
    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    X_validation = X_validation.loc[validation_mask].copy()
    y_validation = y_validation.loc[validation_mask].copy()

    if X_train.empty:
        raise ValueError(f"No known-class training samples found for {dataset}.")
    if X_validation.empty:
        raise ValueError(f"No known-class validation samples found for {dataset}.")

    preprocessor = build_preprocessor(X_train)
    X_train_processed = np.asarray(
        preprocessor.fit_transform(X_train),
        dtype=np.float32,
    )
    X_val_processed = np.asarray(
        preprocessor.transform(X_validation),
        dtype=np.float32,
    )

    if not np.isfinite(X_train_processed).all():
        raise ValueError("Training features contain NaN or infinity.")
    if not np.isfinite(X_val_processed).all():
        raise ValueError("Validation features contain NaN or infinity.")

    y_train_encoded = encode_labels(y_train, dataset)
    y_val_encoded = encode_labels(y_validation, dataset)

    model_keys = list(MODEL_NAMES) if args.model == "both" else [args.model]
    output_dir = args.output / dataset / args.feature_set
    summary_rows = []

    print(f"\n{'#' * 72}\nDataset: {dataset} | Feature set: {args.feature_set}")
    print(f"Known classes (index order): {class_names}")
    print(
        f"Training samples: {len(y_train_encoded)} | "
        f"Validation samples (known classes): {len(y_val_encoded)}"
    )
    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index().to_string())
    print("\nValidation class distribution (known classes only):")
    print(y_validation.value_counts().sort_index().to_string())

    for model_key in model_keys:
        model = build_models(args)[model_key]
        model_output_dir = output_dir / model_key
        model_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 72}\n{MODEL_NAMES[model_key]}")
        start_time = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train_processed, y_train_encoded)
        fit_seconds = time.perf_counter() - start_time

        # Iterations actually used: trees for RF, solver iterations for LR.
        if model_key == "random-forest":
            budget = {"n_estimators": args.n_estimators}
            converged = None
        else:
            budget = {"max_iter": args.max_iter, "n_iter": int(np.max(model.n_iter_))}
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

        train_predictions = model.predict(X_train_processed)
        val_predictions = model.predict(X_val_processed)
        train_f1 = f1_score(
            y_train_encoded,
            train_predictions,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )
        val_f1 = f1_score(
            y_val_encoded,
            val_predictions,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )

        print(f"Fit time: {fit_seconds:.1f}s | {budget}" + ("" if converged is None else f" | converged: {converged}"))
        print(f"Training Macro-F1: {train_f1:.4f}")
        print(f"Validation Macro-F1: {val_f1:.4f}")
        print("\nPer-class report (validation, known classes):")
        print(
            classification_report(
                y_val_encoded,
                val_predictions,
                labels=list(range(num_classes)),
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
        )

        title = (
            f"{MODEL_NAMES[model_key]} | {dataset} | {args.feature_set}\n"
            f"Validation, known classes (Macro-F1 {val_f1:.4f})"
        )
        counts = plot_confusion_matrix(
            y_true=y_val_encoded,
            y_pred=val_predictions,
            class_names=class_names,
            title=title,
            save_path=model_output_dir / "confusion_matrix_val.png",
            show=not args.no_show,
        )
        print("Confusion matrix (rows = true, columns = predicted):")
        print(pd.DataFrame(counts, index=class_names, columns=class_names).to_string())

        joblib.dump(model, model_output_dir / "model.joblib")
        joblib.dump(preprocessor, model_output_dir / "preprocessor.joblib")
        with open(model_output_dir / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "dataset": dataset,
                    "feature_set": args.feature_set,
                    "model": model_key,
                    **budget,
                    "converged": converged,
                    "seed": args.seed,
                    "known_classes": class_names,
                    "train_samples": len(y_train_encoded),
                    "validation_samples": len(y_val_encoded),
                    "train_macro_f1": float(train_f1),
                    "validation_macro_f1": float(val_f1),
                    "fit_seconds": fit_seconds,
                },
                file,
                indent=2,
            )

        summary_rows.append(
            {
                "model": model_key,
                "dataset": dataset,
                "feature_set": args.feature_set,
                "seed": args.seed,
                "train_samples": len(y_train_encoded),
                "validation_samples": len(y_val_encoded),
                "train_macro_f1": float(train_f1),
                "validation_macro_f1": float(val_f1),
                "fit_seconds": fit_seconds,
                **budget,
                "converged": converged,
            }
        )
        print(f"\nSaved model, preprocessor, metrics and confusion matrix to {model_output_dir}")

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    print(f"\nSaved baseline comparison to {output_dir / 'summary.csv'}")
    return summary_rows


if __name__ == "__main__":
    main()
