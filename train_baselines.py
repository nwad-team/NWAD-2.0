import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from data.data_loading import CONFIG_CLASSES, encode_labels, load_train_validation
from data.preprocessing import build_preprocessor
from utilis.plots import plot_confusion_matrix


MODEL_NAMES = {
    "random-forest": "Random Forest",
    "logistic-regression": "Logistic Regression",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train classical baselines on known classes.")
    parser.add_argument("--dataset",choices=["A", "A-Smote", "C", "C-Smote"],required=True,help="Dataset configuration to load.",)
    parser.add_argument("--feature-set",choices=["behavioral", "shortcut", "combined"],default="behavioral",help="Feature group to load (default: behavioral).",)
    parser.add_argument("--model",choices=["both", *MODEL_NAMES],default="both",help="Baseline to train (default: both).",)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output",type=Path,default=Path("results") / "baselines",help="Root directory for reports and model artifacts.",)
    parser.add_argument("--no-show",action="store_true",help="Do not open confusion-matrix windows.",)
    return parser.parse_args()


def build_models(seed: int):
    return {
        "random-forest": RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=seed,),
        "logistic-regression": LogisticRegression(max_iter=1000,random_state=seed,),
    }


def main():
    args = parse_args()
    class_names = CONFIG_CLASSES[args.dataset]
    num_classes = len(class_names)

    X_train, y_train, X_validation, y_validation = load_train_validation(
        dataset=args.dataset,
        feature_set=args.feature_set,
    )

    train_mask = y_train.isin(class_names)
    validation_mask = y_validation.isin(class_names)
    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    X_validation = X_validation.loc[validation_mask].copy()
    y_validation = y_validation.loc[validation_mask].copy()

    if X_train.empty:
        raise ValueError(f"No known-class training samples found for {args.dataset}.")
    if X_validation.empty:
        raise ValueError(f"No known-class validation samples found for {args.dataset}.")

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

    y_train_encoded = encode_labels(y_train, args.dataset)
    y_val_encoded = encode_labels(y_validation, args.dataset)

    model_keys = list(MODEL_NAMES) if args.model == "both" else [args.model]
    output_dir = args.output / args.dataset / args.feature_set
    summary_rows = []

    print(f"\nDataset: {args.dataset} | Feature set: {args.feature_set}")
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
        model = build_models(args.seed)[model_key]
        model_output_dir = output_dir / model_key
        model_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 72}\n{MODEL_NAMES[model_key]}")
        start_time = time.perf_counter()
        model.fit(X_train_processed, y_train_encoded)
        fit_seconds = time.perf_counter() - start_time

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

        print(f"Fit time: {fit_seconds:.1f}s")
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
            f"{MODEL_NAMES[model_key]} | {args.dataset} | {args.feature_set}\n"
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
                    "dataset": args.dataset,
                    "feature_set": args.feature_set,
                    "model": model_key,
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
                "dataset": args.dataset,
                "feature_set": args.feature_set,
                "seed": args.seed,
                "train_samples": len(y_train_encoded),
                "validation_samples": len(y_val_encoded),
                "train_macro_f1": float(train_f1),
                "validation_macro_f1": float(val_f1),
                "fit_seconds": fit_seconds,
            }
        )
        print(f"\nSaved model, preprocessor, metrics and confusion matrix to {model_output_dir}")

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    print(f"\nSaved baseline comparison to {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
