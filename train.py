import argparse
import copy
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data.data_loading import CONFIG_CLASSES, encode_labels, load_train_validation
from data.preprocessing import build_preprocessor
from eval import evaluate
from models.MLP import MLPClassifier
from utilis.loss import JointLoss
from utilis.metrics import macro_f1
from utilis.plots import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train the known-class MLP classifier.")
    parser.add_argument("--dataset", type=str, choices=["A", "A-Smote", "C", "C-Smote"], required=True, help="Dataset configuration to load.")
    parser.add_argument("--feature-set", type=str, choices=["behavioral", "shortcut", "combined"], default="behavioral", help="Feature group to load (default: behavioral).")
    parser.add_argument("--batch-size", "-b", type=int, default=256, help="Mini-batch size (default: 256).")
    parser.add_argument("--epochs", type=int, default=100, metavar="N", help="Maximum number of epochs (default: 100). Early stopping usually ends training sooner.")
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR", help="AdamW learning rate (default: 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW decoupled weight decay (default: 1e-4).")
    parser.add_argument("--patience", type=int, default=10, help="Epochs without validation Macro-F1 improvement before stopping (default: 10).")
    parser.add_argument("--lr-patience", type=int, default=4, help="Epochs without improvement before halving the learning rate (default: 4).")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum Macro-F1 gain that counts as an improvement (default: 1e-4).")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128], help="Hidden layer widths (default: 256 128).")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding (penultimate layer) dimension (default: 32).")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout after each hidden layer (default: 0.3).")
    parser.add_argument("--loss", type=str, choices=["ce", "center"], default="ce", help="Training objective: plain cross-entropy or CE + lambda * center loss (default: ce).")
    parser.add_argument("--center-lambda", type=float, default=0.01, help="Weight of the center-loss term (default: 0.01). Only used with --loss center.")
    parser.add_argument("--center-lr", type=float, default=0.5, help="SGD learning rate for the class centers (default: 0.5, as in Wen et al. 2016).")
    parser.add_argument("--num-workers", "-j", type=int, default=0, help="DataLoader workers (default: 0; the data is already an in-memory tensor).")
    parser.add_argument("--precision", type=lambda value: value.upper(), choices=["FP32", "FP16", "BF16"], default="FP32", help="Requested training precision; AMP is used for FP16/BF16 on CUDA.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"), help="Root directory for experiment outputs (default: checkpoints).")
    parser.add_argument("--experiment-name", type=str, default=None, help="If given, save best.pt, last.pt, the fitted preprocessor, history and plots under <output>/<experiment-name>/.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the confusion-matrix window at the end of training.")

    args = parser.parse_args()

    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 (BatchNorm needs more than one sample).")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.patience < 1:
        parser.error("--patience must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if args.center_lambda <= 0:
        parser.error("--center-lambda must be positive")

    return args


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model, criterion, optimizer, scheduler, epoch, val_metrics, best_val_macro_f1, args, class_names):
    """Save a checkpoint that loads with torch.load(..., weights_only=True)."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "val_loss": float(val_metrics["loss"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "best_val_macro_f1": float(best_val_macro_f1),
            "model_config": model.config,
            "loss_type": args.loss,
            "center_lambda": float(args.center_lambda) if args.loss == "center" else None,
            "criterion_state_dict": criterion.state_dict(),
            "class_names": list(class_names),
            "dataset": args.dataset,
            "feature_set": args.feature_set,
            "seed": args.seed,
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    experiment_dir = None
    if args.experiment_name:
        experiment_dir = args.output / args.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load training and validation splits.
    X_train, y_train, X_validation, y_validation = load_train_validation(
        dataset=args.dataset,
        feature_set=args.feature_set,
    )

    # Classical validation uses known classes only. The DoS pseudo-unknown
    # rows are kept out of model selection and reserved for threshold calibration.
    class_names = CONFIG_CLASSES[args.dataset]
    num_classes = len(class_names)

    known_validation_mask = y_validation.isin(class_names)
    X_validation = X_validation.loc[known_validation_mask].copy()
    y_validation = y_validation.loc[known_validation_mask].copy()

    # 2. Build and apply the preprocessing pipeline (fit on training data only).
    preprocessor = build_preprocessor(X_train)

    X_train_processed = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
    X_val_processed = np.asarray(preprocessor.transform(X_validation), dtype=np.float32)

    if not np.isfinite(X_train_processed).all():
        raise ValueError("Training features contain NaN or infinity.")
    if not np.isfinite(X_val_processed).all():
        raise ValueError("Validation features contain NaN or infinity.")

    # 3. Encode target labels for the known-class model.
    y_train_encoded = encode_labels(y_train, args.dataset).astype(np.int64, copy=False)
    y_val_encoded = encode_labels(y_validation, args.dataset).astype(np.int64, copy=False)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_processed),
        torch.from_numpy(y_train_encoded.copy()),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(X_val_processed),
        torch.from_numpy(y_val_encoded.copy()),
    )

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index().to_string())
    print("\nValidation class distribution (known classes only):")
    print(y_validation.value_counts().sort_index().to_string())
    print(f"\nKnown classes (index order): {class_names}")

    # 4. Configure DataLoaders.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_options["persistent_workers"] = True

    # drop_last avoids a final batch of size 1, which BatchNorm cannot train on.
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    val_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    # 5. Initialize the model.
    input_dim = X_train_processed.shape[1]
    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=tuple(args.hidden_dims),
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)

    use_amp = args.precision != "FP32" and device.type == "cuda"
    amp_dtype = {"FP32": torch.float32, "FP16": torch.float16, "BF16": torch.bfloat16}[args.precision]
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    total_params = sum(p.numel() for p in model.parameters())

    # 6. Loss, optimizer and learning-rate schedule.
    criterion = JointLoss(
        loss_type=args.loss,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        center_lambda=args.center_lambda,
    ).to(device)
    # Validation loss is always plain cross-entropy so runs are comparable.
    val_criterion = criterion.ce

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Class centers get their own plain SGD optimizer (Wen et al. 2016).
    center_optimizer = None
    if criterion.center is not None:
        center_optimizer = optim.SGD(criterion.center.parameters(), lr=args.center_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
    )

    # 7. Print run configuration.
    print(f"\nDataset: {args.dataset}")
    print(f"Feature set: {args.feature_set}")
    print(f"Input features after preprocessing: {input_dim}")
    print(f"Model: hidden={args.hidden_dims}, embedding={args.embedding_dim}, dropout={args.dropout}, params={total_params:,}")
    loss_text = "cross-entropy" if args.loss == "ce" else f"cross-entropy + {args.center_lambda} * center loss (center lr {args.center_lr})"
    print(f"Loss: {loss_text}")
    print(f"Optimizer: AdamW(lr={args.lr}, weight_decay={args.weight_decay}), ReduceLROnPlateau(factor=0.5, patience={args.lr_patience})")
    print(f"Early stopping: patience={args.patience} on validation Macro-F1")
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(validation_dataset)}")
    print(f"Batch size: {args.batch_size} | Max epochs: {args.epochs}")
    print(f"Device: {device} | Precision: {args.precision} | Workers: {args.num_workers}")
    print(f"Checkpoints: {experiment_dir if experiment_dir else 'disabled (no --experiment-name)'}\n")

    if experiment_dir is not None:
        joblib.dump(preprocessor, experiment_dir / "preprocessor.joblib")
        with open(experiment_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    "class_names": class_names,
                    "input_dim": input_dim,
                    "model_config": model.config,
                    "total_params": total_params,
                },
                f,
                indent=2,
            )

    # 8. Training loop with early stopping on validation Macro-F1.
    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_ce_sum = 0.0
        train_center_sum = 0.0
        train_samples_seen = 0
        train_targets = []
        train_predictions = []
        epoch_start = time.perf_counter()

        with tqdm(total=len(train_loader) * args.batch_size, desc=f"Epoch {epoch}/{args.epochs}", unit="sample", dynamic_ncols=True, leave=False) as pbar:
            for features, targets in train_loader:
                features = features.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)

                optimizer.zero_grad(set_to_none=True)
                if center_optimizer is not None:
                    center_optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits, embeddings = model(features, return_embedding=True)

                # Loss in FP32 for numerical stability.
                loss, ce_term, center_term = criterion(logits.float(), embeddings.float(), targets)

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)

                if center_optimizer is not None:
                    # Undo lambda on the center gradients so the center update
                    # speed is set by --center-lr alone (Wen et al. 2016).
                    grad_scaler.unscale_(center_optimizer)
                    for parameter in criterion.center.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(1.0 / args.center_lambda)
                    grad_scaler.step(center_optimizer)

                grad_scaler.update()

                batch_size = targets.size(0)
                train_loss_sum += loss.item() * batch_size
                train_ce_sum += ce_term.item() * batch_size
                train_center_sum += center_term.item() * batch_size
                train_samples_seen += batch_size

                train_targets.append(targets.detach().cpu())
                train_predictions.append(logits.argmax(dim=1).detach().cpu())

                pbar.update(batch_size)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_sum / train_samples_seen
        avg_train_ce = train_ce_sum / train_samples_seen
        avg_train_center = train_center_sum / train_samples_seen
        train_macro_f1 = macro_f1(
            y_true=torch.cat(train_targets).numpy(),
            y_pred=torch.cat(train_predictions).numpy(),
            num_classes=num_classes,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=val_criterion,
            device=device,
            num_classes=num_classes,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            desc=f"Validation {epoch}/{args.epochs}",
        )
        val_macro_f1 = val_metrics["macro_f1"]
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_macro_f1)

        improved = val_macro_f1 > best_val_macro_f1 + args.min_delta
        if improved:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_time = time.perf_counter() - epoch_start
        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": avg_train_loss,
            "train_ce": avg_train_ce,
            "train_center": avg_train_center,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_macro_f1,
            "epoch_seconds": epoch_time,
        })

        print(
            f"Epoch {epoch:3d} | lr {current_lr:.2e} | "
            f"train loss {avg_train_loss:.4f}"
            + (f" (ce {avg_train_ce:.4f}, center {avg_train_center:.3f})" if args.loss == "center" else "")
            + f" F1 {train_macro_f1:.4f} | "
            f"val loss {val_metrics['loss']:.4f} F1 {val_macro_f1:.4f} | "
            f"{epoch_time:.1f}s" + ("  *best*" if improved else "")
        )

        if experiment_dir is not None:
            save_checkpoint(experiment_dir / "last.pt", model, criterion, optimizer, scheduler, epoch, val_metrics, best_val_macro_f1, args, class_names)
            if improved:
                save_checkpoint(experiment_dir / "best.pt", model, criterion, optimizer, scheduler, epoch, val_metrics, best_val_macro_f1, args, class_names)
            pd.DataFrame(history).to_csv(experiment_dir / "history.csv", index=False)

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    # 9. Final report on the best model (known-class validation).
    model.load_state_dict(best_state)

    final_metrics = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=val_criterion,
        device=device,
        num_classes=num_classes,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        desc="Best-model validation",
    )

    print(f"\nBest epoch: {best_epoch} | Validation Macro-F1: {final_metrics['macro_f1']:.4f}")
    print("\nPer-class report (validation, known classes):")
    print(classification_report(
        final_metrics["targets"],
        final_metrics["predictions"],
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))

    config_label = args.dataset.split("-")[0]
    smote_label = " + SMOTE" if "Smote" in args.dataset else ""
    title = (
        f"Config {config_label}{smote_label} | {args.feature_set} features | loss: {args.loss}\n"
        f"Validation, known classes (best epoch {best_epoch}, Macro-F1 {final_metrics['macro_f1']:.4f})"
    )

    counts = plot_confusion_matrix(
        y_true=final_metrics["targets"],
        y_pred=final_metrics["predictions"],
        class_names=class_names,
        title=title,
        save_path=(experiment_dir / "confusion_matrix_val.png") if experiment_dir else None,
        show=not args.no_show,
    )

    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(counts, index=class_names, columns=class_names).to_string())

    if experiment_dir is not None:
        print(f"\nSaved best.pt, last.pt, preprocessor.joblib, config.json, history.csv and confusion_matrix_val.png to {experiment_dir}")


if __name__ == "__main__":
    main()
