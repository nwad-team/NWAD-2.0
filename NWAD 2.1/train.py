import argparse
import numpy as np
import torch
import random
import pandas as pd
import time
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from data.data_loading import CONFIG_CLASSES, encode_labels, load_train_validation
from data.preprocessing import (build_preprocessor,encode_global_labels,)
from models.MLP import MLPClassifier
from utilis.loss import ClassificationLoss
from torch import optim
from tqdm import tqdm
from eval import evaluate
from utilis.metrics import macro_f1

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare settings for training")
    parser.add_argument("--batch-size", "-b", type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
    parser.add_argument("--num-workers", "-j", type=int, default=0)
    parser.add_argument("--precision",type=lambda value: value.upper(),choices=["FP32", "FP16", "BF16"],default="FP32",help="Requested training precision; AMP will be used in the training loop.",)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',help='SGD momentum (default: 0.5)')
    parser.add_argument("--output",type=Path,default=Path("checkpoints"),help="Directory for saved model checkpoints.",)
    parser.add_argument("--dataset",type=str,choices=["A", "A-Smote", "C", "C-Smote"],required=True,help="Dataset configuration to load.",)
    parser.add_argument("--feature-set",type=str,choices=["behavioral", "shortcut", "combined"],default="behavioral",help="Feature group to load (default: behavioral).",)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    return args

def main():
    args = parse_args()

    # Set random seeds.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1. Load training and validation splits.
    X_train, y_train, X_validation, y_validation = load_train_validation(
        dataset=args.dataset,
        feature_set=args.feature_set,
    )

    # Keep only known classes for classical validation, for now
    known_classes = CONFIG_CLASSES[args.dataset]

    known_validation_mask = y_validation.isin(known_classes)

    X_validation = X_validation.loc[known_validation_mask].copy()
    y_validation = y_validation.loc[known_validation_mask].copy()
    # ------------------------------------------------------------

    # 2. Build and apply the preprocessing pipeline.
    preprocessor = build_preprocessor(X_train)

    # Fit only on training data to prevent data leakage.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_validation)

    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_val_processed = np.asarray(X_val_processed, dtype=np.float32)

    if not np.isfinite(X_train_processed).all():
        raise ValueError("Training features contain NaN or infinity.")

    if not np.isfinite(X_val_processed).all():
        raise ValueError("Validation features contain NaN or infinity.")

    # 3. Get the known classes and encode target labels.
    # Encode training labels for the five-output model.
    y_train_encoded = encode_labels(y_train, args.dataset)
    y_val_encoded = encode_labels(y_validation, args.dataset)

    y_train_encoded = y_train_encoded.astype(np.int64, copy=False)
    y_val_encoded = y_val_encoded.astype(np.int64, copy=False)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_processed),
        torch.from_numpy(y_train_encoded.copy()),
    )

    validation_dataset = TensorDataset(
        torch.from_numpy(X_val_processed),
        torch.from_numpy(y_val_encoded.copy()),
    )

    # Prints for throubleshooting purposes
    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index().to_string())

    print("\nValidation class distribution:")
    print(y_validation.value_counts().sort_index().to_string())

    print(f"\nKnown classes: {CONFIG_CLASSES[args.dataset]}")

    print("\nTraining encoded label distribution:")
    print(
        pd.Series(y_train_encoded)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nValidation encoded label distribution:")
    print(
        pd.Series(y_val_encoded)
        .value_counts()
        .sort_index()
        .to_string()
    )
    # Prints for throubleshooting purposes

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

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )

    val_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    # 5. Initialize the model.
    # Number of input features after preprocessing.
    input_dim = X_train_processed.shape[1]

    # Number of known classes for this configuration.
    class_names = CONFIG_CLASSES[args.dataset]
    num_classes = len(class_names)

    model = MLPClassifier(input_dim=input_dim,num_classes=num_classes,).to(device)

    # Configure automatic mixed precision.
    use_amp = args.precision != "FP32" and device.type == "cuda"

    amp_dtype = {"FP32": torch.float32,"FP16": torch.float16,"BF16": torch.bfloat16,}[args.precision]

    scaler = torch.amp.GradScaler("cuda",enabled=use_amp and amp_dtype == torch.float16,)

    # Count model parameters.
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    # 6. Initialize the loss function.
    criterion = ClassificationLoss()

    # 7. Initialize the optimizer.
    optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,)

    # Alternative optimizer: Adam
    # optimizer = optim.Adam(model.parameters(),lr=args.lr,)

    # 8. Print run configuration.
    print(f"Dataset: {args.dataset}")
    print(f"Feature set: {args.feature_set}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Requested precision: {args.precision}")
    print(f"Device: {device}")
    print(f"DataLoader workers: {args.num_workers}; pin_memory={pin_memory}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")

    # 9. Training loop.
    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_samples_seen = 0
        train_targets = []
        train_predictions = []


        epoch_start = time.perf_counter()

        with tqdm(total=len(train_dataset),desc=f"Epoch {epoch}/{args.epochs}",unit="sample",dynamic_ncols=True,) as pbar:
            for features, targets in train_loader:

                features = features.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)

                optimizer.zero_grad(set_to_none=True)

                # Forward pass.
                with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp,):
                    logits = model(features)

                # Calculate loss in FP32 for numerical stability.
                loss = criterion(logits.float(), targets)

                # Backward pass and optimizer update.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = targets.size(0)   

                train_loss_sum += loss.item() * batch_size
                train_samples_seen += batch_size   

                predictions = logits.argmax(dim=1)
                train_targets.extend(targets.detach().cpu().tolist())
                train_predictions.extend(predictions.detach().cpu().tolist())

                elapsed = time.perf_counter() - epoch_start
                samples_per_second = train_samples_seen / max(elapsed, 1e-9)

                pbar.update(batch_size)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    samples_s=f"{samples_per_second:.2f}",
                )

        avg_train_loss = train_loss_sum / train_samples_seen

        train_macro_f1 = macro_f1(
            y_true=train_targets,
            y_pred=train_predictions,
            num_classes=num_classes,
        )

        # 10. Validation is handled by eval.py.
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=len(CONFIG_CLASSES[args.dataset]),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            desc=f"Validation {epoch}/{args.epochs}",
        )

        avg_val_loss = val_metrics["loss"]
        val_macro_f1 = val_metrics["macro_f1"]

        print(
            f"Epoch {epoch}/{args.epochs} complete | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Macro F1: {train_macro_f1:.4f} | "
            f"Validation Loss: {avg_val_loss:.4f} | "
            f"Validation Macro F1: {val_macro_f1:.4f}\n"
        )

if __name__ == "__main__":
    main()
