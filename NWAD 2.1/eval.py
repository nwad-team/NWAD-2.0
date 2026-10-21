
import time

import torch
from tqdm import tqdm

from utilis.metrics import macro_f1


@torch.no_grad()
def evaluate(model,dataloader,criterion,device,num_classes,use_amp=False,amp_dtype=torch.float32,desc="Validation",):
    """Evaluate a multiclass classifier and return average loss and Macro-F1."""

    model.eval()

    total_loss = 0.0
    samples_seen = 0
    all_targets = []
    all_predictions = []

    start_time = time.perf_counter()

    with tqdm(total=len(dataloader.dataset),desc=desc,unit="sample",dynamic_ncols=True,) as pbar:
        for features, targets in dataloader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass with optional automatic mixed precision.
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp,):
                logits = model(features)

            # Calculate loss in FP32 for numerical stability.
            loss = criterion(logits.float(), targets)

            predictions = logits.argmax(dim=1)

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size
            samples_seen += batch_size

            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

            elapsed = time.perf_counter() - start_time
            samples_per_second = samples_seen / max(elapsed, 1e-9)

            pbar.update(batch_size)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                samples_s=f"{samples_per_second:.2f}",
            )

    if samples_seen == 0:
        raise ValueError("Cannot evaluate an empty DataLoader.")

    avg_loss = total_loss / samples_seen

    score = macro_f1(y_true=all_targets,y_pred=all_predictions,num_classes=num_classes,)

    elapsed = time.perf_counter() - start_time
    throughput = samples_seen / max(elapsed, 1e-9)

    return {
        "loss": avg_loss,
        "macro_f1": score,
        "samples_per_second": throughput,
        "elapsed_seconds": elapsed,
    }
