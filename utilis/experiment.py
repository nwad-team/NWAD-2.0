import json
from pathlib import Path

import joblib
import numpy as np
import torch

from data.data_loading import load_processed_split
from models.MLP import MLPClassifier

PSEUDO_UNKNOWN_CLASS = "DoS"
UNKNOWN_LABEL = "UNKNOWN"

DTYPES = {"FP32": torch.float32, "FP16": torch.float16, "BF16": torch.bfloat16}


def load_experiment(experiment_dir: Path, checkpoint: str = "best", device=None):

    experiment_dir = Path(experiment_dir)
    checkpoint_path = experiment_dir / f"{checkpoint}.pt"

    for path in (checkpoint_path, experiment_dir / "preprocessor.joblib", experiment_dir / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Train with --experiment-name first.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(experiment_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = MLPClassifier(**state["model_config"]).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    preprocessor = joblib.load(experiment_dir / "preprocessor.joblib")

    return {
        "model": model,
        "preprocessor": preprocessor,
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_epoch": int(state["epoch"]),
        "class_names": list(state["class_names"]),
        "dataset": state["dataset"],
        "feature_set": state["feature_set"],
        "loss_type": state.get("loss_type", "ce"),
        "device": device,
    }


def base_dataset(dataset: str) -> str:
    return dataset.split("-")[0]


def load_split(experiment: dict, split: str, real_rows_only: bool = False):
    dataset = experiment["dataset"]
    if real_rows_only and split == "train":
        dataset = base_dataset(dataset)

    return load_processed_split(dataset, split, experiment["feature_set"])


@torch.no_grad()
def compute_outputs(experiment: dict, X, batch_size: int = 2048, precision: str = "FP32", cast: str | None = None):
    model = experiment["model"]
    device = experiment["device"]

    features = np.asarray(experiment["preprocessor"].transform(X), dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinity after preprocessing.")

    features = torch.from_numpy(features)

    if cast is not None:
        model = model.to(dtype=DTYPES[cast])
        features = features.to(dtype=DTYPES[cast])

    use_amp = precision != "FP32" and device.type == "cuda"

    all_logits = []
    all_embeddings = []

    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size].to(device)

        with torch.autocast(device_type=device.type, dtype=DTYPES[precision], enabled=use_amp):
            logits, embeddings = model(batch, return_embedding=True)

        all_logits.append(logits.float().cpu())
        all_embeddings.append(embeddings.float().cpu())

    if cast is not None:
        experiment["model"] = model.to(dtype=torch.float32)

    return (
        torch.cat(all_logits).numpy().astype(np.float64),
        torch.cat(all_embeddings).numpy().astype(np.float64),
    )


def encode_known(labels, class_names):
    index = {name: i for i, name in enumerate(class_names)}
    return np.asarray([index.get(label, -1) for label in labels], dtype=np.int64)
