import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from utilis.experiment import PSEUDO_UNKNOWN_CLASS, UNKNOWN_LABEL

METHODS = ("confidence", "distance", "combined")
METRICS = ("euclidean", "mahalanobis")

class CentroidModel:

    def __init__(self, metric: str = "euclidean", quantile: float = 0.95, ridge: float = 1e-6):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        self.metric = metric
        self.quantile = quantile
        self.ridge = ridge
        self.means = None
        self.precision_matrix = None
        self.scales = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray, num_classes: int):
        missing = [k for k in range(num_classes) if not np.any(labels == k)]
        if missing:
            raise ValueError(f"No training embeddings for class indices {missing}.")

        self.means = np.stack([embeddings[labels == k].mean(axis=0) for k in range(num_classes)])

        if self.metric == "mahalanobis":
            centered = embeddings - self.means[labels]
            covariance = centered.T @ centered / max(len(embeddings) - num_classes, 1)
            dim = covariance.shape[0]
            covariance += self.ridge * np.trace(covariance) / dim * np.eye(dim)
            self.precision_matrix = np.linalg.inv(covariance)

        raw = self.raw_distances(embeddings)
        own = raw[np.arange(len(labels)), labels]
        self.scales = np.array([np.quantile(own[labels == k], self.quantile) for k in range(num_classes)])
        return self

    def raw_distances(self, embeddings: np.ndarray) -> np.ndarray:
        diff = embeddings[:, None, :] - self.means[None, :, :]

        if self.metric == "euclidean":
            squared = np.einsum("ncd,ncd->nc", diff, diff)
        else:
            squared = np.einsum("ncd,de,nce->nc", diff, self.precision_matrix, diff)

        return np.sqrt(np.maximum(squared, 0.0))

    def distances(self, embeddings: np.ndarray) -> np.ndarray:
        return self.raw_distances(embeddings) / self.scales[None, :]

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "quantile": self.quantile,
            "ridge": self.ridge,
            "means": self.means.tolist(),
            "precision_matrix": None if self.precision_matrix is None else self.precision_matrix.tolist(),
            "scales": self.scales.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CentroidModel":
        model = cls(data["metric"], data["quantile"], data["ridge"])
        model.means = np.asarray(data["means"], dtype=np.float64)
        model.scales = np.asarray(data["scales"], dtype=np.float64)
        if data["precision_matrix"] is not None:
            model.precision_matrix = np.asarray(data["precision_matrix"], dtype=np.float64)
        return model


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_scores(logits: np.ndarray, embeddings: np.ndarray, centroids: CentroidModel) -> dict:
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    distances = centroids.distances(embeddings)
    nearest = distances.argmin(axis=1)

    return {
        "pred": pred,
        "conf": probs.max(axis=1),
        "dist_pred": distances[np.arange(len(pred)), pred],
        "nearest": nearest,
        "agree": nearest == pred,
    }


def subset(scores: dict, mask: np.ndarray) -> dict:
    return {key: value[mask] for key, value in scores.items()}


def accept_mask(scores: dict, method: str, tau_c: float, tau_d: float, require_agreement: bool = False) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    accepted = np.ones(len(scores["pred"]), dtype=bool)

    if method in ("confidence", "combined"):
        accepted &= scores["conf"] >= tau_c

    if method in ("distance", "combined"):
        accepted &= scores["dist_pred"] <= tau_d
        if require_agreement:
            accepted &= scores["agree"]

    return accepted


def apply_rule(scores: dict, class_names: list[str], method: str, tau_c: float, tau_d: float, require_agreement: bool = False) -> np.ndarray:
    accepted = accept_mask(scores, method, tau_c, tau_d, require_agreement)
    names = np.asarray(class_names, dtype=object)[scores["pred"]]
    return np.where(accepted, names, UNKNOWN_LABEL)


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def search_thresholds(known: dict, unknown: dict, method: str, max_frr: float, require_agreement: bool = False, n_grid: int = 51) -> dict:

    q = np.linspace(0.0, max_frr, n_grid)[1:]

    if method in ("confidence", "combined"):
        tau_c_grid = np.concatenate([[0.0], np.quantile(known["conf"], q)])
    else:
        tau_c_grid = np.array([0.0])

    if method in ("distance", "combined"):
        tau_d_grid = np.concatenate([[np.inf], np.quantile(known["dist_pred"], 1.0 - q)])
    else:
        tau_d_grid = np.array([np.inf])

    best = None
    for tau_c in np.unique(tau_c_grid):
        for tau_d in np.unique(tau_d_grid):
            frr = 1.0 - accept_mask(known, method, tau_c, tau_d, require_agreement).mean()
            if frr > max_frr + 1e-12:
                continue
            rejection = 1.0 - accept_mask(unknown, method, tau_c, tau_d, require_agreement).mean()
            key = (rejection, -frr)
            if best is None or key > best[0]:
                best = (key, float(tau_c), float(tau_d), float(frr), float(rejection))

    if best is None:
        return {"method": method, "feasible": False}

    _, tau_c, tau_d, frr, rejection = best
    return {
        "method": method,
        "feasible": True,
        "tau_c": tau_c,
        "tau_d": tau_d,  # may be inf for the confidence-only method
        "known_false_rejection": frr,
        "pseudo_unknown_rejection": rejection,
    }


def tradeoff_curve(known: dict, unknown: dict, method: str, require_agreement: bool = False, n_points: int = 101):

    q = np.linspace(0.0, 1.0, n_points)

    if method == "confidence":
        pairs = [(t, np.inf) for t in np.quantile(known["conf"], q)]
    elif method == "distance":
        pairs = [(0.0, t) for t in np.quantile(known["dist_pred"], 1.0 - q)]
    else:
        qq = np.linspace(0.0, 0.5, 41)
        pairs = [
            (c, d)
            for c in np.quantile(known["conf"], qq)
            for d in np.quantile(known["dist_pred"], 1.0 - qq)
        ]

    points = []
    for tau_c, tau_d in pairs:
        frr = 1.0 - accept_mask(known, method, tau_c, tau_d, require_agreement).mean()
        rej = 1.0 - accept_mask(unknown, method, tau_c, tau_d, require_agreement).mean()
        points.append((frr, rej))

    points = np.array(sorted(points))

    if method == "combined":
        # Keep the best rejection reached at or below each false-rejection level.
        points[:, 1] = np.maximum.accumulate(points[:, 1])

    return points


def auroc(known: dict, unknown: dict) -> dict:
    y = np.concatenate([np.zeros(len(known["conf"])), np.ones(len(unknown["conf"]))])
    return {
        "confidence": float(roc_auc_score(y, -np.concatenate([known["conf"], unknown["conf"]]))),
        "distance": float(roc_auc_score(y, np.concatenate([known["dist_pred"], unknown["dist_pred"]]))),
    }



def open_set_report(true_labels, final_labels, closed_labels, class_names: list[str]) -> dict:

    true_labels = np.asarray(true_labels, dtype=object)
    final_labels = np.asarray(final_labels, dtype=object)
    closed_labels = np.asarray(closed_labels, dtype=object)

    known = np.isin(true_labels, class_names)
    pseudo = true_labels == PSEUDO_UNKNOWN_CLASS
    real_unknown = ~known & ~pseudo
    rejected = final_labels == UNKNOWN_LABEL

    report = {"n_samples": int(len(true_labels))}

    # Known-class classification.
    report["known"] = {
        "n": int(known.sum()),
        "closed_set_macro_f1": float(f1_score(true_labels[known], closed_labels[known], labels=class_names, average="macro", zero_division=0)),
        # UNKNOWN on a known sample counts as a miss for its class.
        "open_set_macro_f1": float(f1_score(true_labels[known], final_labels[known], labels=class_names, average="macro", zero_division=0)),
        "false_rejection_rate": float(rejected[known].mean()),
        "accuracy_on_accepted": float((final_labels == true_labels)[known & ~rejected].mean()) if np.any(known & ~rejected) else float("nan"),
    }

    # Unknown rejection, with DoS (used for calibration) kept apart.
    for name, mask in (("pseudo_unknown_DoS", pseudo), ("real_unknown", real_unknown)):
        if mask.any():
            report[name] = {
                "n": int(mask.sum()),
                "rejection_rate": float(rejected[mask].mean()),
                "false_acceptance_rate": float((~rejected[mask]).mean()),
                "accepted_as_benign_rate": float((final_labels[mask] == "Benign").mean()),
            }

    # Benign versus attack. UNKNOWN is a triage signal, not an attack verdict,
    # so it is counted separately rather than as a positive detection.
    benign = true_labels == "Benign"
    attack = ~benign
    predicted_attack_class = ~rejected & (final_labels != "Benign")

    if benign.any():
        report["benign"] = {
            "n": int(benign.sum()),
            "false_positive_rate": float(predicted_attack_class[benign].mean()),
            "triage_rate": float(rejected[benign].mean()),
        }
    if attack.any():
        report["attack"] = {
            "n": int(attack.sum()),
            "missed_as_benign_rate": float((final_labels[attack] == "Benign").mean()),
            "flagged_as_known_attack_rate": float(predicted_attack_class[attack].mean()),
            "sent_to_triage_rate": float(rejected[attack].mean()),
        }

    return report


def outcome_table(true_labels, final_labels, class_names: list[str]) -> pd.DataFrame:

    true_labels = pd.Series(np.asarray(true_labels, dtype=object), name="true")
    final_labels = pd.Series(np.asarray(final_labels, dtype=object), name="predicted")

    present = list(pd.unique(true_labels))
    others = sorted(c for c in present if c not in class_names and c != PSEUDO_UNKNOWN_CLASS)
    rows = [c for c in class_names if c in present]
    rows += [PSEUDO_UNKNOWN_CLASS] if PSEUDO_UNKNOWN_CLASS in present else []
    rows += others

    table = pd.crosstab(true_labels, final_labels)
    return table.reindex(index=rows, columns=list(class_names) + [UNKNOWN_LABEL], fill_value=0)


def format_report(report: dict) -> str:
    lines = []
    k = report["known"]
    lines.append(
        f"  Known ({k['n']}): closed-set macro-F1 {k['closed_set_macro_f1']:.4f} | "
        f"open-set macro-F1 {k['open_set_macro_f1']:.4f} | "
        f"false rejection {k['false_rejection_rate']:.2%} | accuracy on accepted {k['accuracy_on_accepted']:.4f}"
    )
    for name, label in (("pseudo_unknown_DoS", "Pseudo-unknown DoS"), ("real_unknown", "Real unknown")):
        if name in report:
            u = report[name]
            lines.append(
                f"  {label} ({u['n']}): rejected {u['rejection_rate']:.2%} | "
                f"falsely accepted {u['false_acceptance_rate']:.2%} | accepted as Benign {u['accepted_as_benign_rate']:.2%}"
            )
    if "benign" in report:
        b = report["benign"]
        lines.append(f"  Benign ({b['n']}): false positive {b['false_positive_rate']:.2%} | sent to triage {b['triage_rate']:.2%}")
    if "attack" in report:
        a = report["attack"]
        lines.append(
            f"  Attacks ({a['n']}): missed as Benign {a['missed_as_benign_rate']:.2%} | "
            f"flagged as known attack {a['flagged_as_known_attack_rate']:.2%} | sent to triage {a['sent_to_triage_rate']:.2%}"
        )
    return "\n".join(lines)
