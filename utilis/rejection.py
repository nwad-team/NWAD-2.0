"""Open-set rejection rules, threshold search and metrics.

Two modes share this file:

UNKNOWN mode (one centroid per known class)
    pred   = argmax softmax(logits)          conf = max softmax(logits)
    d_pred = ||z - mu_pred|| / scale_pred    (distance to the PREDICTED class only)

    confidence: accept iff conf >= tau_c
    distance:   accept iff d_pred <= tau_d
    combined:   accept iff conf >= tau_c and d_pred <= tau_d
    accepted -> pred, otherwise -> UNKNOWN

ANOMALY mode (only the Benign centroid)
    H   = entropy(softmax(logits)) / log(C)  (0 = certain, 1 = uniform)
    d_B = ||z - mu_Benign|| / scale_Benign   (distance to Benign only)

    flag iff H >= tau_h and d_B > tau_b
    flagged -> ANOMALY, otherwise -> pred

In both modes the distance is divided by a scale: the q-quantile (default 0.95)
of the training distances of that class to its own centroid. A normalized
distance of 1.0 therefore means "as far as the 95th-percentile training sample
of that class"; 2.0 means twice that radius.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from utilis.experiment import ANOMALY_LABEL, BENIGN_CLASS, PSEUDO_UNKNOWN_CLASS, UNKNOWN_LABEL

METHODS = ("confidence", "distance", "combined")
METRICS = ("euclidean", "mahalanobis")


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def _regularized_inverse(covariance: np.ndarray, ridge: float) -> np.ndarray:
    dim = covariance.shape[0]
    covariance = covariance + ridge * np.trace(covariance) / dim * np.eye(dim)
    return np.linalg.inv(covariance)


class CentroidModel:
    """One centroid per known class (UNKNOWN mode).

    Mahalanobis uses one covariance shared by all classes (pooled within-class),
    as in Lee et al., NeurIPS 2018.
    """

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
            self.precision_matrix = _regularized_inverse(covariance, self.ridge)

        raw = self.raw_distances(embeddings)
        own = raw[np.arange(len(labels)), labels]
        self.scales = np.array([np.quantile(own[labels == k], self.quantile) for k in range(num_classes)])
        return self

    def raw_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """(N, C) distances from every sample to every centroid, before scaling."""
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


class BenignReference:
    """A single reference point: the Benign centroid (ANOMALY mode).

    Mahalanobis uses Benign's OWN covariance, so "far" means far along the
    directions in which benign traffic normally does not vary.
    """

    def __init__(self, metric: str = "euclidean", quantile: float = 0.95, ridge: float = 1e-6):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        self.metric = metric
        self.quantile = quantile
        self.ridge = ridge
        self.mean = None
        self.precision_matrix = None
        self.scale = None

    def fit(self, benign_embeddings: np.ndarray):
        if len(benign_embeddings) < 2:
            raise ValueError("Need at least two Benign training embeddings.")

        self.mean = benign_embeddings.mean(axis=0)

        if self.metric == "mahalanobis":
            centered = benign_embeddings - self.mean
            covariance = centered.T @ centered / (len(benign_embeddings) - 1)
            self.precision_matrix = _regularized_inverse(covariance, self.ridge)

        self.scale = float(np.quantile(self.raw_distance(benign_embeddings), self.quantile))
        return self

    def raw_distance(self, embeddings: np.ndarray) -> np.ndarray:
        diff = embeddings - self.mean
        if self.metric == "euclidean":
            squared = np.einsum("nd,nd->n", diff, diff)
        else:
            squared = np.einsum("nd,de,ne->n", diff, self.precision_matrix, diff)
        return np.sqrt(np.maximum(squared, 0.0))

    def distance(self, embeddings: np.ndarray) -> np.ndarray:
        return self.raw_distance(embeddings) / self.scale

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "quantile": self.quantile,
            "ridge": self.ridge,
            "mean": self.mean.tolist(),
            "precision_matrix": None if self.precision_matrix is None else self.precision_matrix.tolist(),
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenignReference":
        model = cls(data["metric"], data["quantile"], data["ridge"])
        model.mean = np.asarray(data["mean"], dtype=np.float64)
        model.scale = float(data["scale"])
        if data["precision_matrix"] is not None:
            model.precision_matrix = np.asarray(data["precision_matrix"], dtype=np.float64)
        return model


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy divided by log(C): 0 = one-hot, 1 = uniform."""
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    return entropy / np.log(probs.shape[1])


def compute_scores(logits: np.ndarray, embeddings: np.ndarray, centroids: CentroidModel) -> dict:
    """Scores for UNKNOWN mode."""
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


def compute_anomaly_scores(logits: np.ndarray, embeddings: np.ndarray, benign: BenignReference) -> dict:
    """Scores for ANOMALY mode."""
    probs = softmax(logits)
    return {
        "pred": probs.argmax(axis=1),
        "conf": probs.max(axis=1),
        "entropy": normalized_entropy(probs),
        "dist_benign": benign.distance(embeddings),
    }


def subset(scores: dict, mask: np.ndarray) -> dict:
    return {key: value[mask] for key, value in scores.items()}


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def accept_mask(scores: dict, method: str, tau_c: float, tau_d: float, require_agreement: bool = False) -> np.ndarray:
    """UNKNOWN mode: True where the prediction is kept."""
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


def anomaly_mask(scores: dict, tau_h: float, tau_b: float) -> np.ndarray:
    """ANOMALY mode: True where the sample is flagged (uncertain AND far from Benign)."""
    return (scores["entropy"] >= tau_h) & (scores["dist_benign"] > tau_b)


def apply_anomaly_rule(scores: dict, class_names: list[str], tau_h: float, tau_b: float) -> np.ndarray:
    flagged = anomaly_mask(scores, tau_h, tau_b)
    names = np.asarray(class_names, dtype=object)[scores["pred"]]
    return np.where(flagged, ANOMALY_LABEL, names)


# ---------------------------------------------------------------------------
# Threshold search under a false-rejection budget
# ---------------------------------------------------------------------------

def search_thresholds(known: dict, unknown: dict, method: str, max_frr: float, require_agreement: bool = False, n_grid: int = 51,
                      tau_c: float | None = None, tau_d: float | None = None) -> dict:
    """UNKNOWN mode: maximize pseudo-unknown rejection with known FRR <= max_frr.

    Candidates are quantiles of the known-class scores. Each score alone may not
    reject more than max_frr of the knowns, so the grid spans quantiles 0..max_frr.
    A threshold passed as tau_c / tau_d is kept fixed and only the other is searched.
    """
    q = np.linspace(0.0, max_frr, n_grid)[1:]
    uses_c = method in ("confidence", "combined")
    uses_d = method in ("distance", "combined")

    if not uses_c:
        tau_c_grid = np.array([0.0])
    elif tau_c is not None:
        tau_c_grid = np.array([tau_c])
    else:
        tau_c_grid = np.concatenate([[0.0], np.quantile(known["conf"], q)])

    if not uses_d:
        tau_d_grid = np.array([np.inf])
    elif tau_d is not None:
        tau_d_grid = np.array([tau_d])
    else:
        tau_d_grid = np.concatenate([[np.inf], np.quantile(known["dist_pred"], 1.0 - q)])

    best = None
    for c in np.unique(tau_c_grid):
        for d in np.unique(tau_d_grid):
            frr = 1.0 - accept_mask(known, method, c, d, require_agreement).mean()
            if frr > max_frr + 1e-12:
                continue
            rejection = 1.0 - accept_mask(unknown, method, c, d, require_agreement).mean()
            key = (rejection, -frr)
            if best is None or key > best[0]:
                best = (key, float(c), float(d), float(frr), float(rejection))

    if best is None:
        return {"method": method, "feasible": False}

    _, tau_c, tau_d, frr, rejection = best
    return {
        "method": method,
        "feasible": True,
        "tau_c": tau_c,
        "tau_d": tau_d,
        "known_false_rejection": frr,
        "pseudo_unknown_rejection": rejection,
    }


def search_anomaly_thresholds(known: dict, unknown: dict, max_frr: float, n_grid: int = 51,
                              tau_h: float | None = None, tau_b: float | None = None) -> dict:
    """ANOMALY mode: maximize the share of DoS flagged with known false flags <= max_frr.

    Free thresholds sweep the full range of the known-class scores, because a
    low entropy threshold can still be feasible when the Benign-distance
    threshold protects most known samples (and vice versa). A threshold passed
    as tau_h / tau_b is kept fixed. Fixing tau_b by hand is recommended: the
    free search can place it inside the cloud of attacks, where it no longer
    means "far from Benign" but just fits the pseudo-unknown.
    """
    q = np.linspace(0.0, 1.0, n_grid)
    tau_h_grid = np.array([tau_h]) if tau_h is not None else np.unique(np.concatenate([np.quantile(known["entropy"], q), [np.inf]]))
    tau_b_grid = np.array([tau_b]) if tau_b is not None else np.unique(np.concatenate([[0.0], np.quantile(known["dist_benign"], q)]))

    best = None
    for h in tau_h_grid:
        known_uncertain = known["entropy"] >= h
        unknown_uncertain = unknown["entropy"] >= h
        for b in tau_b_grid:
            frr = float((known_uncertain & (known["dist_benign"] > b)).mean())
            if frr > max_frr + 1e-12:
                continue
            flagged = float((unknown_uncertain & (unknown["dist_benign"] > b)).mean())
            key = (flagged, -frr)
            if best is None or key > best[0]:
                best = (key, float(h), float(b), frr, flagged)

    if best is None:  # only possible when the user fixed both thresholds
        return {"method": "anomaly", "feasible": False}

    _, tau_h, tau_b, frr, flagged = best
    return {
        "method": "anomaly",
        "feasible": True,
        "tau_h": tau_h,
        "tau_b": tau_b,
        "known_false_rejection": frr,
        "pseudo_unknown_rejection": flagged,
    }


def tradeoff_curve(known: dict, unknown: dict, method: str, require_agreement: bool = False, n_points: int = 101):
    """(known false-rejection, unknown rejection) pairs across thresholds (UNKNOWN mode)."""
    q = np.linspace(0.0, 1.0, n_points)

    if method == "confidence":
        pairs = [(t, np.inf) for t in np.quantile(known["conf"], q)]
    elif method == "distance":
        pairs = [(0.0, t) for t in np.quantile(known["dist_pred"], 1.0 - q)]
    else:
        qq = np.linspace(0.0, 0.5, 41)
        pairs = [(c, d) for c in np.quantile(known["conf"], qq) for d in np.quantile(known["dist_pred"], 1.0 - qq)]

    points = []
    for tau_c, tau_d in pairs:
        frr = 1.0 - accept_mask(known, method, tau_c, tau_d, require_agreement).mean()
        rej = 1.0 - accept_mask(unknown, method, tau_c, tau_d, require_agreement).mean()
        points.append((frr, rej))

    points = np.array(sorted(points))
    if method == "combined":
        points[:, 1] = np.maximum.accumulate(points[:, 1])
    return points


def anomaly_tradeoff_curve(known: dict, unknown: dict, tau_b: float, n_points: int = 101):
    """Sweep the entropy threshold with the Benign-distance threshold fixed at tau_b."""
    points = []
    for tau_h in np.quantile(known["entropy"], np.linspace(0.0, 1.0, n_points)):
        frr = anomaly_mask(known, tau_h, tau_b).mean()
        flagged = anomaly_mask(unknown, tau_h, tau_b).mean()
        points.append((frr, flagged))
    return np.array(sorted(points))


def auroc(known: dict, unknown: dict) -> dict:
    """Threshold-free separability of known vs pseudo-unknown for each available score."""
    y = np.concatenate([np.zeros(len(known["pred"])), np.ones(len(unknown["pred"]))])

    def score(key, sign=1.0):
        return float(roc_auc_score(y, sign * np.concatenate([known[key], unknown[key]])))

    result = {"confidence": score("conf", -1.0)}
    if "dist_pred" in known:
        result["distance"] = score("dist_pred")
    if "entropy" in known:
        result["entropy"] = score("entropy")
        result["benign_distance"] = score("dist_benign")
    return result


# ---------------------------------------------------------------------------
# Open-set metrics
# ---------------------------------------------------------------------------

def open_set_report(true_labels, final_labels, closed_labels, class_names: list[str], reject_label: str = UNKNOWN_LABEL) -> dict:
    """Metrics for the three goals: attack detection, known-class classification
    and unknown rejection. reject_label is UNKNOWN or ANOMALY depending on the mode."""
    true_labels = np.asarray(true_labels, dtype=object)
    final_labels = np.asarray(final_labels, dtype=object)
    closed_labels = np.asarray(closed_labels, dtype=object)

    known = np.isin(true_labels, class_names)
    pseudo = true_labels == PSEUDO_UNKNOWN_CLASS
    real_unknown = ~known & ~pseudo
    rejected = final_labels == reject_label

    report = {"n_samples": int(len(true_labels)), "reject_label": reject_label}

    report["known"] = {
        "n": int(known.sum()),
        "closed_set_macro_f1": float(f1_score(true_labels[known], closed_labels[known], labels=class_names, average="macro", zero_division=0)),
        "open_set_macro_f1": float(f1_score(true_labels[known], final_labels[known], labels=class_names, average="macro", zero_division=0)),
        "false_rejection_rate": float(rejected[known].mean()),
        "accuracy_on_accepted": float((final_labels == true_labels)[known & ~rejected].mean()) if np.any(known & ~rejected) else float("nan"),
    }

    for name, mask in (("pseudo_unknown_DoS", pseudo), ("real_unknown", real_unknown)):
        if mask.any():
            report[name] = {
                "n": int(mask.sum()),
                "rejection_rate": float(rejected[mask].mean()),
                "false_acceptance_rate": float((~rejected[mask]).mean()),
                "accepted_as_benign_rate": float((final_labels[mask] == BENIGN_CLASS).mean()),
            }

    benign = true_labels == BENIGN_CLASS
    attack = ~benign
    predicted_attack_class = ~rejected & (final_labels != BENIGN_CLASS)

    if benign.any():
        report["benign"] = {
            "n": int(benign.sum()),
            "false_positive_rate": float(predicted_attack_class[benign].mean()),
            "triage_rate": float(rejected[benign].mean()),
        }
    if attack.any():
        report["attack"] = {
            "n": int(attack.sum()),
            "missed_as_benign_rate": float((final_labels[attack] == BENIGN_CLASS).mean()),
            "flagged_as_known_attack_rate": float(predicted_attack_class[attack].mean()),
            "sent_to_triage_rate": float(rejected[attack].mean()),
        }

    return report


def outcome_table(true_labels, final_labels, class_names: list[str], reject_label: str = UNKNOWN_LABEL) -> pd.DataFrame:
    """Rows: true class (known, then DoS, then real unknowns). Columns: output. Counts."""
    true_labels = pd.Series(np.asarray(true_labels, dtype=object), name="true")
    final_labels = pd.Series(np.asarray(final_labels, dtype=object), name="predicted")

    present = list(pd.unique(true_labels))
    others = sorted(c for c in present if c not in class_names and c != PSEUDO_UNKNOWN_CLASS)
    rows = [c for c in class_names if c in present]
    rows += [PSEUDO_UNKNOWN_CLASS] if PSEUDO_UNKNOWN_CLASS in present else []
    rows += others

    table = pd.crosstab(true_labels, final_labels)
    return table.reindex(index=rows, columns=list(class_names) + [reject_label], fill_value=0)


def format_report(report: dict) -> str:
    word = "flagged ANOMALY" if report.get("reject_label") == ANOMALY_LABEL else "rejected"
    lines = []
    k = report["known"]
    lines.append(
        f"  Known ({k['n']}): closed-set macro-F1 {k['closed_set_macro_f1']:.4f} | "
        f"open-set macro-F1 {k['open_set_macro_f1']:.4f} | "
        f"{word} {k['false_rejection_rate']:.2%} | accuracy on accepted {k['accuracy_on_accepted']:.4f}"
    )
    for name, label in (("pseudo_unknown_DoS", "Pseudo-unknown DoS"), ("real_unknown", "Real unknown")):
        if name in report:
            u = report[name]
            lines.append(
                f"  {label} ({u['n']}): {word} {u['rejection_rate']:.2%} | "
                f"accepted as a known class {u['false_acceptance_rate']:.2%} | accepted as Benign {u['accepted_as_benign_rate']:.2%}"
            )
    if "benign" in report:
        b = report["benign"]
        lines.append(f"  Benign ({b['n']}): predicted as attack {b['false_positive_rate']:.2%} | {word} {b['triage_rate']:.2%}")
    if "attack" in report:
        a = report["attack"]
        lines.append(
            f"  Attacks ({a['n']}): missed as Benign {a['missed_as_benign_rate']:.2%} | "
            f"flagged as known attack {a['flagged_as_known_attack_rate']:.2%} | {word} {a['sent_to_triage_rate']:.2%}"
        )
    return "\n".join(lines)
