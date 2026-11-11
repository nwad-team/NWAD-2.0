import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred, num_classes):
    """Calculate Macro-F1 across all known classes."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Cannot calculate Macro-F1 on empty inputs.")

    return f1_score(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
