
import torch.nn as nn

class ClassificationLoss(nn.Module):
    """Cross-entropy loss for multiclass classification."""

    def __init__(self, weight=None, label_smoothing=0.0):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits, target):

        return self.loss(logits, target.long())