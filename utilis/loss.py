import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):

    def __init__(self, weight=None, label_smoothing=0.0):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits, target):
        return self.loss(logits, target.long())


class CenterLoss(nn.Module):

    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))

    def forward(self, embeddings, target):
        centers = self.centers[target.long()]
        return 0.5 * (embeddings - centers).pow(2).sum(dim=1).mean()


class JointLoss(nn.Module):

    def __init__(self,loss_type: str,num_classes: int,embedding_dim: int,center_lambda: float = 0.01,label_smoothing: float = 0.0,):
        super().__init__()

        if loss_type not in {"ce", "center"}:
            raise ValueError("loss_type must be 'ce' or 'center'.")

        self.loss_type = loss_type
        self.center_lambda = center_lambda
        self.ce = ClassificationLoss(label_smoothing=label_smoothing)
        self.center = CenterLoss(num_classes, embedding_dim) if loss_type == "center" else None

    def forward(self, logits, embeddings, target):
        ce = self.ce(logits, target)

        if self.center is None:
            zero = torch.zeros((), device=logits.device)
            return ce, ce.detach(), zero

        center = self.center(embeddings, target)
        total = ce + self.center_lambda * center
        return total, ce.detach(), center.detach()
