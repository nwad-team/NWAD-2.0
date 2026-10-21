import torch.nn as nn

class MLPClassifier(nn.Module):
    """Fully connected classifier with an extractable embedding."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 6,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()

        hidden1, hidden2 = hidden_dims

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        embedding = self.feature_extractor(x)
        logits = self.classifier(embedding)
        return logits

    def get_embedding(self, x):
        """Return the representation before the classification layer."""
        return self.feature_extractor(x)
