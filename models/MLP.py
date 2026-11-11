import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self,input_dim: int,num_classes: int = 5,hidden_dims: tuple[int, ...] = (256, 128),embedding_dim: int = 32,dropout: float = 0.3,):
        super().__init__()

        self.config = {
            "input_dim": int(input_dim),
            "num_classes": int(num_classes),
            "hidden_dims": [int(h) for h in hidden_dims],
            "embedding_dim": int(embedding_dim),
            "dropout": float(dropout),
        }

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(previous_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            previous_dim = hidden_dim

        layers.extend([
            nn.Linear(previous_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim, affine=False),
        ])

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding: bool = False):
        embedding = self.feature_extractor(x)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding

        return logits

    def get_embedding(self, x):
        return self.feature_extractor(x)
