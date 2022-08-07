import torch
import torch.nn as nn
from torch.nn.functional import relu


class EntityEmbNet(nn.Module):
    def __init__(self, cat_vars: dict, num_vars: list) -> None:
        super(EntityEmbNet, self).__init__()
        self.emb_dim = 0
        self.embeddings = nn.ModuleDict({key: self.generate_embedding_layer(value) for key, value in cat_vars.items()})
        self.fc = nn.Sequential(
            nn.Dropout(.1),
            nn.Linear(self.emb_dim + len(num_vars), 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        self.output_layer = nn.Linear(256, 1)

    def generate_embedding_layer(self, num_uniques: int) -> nn.Module:
        self.emb_dim += int(num_uniques / 2)
        return nn.Sequential(
            nn.Embedding(num_uniques, int(num_uniques / 2)),
            nn.Flatten(),
            nn.BatchNorm1d(int(num_uniques / 2)),
            nn.GELU()
        )

    def forward(self,
                x_num: torch.Tensor,
                x_cat: dict) -> torch.Tensor:
        embedding = torch.cat([self.embeddings[key](value) for key, value in x_cat.items()], axis=1)
        logit = torch.cat([x_num, embedding], axis=1)
        logit = self.fc(logit)
        return relu(self.output_layer(logit))
