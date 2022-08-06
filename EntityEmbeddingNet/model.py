import torch
import torch.nn as nn


class EntityEmbNet(nn.Module):
    def __init__(self, cat_vars: dict, num_vars: list) -> None:
        super(EntityEmbNet, self).__init__()
        self.emb_dim = 0
        self.embeddings = nn.ModuleDict({key: self.generate_embedding_layer(value) for key, value in cat_vars.items()})
        self.fc = nn.Sequential(
            nn.Dropout(.1),
            nn.Linear(self.emb_dim + len(num_vars), 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.output_layer = nn.Linear(64, 1)

    def generate_embedding_layer(self, num_uniques: int) -> nn.Module:
        self.emb_dim += int(num_uniques / 2)
        return nn.Sequential(nn.Embedding(num_uniques, int(num_uniques / 2)), nn.BatchNorm1d(1), nn.GELU())

    def forward(self,
                x_num: torch.Tensor,
                x_cat: dict) -> torch.Tensor:
        embedding = torch.cat([self.embeddings[key](value).squeeze(1) for key, value in x_cat.items()], axis=1)
        logit = torch.cat([x_num.squeeze(), embedding], axis=1)
        logit = self.fc(logit)
        return self.output_layer(logit)
