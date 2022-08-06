import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class EntityEmbNetDataset(Dataset):
    def __init__(self, xs: pd.DataFrame, ys: pd.Series,
                 cat_vars: dict, num_vars: list, is_test: bool = False) -> None:
        super(EntityEmbNetDataset, self).__init__()
        self.xs = xs.reset_index(drop=True)
        self.ys = ys.reset_index(drop=True)
        self.cat_vars = cat_vars
        self.num_vars = num_vars
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx) -> tuple:
        x = self.xs.iloc[idx]
        y = self.ys[idx]

        x_num = torch.Tensor([[float(x[var])] for var in self.num_vars]).long()
        x_cat = {var: torch.Tensor([int(x[var])]).long() for var in list(self.cat_vars.keys())}

        if self.is_test:
            return x_num, x_cat
        else:
            return x_num, x_cat, torch.Tensor([int(y)]).float()


class EntityEmbNetDataModule(LightningDataModule):
    def __init__(self,
                 x_train: pd.DataFrame,
                 x_val: pd.DataFrame,
                 y_train: pd.Series,
                 y_val: pd.Series,
                 cat_vars: dict,
                 num_vars: list) -> None:
        super(EntityEmbNetDataModule, self).__init__()
        self.train_set = EntityEmbNetDataset(xs=x_train, ys=y_train, cat_vars=cat_vars, num_vars=num_vars)
        self.val_set = EntityEmbNetDataset(xs=x_val, ys=y_val, cat_vars=cat_vars, num_vars=num_vars)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_set,
                          batch_size=64,
                          num_workers=2,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_set,
                          batch_size=128,
                          num_workers=2,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=False)
