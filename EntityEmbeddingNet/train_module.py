import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from EntityEmbeddingNet.model import EntityEmbNet


class EntityEmbNetModule(LightningModule):
    def __init__(self, cat_vars: dict, num_vars: list) -> None:
        super(EntityEmbNetModule, self).__init__()
        self.model = EntityEmbNet(cat_vars=cat_vars, num_vars=num_vars)
        self.loss_fn = MSELoss()

    def configure_optimizers(self):
        return AdamW(lr=1e-4, params=self.model.parameters())

    def forward(self, x_num: torch.Tensor, x_cat: dict) -> torch.Tensor:
        return self.model(x_num, x_cat)

    def _step(self, batch, is_train: bool = True):
        x_num, x_cat, y = batch
        y_hat = self(x_num, x_cat)
        if is_train:
            return self.loss_fn(y, y_hat)
        else:
            return self.rmsle(torch.expm1(y), torch.expm1(y_hat))

    @staticmethod
    def rmsle(y: torch.Tensor, pred: torch.Tensor) -> float:
        log_y = torch.log1p(y)
        log_pred = torch.log1p(pred)
        squared_error = (log_y - log_pred) ** 2
        return torch.mean(squared_error) ** .5

    def training_step(self, batch, batch_idx, **kwargs) -> dict:
        loss = self._step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, **kwargs) -> dict:
        loss = self._step(batch, False)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return {'val_loss': loss}
