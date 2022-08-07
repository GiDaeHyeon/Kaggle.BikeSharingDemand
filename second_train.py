import numpy as np
import pandas as pd

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from EntityEmbeddingNet.train_module import EntityEmbNetModule
from EntityEmbeddingNet.data_module import EntityEmbNetDataModule, EntityEmbNetDataset

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

train = pd.read_csv('./data/train.csv')
target = np.log1p(train['count'])
del train['count']

def data_preprocessing(input_data: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    if is_train:
        del input_data['casual'], input_data['registered']

    input_data.datetime = input_data.datetime.apply(pd.to_datetime)
    input_data['year'] = input_data.datetime.apply(lambda x: x.year)
    input_data['month'] = input_data.datetime.apply(lambda x: x.month)
    input_data['time'] = input_data.datetime.apply(lambda x: x.hour)
    input_data['weekday'] = input_data.datetime.apply(lambda x: x.weekday())
    del input_data['datetime']
    return pd.get_dummies(input_data)

train = data_preprocessing(train, True)

num_vars = ['temp', 'atemp', 'humidity', 'windspeed']
cat_vars = list(train.columns)
for v in num_vars:
    cat_vars.remove(v)
cat_vars = {c: len(train[c].unique()) for c in cat_vars}

for v in cat_vars:
    if min(train[v]) != 0:
        train[v] -= 1
    if v == 'year':
        train[v] -= 2010

ct = ColumnTransformer(
    transformers=[
        ("minmax", MinMaxScaler(), num_vars)
    ]
)
train[num_vars] = ct.fit_transform(train)
x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.3, random_state=1993)
data_module = EntityEmbNetDataModule(x_train, x_val, y_train, y_val, cat_vars, num_vars)
network = EntityEmbNetModule(cat_vars, num_vars)

logger = TensorBoardLogger(
                           save_dir="./logs/EntityEmbeddingNet",
                           name="regression",
                           default_hp_metric=False,
                           )

early_stop_callback = EarlyStopping(
                                    monitor='val_loss',
                                    min_delta=1e-4,
                                    patience=20,
                                    verbose=True,
                                    mode='min'
                                    )

trainer = Trainer(max_epochs=200,
                  callbacks=[early_stop_callback],
                  logger=logger
                  )

if __name__ == '__main__':
    trainer.fit(network, datamodule=data_module)
