import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from datetime import datetime
from logger import get_logger


class MLModelCollection:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, submit: pd.DataFrame) -> None:
        """

        :param train_data:
        :param test_data:
        :param submit:
        """
        self.log = get_logger('ml_modeling')
        self.models = [LinearRegression(),
                       ElasticNet(),
                       Ridge(),
                       Lasso(),
                       RandomForestRegressor(),
                       XGBRegressor(),
                       CatBoostRegressor(verbose=False, ),
                       GradientBoostingRegressor()]
        self.train_data = train_data
        self.target = np.log1p(train_data['count'])
        del self.train_data['count']
        self.test_data = test_data
        self.submit = submit

        self.num_columns = ['temp', 'atemp', 'humidity', 'windspeed']
        self.ct = ColumnTransformer(
                transformers=[
                    ("minmax", MinMaxScaler(), self.num_columns)
                ]
            )
        self.model_scores = {}

        # TODO 각 ML 모델별 parameter
        self.parameters = {
            'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],
            'max_depth': [3, 5, 7, 9, 11, 13, 15],
            'l2_leaf_reg': [1, 3, 5, 7, 9]
        }  # CatBoostRegressor parameter

    def data_preprocessing(self, is_train: bool, is_one_hot: bool = True) -> pd.DataFrame:
        """
        데이터 전처리 메서드입니다.


        :param is_train:
        :param is_one_hot:
        :return:
        """
        if is_train:
            input_data = self.train_data
            del input_data['casual'], input_data['registered']
        else:
            input_data = self.test_data

        input_data.datetime = input_data.datetime.apply(pd.to_datetime)
        input_data['year'] = input_data.datetime.apply(lambda x: str(x.year))
        input_data['month'] = input_data.datetime.apply(lambda x: str(x.month))
        input_data['time'] = input_data.datetime.apply(lambda x: str(x.hour))
        input_data['weekday'] = input_data.datetime.apply(lambda x: str(x.weekday()))
        del input_data['datetime']

        input_data.holiday = input_data.holiday.apply(lambda x: str(x))
        input_data.workingday = input_data.workingday.apply(lambda x: str(x))
        input_data.weather = input_data.weather.apply(lambda x: str(x))
        input_data.season = input_data.season.apply(lambda x: str(x))

        try:
            input_data[self.num_columns] = self.ct.transform(input_data)
        except NotFittedError as E:
            self.log.warning("ColumnTransformer is not Fitted")
            self.log.warning(E)
            input_data[self.num_columns] = self.ct.fit_transform(input_data)

        if is_one_hot:
            return pd.get_dummies(input_data)
        else:
            return input_data

    @staticmethod
    def rmsle(y: np.array, pred: np.array) -> float:
        log_y = np.log1p(y)
        log_pred = np.log1p(pred)
        squared_error = (log_y - log_pred) ** 2
        return np.mean(squared_error) ** .5

    def main(self) -> None:
        data = self.data_preprocessing(is_train=True, is_one_hot=True)
        x_train, x_val, y_train, y_val = train_test_split(data, self.target, test_size=0.3, random_state=1993)
        for idx, model in enumerate(self.models):
            self.log.info(f"{model.__class__.__name__} Start")
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            score = self.rmsle(np.expm1(y_val), np.expm1(pred))
            self.log.info(f"{model.__class__.__name__}'s RMSLE is {score}")
            self.model_scores[(idx, model.__class__.__name__)] = score
        best_model_idx, best_model_name = min(self.model_scores, key=self.model_scores.get)
        self.log.info(f"The Best Model is {best_model_name}")

        best_model = self.models[best_model_idx]
        self.log.info(f"GridSearchCV is Started!")
        gs = GridSearchCV(estimator=best_model, param_grid=self.parameters,
                          cv=3, n_jobs=-1, scoring='neg_mean_squared_log_error')
        gs.fit(x_train, y_train)
        pred = gs.predict(x_val)
        score = self.rmsle(np.expm1(y_val), np.expm1(pred))
        self.log.info(f"The Best Parameter is {gs.best_params_}")
        self.log.info(f"Score is {score}")

        dummy = pd.DataFrame(columns=data.columns)
        test_data = self.data_preprocessing(is_train=False, is_one_hot=True)
        test_data = pd.concat([dummy, test_data])

        for col in test_data.columns:
            if test_data[col].dtype == 'object':
                test_data[col] = test_data[col].apply(lambda x: np.uint8(x))

        pred = np.expm1(gs.predict(test_data))
        self.submit['count'] = pred
        self.submit.to_csv(f'./submit/{datetime.now().strftime(f"%Y%m%d_{best_model_name}")}.csv',
                           encoding='utf-8', index=False)


if __name__ == '__main__':
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    submit = pd.read_csv('./data/sampleSubmission.csv')
    model_collection = MLModelCollection(train_data=train, test_data=test, submit=submit)
    model_collection.main()
