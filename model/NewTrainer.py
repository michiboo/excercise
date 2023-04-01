from vowpalwabbit.sklearn import (
    VWRegressor,
)
from sklearn.metrics import mean_squared_error
import numpy as np
from interface.model_interface import ModelInterface

# for simplicity I used VMRegressor from vowpalwabbit
class NewTrainer(ModelInterface):
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", None)
        self.passes = kwargs.get("passes", None)
        self.model = VWRegressor(learning_rate=self.learning_rate, passes=self.passes)

    def train(self, x: list[list[float]], y: list[float]):
        self.model.fit(np.array(x), np.array(y).ravel())

    def predict(self, x: list[float]) -> float:
        return self.model.predict(np.array(x))

    def weights(self):
        return self.model.get_vw()

    def score(self, x: list[list[float]], y: list[float]):
        return self.model.score(np.array(x), np.array(y).ravel())

    def loss(self, y_true: list[float], y_pred: list[float]):
        return mean_squared_error(np.array(y_true), np.array(y_pred))

    def get_coefs(self):
        return self.model.get_coefs()

    def save(self, fileName: str):
        self.model.save(fileName)


