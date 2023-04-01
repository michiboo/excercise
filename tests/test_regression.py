from test_base import BaseModelTest
import pytest
from scipy.stats import pearsonr
import numpy as np


class TestRegression(BaseModelTest):
    @pytest.fixture(scope="function", autouse=True)
    def execute_before_any_test(self, regression_models):
        '''Set the models to test'''
        self.estimators = regression_models

    def test_linearity(self, data):
        '''Test that the model is linear'''
        for modelClass in self.estimators:
            model = modelClass()
            model.train(data.x, data.y)
            y_pred = model.predict(data.x_test)
            corrections = []
            for v in range(data.x_test.shape[1]):
                for i in range(y_pred.shape[0]):
                    vars = [data.x_test[i][v] for i in range(y_pred.shape[0])]
                    corrections.append(abs(pearsonr(vars, y_pred).correlation))
            assert np.mean(corrections) > 0.2

