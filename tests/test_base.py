import numpy as np
from vowpalwabbit import pyvw as vw

class BaseModelTest:
    estimators = None

    def test_train(self, data):
        '''Test that the model can be trained'''
        for modelClass in self.estimators:
            model = modelClass()
            assert model.weights() is None
            model.train(data.x, data.y)
            assert model.weights() is not None

    def test_predict(self, data):
        '''Test that the model can be trained'''
        for model in self.estimators:
            model = model()
            model.train(data.x, data.y)
            assert np.allclose(model.predict(data.x_test), model.predict(data.x_test))
            assert np.allclose(model.predict(data.x_test), model.predict(data.x_test))

    def test_loss(self, data):
        '''Test model loss reduce with more passes'''
        for modelClass in self.estimators:
            n_passes = 2
            model = modelClass(learning_rate=0.5, passes=n_passes)
            model.train(data.x, data.y)
            lossBefore = model.loss(data.y, model.predict(data.x))
            model = modelClass(learning_rate=0.5, passes=n_passes + 2)
            model.train(data.x, data.y)
            lossAfter = model.loss(data.y, model.predict(data.x))
            assert abs(lossBefore) > abs(lossAfter)
            # model that has more passes should have a lower loss function

    def test_passes(self, data):
        '''Test that the model's score is better with more passes'''
        for modelClass in self.estimators:
            n_passes = 2
            model = modelClass(passes=n_passes)
            assert getattr(model, "passes") == n_passes
            model.train(data.x, data.y)
            scoresBefore = model.score(data.x, data.y)
            model = modelClass(passes=n_passes + 2)
            model.train(data.x, data.y)
            assert model.score(data.x, data.y) > scoresBefore

    def test_score(self, data):
        '''Test model score is within a reasonable range'''
        for modelClass in self.estimators:
            model = modelClass()
            model.train(data.x, data.y)
            residual = model.score(data.x_test, data.y_test)
            assert residual > -0.2

    def test_features_importance(self, data):
        '''Test that the model is using most of the features'''
        for modelClass in self.estimators:
            model = modelClass()
            model.train(data.x, data.y)
            importance = model.get_coefs()
            importance_Features_count = 0
            for i in range(0, importance.shape[0]):
                for v in importance.getrow(i).toarray()[0]:
                    if abs(v) > 0.01:
                        importance_Features_count += 1
            assert importance_Features_count >= 0.8 * data.x.shape[1]

    def test_overfitting(self, data):
        '''Test that the model is not overfitting by comparing training and test score'''
        for modelClass in self.estimators:
            model = modelClass()
            model.train(data.x, data.y)
            training_score = model.score(data.x, data.y)
            test_score = model.score(data.x_test, data.y_test)
            assert abs(training_score - test_score) < 1
