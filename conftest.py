import pytest
from collections import namedtuple
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


Dataset = namedtuple("Dataset", "x, y, x_test, y_test")


# modify this fixture to return a Dataset object
@pytest.fixture(scope="session")
def data(request):
    dataFile = request.config.getoption("--data")
    if dataFile:
        with open(dataFile, "r") as f:
            tmp = f.readlines()
            tmp = [i.split(",") for i in tmp]
            x, y = [i[:-1]for i in tmp], [[i[-1]] for i in tmp]
            x = np.array(x).astype(np.float32)
            y = np.array(y).astype(np.float32)
            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=256
            )
    else:
        n = 100
        x, y = datasets.make_hastie_10_2(n_samples=n, random_state=1)
        x = x.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=256
    )
    return Dataset(x=X_train, y=y_train, x_test=X_test, y_test=y_test)


def class_import(name):
    components = name.split(".")
    mod = __import__(".".join(components[:2]),  fromlist=['*'])
    for comp in components[2:]:
        mod = getattr(mod, comp)
    return mod


@pytest.fixture(scope="session", autouse=True)
# python -m pytest .\tests\ --train NewTrainer.NewTrainer --save
def regression_models(request, data):
    model = request.config.getoption("--train")
    if model:
        yield [class_import(model)]
    save_model = request.config.getoption("--save")
    
    if save_model:
        for x in [class_import(model)]:
            tmp = x()
            tmp.train(data.x, data.y)
            tmp.save(f"{model}.vw")

def pytest_addoption(parser):
    parser.addoption(
        "--train", action="store", help="train a new model from a class", default=None
    )
    parser.addoption(
        "--data", action="store", help="specify csv data path", default=None
    )
    parser.addoption(
        "--save", action="store_true", help="save model if all test passed", default=False
    )
