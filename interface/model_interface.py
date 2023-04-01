import abc

class ModelInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        methods = ["train", "predict", "weights", "score", "loss", "get_coefs", "save"]
        return all(hasattr(subclass, method) for method in methods) and all(callable(getattr(subclass, method)) for method in methods) or NotImplemented

    @abc.abstractmethod
    def train(self, x: list[list[float]], y: list[float]):
        """train the model given the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: list[float]) -> float:
        """predict the model given the data x"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def weights(self):
        """get the weights of the model"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def score(self, x: list[list[float]], y: list[float]):
        """get the score of the model"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def loss(self, y_true: list[float], y_pred: list[float]):
        """get the loss of the model"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_coefs(self):
        """get the coefs of the model"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def save(self, fileName: str):
        """save the model"""
        raise NotImplementedError
