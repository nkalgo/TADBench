from abc import ABC, abstractmethod


class TADTemplate(ABC):
    def __init__(self, dataset_name=None, data_path=None):
        self.dataset_name = dataset_name
        self.data_path = data_path

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
