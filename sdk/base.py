from abc import ABC, abstractmethod
import pandas as pd


class TADTemplate(ABC):
    def __init__(self, dataset_name=None, train_path=None, test_path=None, output_path=None):
        self.dataset_name = dataset_name
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = output_path

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass




