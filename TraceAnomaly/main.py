import os
import click
import pandas as pd
from sdk.base import TADTemplate
from data_to_STV import traceanomaly_preprocess
from train import traceanomaly_train
from test import traceanomaly_test
from data_format import *


class TraceAnomaly(TADTemplate):
    def preprocess_data(self):
        traceanomaly_preprocess(self.dataset_name, self.data_path)

    def train(self):
        traceanomaly_train(self.dataset_name)

    def test(self, anomaly_type='total'):  # total, structure, latency
        traceanomaly_test(self.dataset_name, anomaly_type)


@click.command()
@click.option('--mode', required=True, type=str)
@click.option('--dataset_name', required=True, type=str)
@click.option('--data_path', help='The path of original data', required=False, type=str)
@click.option('--anomaly_type', required=False, type=str)
def main(mode, dataset_name, data_path, anomaly_type):
    arg = TraceAnomaly(dataset_name=dataset_name, data_path=data_path)
    if mode == 'preprocess':
        arg.preprocess_data()
    elif mode == 'train':
        arg.train()
    elif mode == 'test':
        arg.test(anomaly_type=anomaly_type)
    else:
        print("Please choose specific mode from 'preprocess', 'train' and 'test'")


if __name__ == '__main__':
    main()
