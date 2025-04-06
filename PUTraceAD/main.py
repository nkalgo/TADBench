import os
import click
import pandas as pd
from bert import get_bert_embeddings
from sdk.base import TADTemplate
from process_data import putracead_preprocess
from train import putracead_train
from test import putracead_test
from data_format import *


class PUTraceAD(TADTemplate):
    def preprocess_data(self):
        putracead_preprocess(self.dataset_name, self.data_path)
        get_bert_embeddings(self.dataset_name)

    def train(self):
        putracead_train(self.dataset_name)

    def test(self, anomaly_type='total'):  # total, structure, latency
        putracead_test(self.dataset_name, anomaly_type)
        pass

@click.command()
@click.option('--mode', required=True, type=str)
@click.option('--dataset_name', required=True, type=str)
@click.option('--data_path', help='The path of original data', required=False, type=str)
@click.option('--anomaly_type', required=False, type=str)
def main(mode, dataset_name, data_path, anomaly_type):
    arg = PUTraceAD(dataset_name=dataset_name, data_path=data_path)
    if mode == 'preprocess':
        arg.preprocess_data()
    elif mode == 'train':
        arg.train()
    elif mode == 'test':
        arg.test(anomaly_type)
    else:
        print("Please choose specific mode from 'preprocess', 'train' and 'test'")


if __name__ == '__main__':
    main()
