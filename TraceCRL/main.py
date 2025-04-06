import os
import click
import pandas as pd
from sdk.base import TADTemplate
from process_data import tracecrl_preprocess
from train import tracecrl_train
from test import tracecrl_test
from generate_operations import get_operations
from node_embeddings import get_node_embeddings
from data_format import *


class TraceCRL(TADTemplate):
    def preprocess_data(self):
        tracecrl_preprocess(self.dataset_name, self.data_path)
        get_operations(self.dataset_name)
        get_node_embeddings(self.dataset_name)

    def train(self):
        tracecrl_train(self.dataset_name)

    def test(self):
        tracecrl_test(self.dataset_name)


@click.command()
@click.option('--mode', required=True, type=str)
@click.option('--dataset_name', required=True, type=str)
@click.option('--data_path', help='The path of original data', required=False, type=str)
def main(mode, dataset_name, data_path):
    arg = TraceCRL(dataset_name=dataset_name, data_path=data_path)
    if mode == 'preprocess':
        arg.preprocess_data()
    elif mode == 'train':
        arg.train()
    elif mode == 'test':
        arg.test()
    else:
        print("Please choose specific mode from 'preprocess', 'train' and 'test'")


if __name__ == '__main__':
    main()
