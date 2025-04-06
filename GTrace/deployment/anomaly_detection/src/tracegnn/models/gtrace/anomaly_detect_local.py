import mltk
import time
import click
import json
import dgl
import torch
import pandas as pd

from deployment.anomaly_detection.src.tracegnn.models.gtrace.cache.tree_cache import initialize_data_fetch, \
    start_workers, get_tree_cache_batch
from deployment.anomaly_detection.src.tracegnn.models.gtrace.evaluate import Evaluator
from deployment.anomaly_detection.src.tracegnn.models.gtrace.test_utils import load_model_local
from tracegnn.data import *
from tracegnn.data.trace_graph import *
# from tracegnn.models.gtrace.evaluate import Evaluator
# from tracegnn.models.gtrace.test_utils import load_model_local
from tracegnn.utils import *
from tracegnn.models.gtrace.utils import dgl_graph_key
# from tracegnn.models.gtrace.cache1.tree_cache import get_tree_cache_batch, initialize_data_fetch, start_workers
from typing import *
import cProfile

from datetime import datetime, timedelta
from loguru import logger
import os


hyper_params = {
    'prefix': '../../trained_model',
    'dataset': '../../../dataset/dataset_a',
    'test_file': '../../../dataset/dataset_a/dataset_a_2022-05-02.csv',
    'dataset_name': 'dataset_a',
    'method': 'gtrace',
    'device': 'cuda',
    'use_biased': True,
    'eval_n_z': 5,
    'detect_freq': 502 if os.getenv('DETECT_FREQ') is None else int(os.getenv('DETECT_FREQ')),
    'refresh_freq': 100000,
    'cache_items': 2 ** 18 + 1000
}


class AnomalyDetector:
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        self.start_time = None
        self.total_cnt = 0
        
        # Evaluator
        self.evaluator: Evaluator = None

    def __load_model(self):
        self.model, self.config, self.id_manager = load_model_local(hyper_params['prefix'], device=hyper_params['device'])
        # Initialize evaluator
        self.evaluator = Evaluator(self.config, self.model, device=hyper_params['device'])
        print("model:", self.model)
        print("config:", self.config)
        print("id_manager:", self.id_manager)
        print("evaluator:", self.evaluator)

    def __consume(self, result_dict):
        # Start detect trace with the newest model
        # Get evaluation results
        self.evaluator.get_batch_nll(result_dict)
        cur_time = time.perf_counter()

        if self.start_time is not None:
            time_consume = cur_time - self.start_time

            logger.info(f'Detect {self.total_cnt} traces in {time_consume} seconds.' 
                        f'(cur_speed = {self.total_cnt / (time_consume + 1e-7)})')
            
    
    def start_service(self):
        self.__load_model()
        initialize_data_fetch(
            test_file=hyper_params['test_file'],
            id_manager_file=os.path.join(hyper_params['dataset'], 'processed')
        )

        # Start serving
        start_workers()

        # Dry run
        logger.info('Cold starting...')
        self.start_time = time.perf_counter()
        result = get_tree_cache_batch(hyper_params['detect_freq'], device=hyper_params['device'])
        print("cold starting result:", result)
        self.__consume(result)


        logger.info('Start service')
        i = 0     
        while i < 30:
            self.start_time = time.perf_counter()
            result = get_tree_cache_batch(hyper_params['detect_freq'], device=hyper_params['device'])
            # print("start service result:", result)
            self.total_cnt = hyper_params['detect_freq']
            self.__consume(result)
            i = i + 1


def main():
    # Print config
    logger.info(f"DETECT_FREQ: {hyper_params['detect_freq']}")

    anomaly_detector = AnomalyDetector(hyper_params)
    anomaly_detector.start_service()


if __name__ == '__main__':
    cProfile.run("main()", filename='run.profile')
