import time
from typing import *

from tracegnn.models.gtrace.models.level_model import LevelModel
from .config import ExpConfig

import mltk
import dgl
from loguru import logger
import dgl.dataloading
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import numpy as np
import multiprocessing as mp
import os

from tracegnn.data import *
from tracegnn.utils.analyze_nll import analyze_anomaly_nll
from .utils import dgl_graph_key
from .models.level_model import calculate_nll


def evaluate(config: ExpConfig,
          dataloader: dgl.dataloading.GraphDataLoader, 
          model: LevelModel):
    device = config.device
    n_z = config.Model.n_z

    logger.add(f'evaluate-log/evaluate-loss-{config.dataset}.log')
    # Train model
    logger.info('Start Evaluation with nll...')
    model.eval()
    start_time1 = time.time()
    nll_list = []
    label_list = []

    latency_nll_list = []
    structure_nll_list = []
    graph_label_list = []

    # sample_label for debugging
    node_sample_label_list = []
    graph_sample_label_list = []

    nll_with_nodes = {}

    with torch.no_grad():
        t = tqdm(dataloader) if config.enable_tqdm else dataloader
        test_graphs: dgl.DGLGraph
        label_graphs: dgl.DGLGraph
        graph_labels: torch.Tensor

        for el, (test_graphs, graph_labels) in enumerate(t):
            # Empty cache first
            if 'cuda' in config.device:
                torch.cuda.empty_cache()

            test_graphs = test_graphs.to(device)
            pred = model(test_graphs, n_z=n_z)
            nll_structure, nll_latency = calculate_nll(config, pred, test_graphs)
            test_graphs.ndata['nll_latency'] = nll_latency
            test_graph_list: List[dgl.DGLGraph] = dgl.unbatch(test_graphs)

            if len(test_graph_list) != len(graph_labels):
                print(
                    f"Mismatch between test_graph_list length ({len(test_graph_list)}) and graph_labels length ({len(graph_labels)}) at batch {el}")

            for i in range(test_graphs.batch_size):
                if i >= len(test_graph_list):
                    print(f"Index {i} out of range for test_graph_list of length {len(test_graph_list)}")

                graph_key = dgl_graph_key(test_graph_list[i])

                nll_list.extend(test_graph_list[i].ndata['nll_latency'].tolist())
                label_list.extend(test_graph_list[i].ndata['anomaly'].tolist())
                node_sample_label_list.extend([graph_key] * test_graph_list[i].num_nodes())
                graph_sample_label_list.append(graph_key)
                graph_label_list.append(graph_labels[i])


                nll_with_nodes.setdefault(test_graph_list[i].num_nodes(), [])
                nll_with_nodes[test_graph_list[i].num_nodes()].append(nll_structure[i].item())

                # The latency nll of the whole graph is the mean of the node degrees
                graph_latency_nll = test_graph_list[i].ndata['nll_latency'].max().item()

                latency_nll_list.append(graph_latency_nll)
                structure_nll_list.append(nll_structure[i].item())
        duration_time1 = time.time() - start_time1
        # Set evaluation output
        logger.info('--------------------Node Level-----------------------')
        # Get node level result
        node_result = analyze_anomaly_nll(
            nll_latency=np.array(nll_list, dtype=np.float32),
            nll_drop=np.array(nll_list, dtype=np.float32),
            label_list=np.array(label_list, dtype=np.int64),
            threshold=np.percentile(nll_list, 98),
            sample_label_list=node_sample_label_list,
            method='gtrace',
            dataset=config.dataset,
            save_dict=True,
            save_filename='evaluate_%s_node_level.csv' % config.dataset
        )
        logger.info(node_result)

        start_time2 = time.time()
        logger.info('-------------------Graph Level-----------------------')
        # arr = np.array(graph_label_list, dtype=np.int64)
        # total_mask = arr != 3
        total_result = analyze_anomaly_nll(
            nll_latency=np.array(latency_nll_list, dtype=np.float32),
            nll_drop=np.array(structure_nll_list, dtype=np.float32),
            label_list=np.array(graph_label_list, dtype=np.int64),
            threshold=None,
            method='gtrace',
            dataset=config.dataset,
            save_dict=True,
            save_filename='evaluate_%s_graph_level.csv' % config.dataset
        )
        logger.info(f'model_time={duration_time1}')
        test_seconds = time.time() - start_time2 + duration_time1
        logger.info(f'model_time + calculate_time={test_seconds}')
        logger.info(total_result)

    model.train()
