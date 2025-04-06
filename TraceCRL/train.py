import random
import time
import torch
from sklearn.svm import OneClassSVM
from tqdm import tqdm
import os
import logging
import json
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from model import SIMCLR
from aug_dataset import TraceDataset, TestDataset


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.multiprocessing.set_sharing_strategy('file_system')


def tracecrl_train(dataset_name):
    # param
    learning_rate = 0.001  # 0.001
    epochs = 20  # 20
    normal_classes = [0]
    abnormal_classes = [1]
    batch_size = 32  # 32
    num_workers = 6  # 6
    num_layers = 2
    pooling_type = 'mean'  # mean  add
    gnn_type = 'CGConv'

    aug = 'random'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    dataset = TraceDataset(root='data/%s/train' % dataset_name, aug=aug)
    # output dim relay on gnn_type
    output_dim = dataset.num_node_features

    normal_idx = dataset.normal_idx
    abnormal_idx = dataset.abnormal_idx
    # aug_idx = dataset.aug_idx

    random.shuffle(abnormal_idx)
    random.shuffle(normal_idx)

    train_normal = normal_idx[:int(len(normal_idx) * 1)]
    test_normal = normal_idx[int(len(normal_idx) * 1):]
    # aug_idx = aug_idx[:int(len(normal_idx) * 0.1)]
    train_abnormal = abnormal_idx[:int(len(abnormal_idx) * 0)]
    test_abnormal = abnormal_idx[int(len(abnormal_idx) * 0):]

    train_dataset = Subset(dataset, train_normal + train_abnormal)

    model = SIMCLR(num_layers=num_layers, input_dim=dataset.num_node_features, output_dim=output_dim,
                   num_edge_attr=dataset.num_edge_features, pooling_type=pooling_type).to(device)

    # Set up logging
    output_path = f"results/" + dataset_name + '_{}epoch_{}_{}_{}layers_{}batch_{}'.format(epochs, gnn_type,
                                                                                          pooling_type, num_layers,
                                                                                          batch_size, aug)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = output_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # logging param
    logger.info('----------------------')
    logger.info("dataset size: {}".format(len(dataset)))
    logger.info("node feature number: {}".format(dataset.num_node_features))
    logger.info("edge feature number: {}".format(dataset.num_edge_features))
    logger.info('batch_size: {}'.format(batch_size))
    logger.info('learning_rate: {}'.format(learning_rate))
    logger.info('num_gc_layers: {}'.format(num_layers))
    logger.info("epochs: {}".format(epochs))
    logger.info('pooling_type: {}'.format(pooling_type))
    logger.info('output_dim: {}'.format(output_dim))
    logger.info('aug: {}'.format(aug))
    logger.info('----------------------')

    # sava param
    train_info = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_layers': num_layers,
        'epochs': epochs,
        'pooling_type': pooling_type,
        'num_node_features': dataset.num_node_features,
        'num_edge_features': dataset.num_edge_features,
        'output_dim': output_dim,
        'aug': aug,
        'train_idx': train_normal + train_abnormal,
        'test_idx': test_normal + test_abnormal
    }
    with open(output_path + '/train_info.json', 'w', encoding='utf-8') as json_file:
        json.dump(train_info, json_file)
        logger.info('write train info success')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    start_time = time.time()
    model.train()

    for epoch in range(epochs):
        # start_time = time.time()
        total_loss = 0
        total = 0

        for data in tqdm(train_loader):
            data_o, data_aug_1, data_aug_2 = data
            if aug == 'random' or aug == 'subgraph':
                edge_idx_1 = data_aug_1.edge_index.numpy()
                _, edge_num_1 = edge_idx_1.shape
                edge_idx_2 = data_aug_2.edge_index.numpy()
                _, edge_num_2 = edge_idx_2.shape

                node_num, _ = data_o.x.size()

                idx_not_missing_1 = [n for n in range(node_num) if (
                        n in edge_idx_1[0] or n in edge_idx_1[1])]
                idx_not_missing_2 = [n for n in range(node_num) if (
                        n in edge_idx_2[0] or n in edge_idx_2[1])]
                node_num_aug_1 = len(idx_not_missing_1)
                node_num_aug_2 = len(idx_not_missing_2)
                data_aug_1.x = data_o.x[idx_not_missing_1]
                data_aug_2.x = data_o.x[idx_not_missing_2]

                data_aug_1.batch = data_o.batch[idx_not_missing_1]
                data_aug_2.batch = data_o.batch[idx_not_missing_2]
                idx_dict_1 = {idx_not_missing_1[n]: n for n in range(
                    node_num_aug_1)}
                idx_dict_2 = {idx_not_missing_2[n]: n for n in range(
                    node_num_aug_2)}
                edge_idx_1 = [[idx_dict_1[edge_idx_1[0, n]], idx_dict_1[edge_idx_1[1, n]]]
                              for n in range(edge_num_1) if not edge_idx_1[0, n] == edge_idx_1[1, n]]

                edge_idx_2 = [[idx_dict_2[edge_idx_2[0, n]], idx_dict_2[edge_idx_2[1, n]]]
                              for n in range(edge_num_2) if not edge_idx_2[0, n] == edge_idx_2[1, n]]

                if len(edge_idx_1) != 0:
                    data_aug_1.edge_index = torch.tensor(
                        edge_idx_1).transpose_(0, 1)
                if len(edge_idx_2) != 0:
                    data_aug_2.edge_index = torch.tensor(
                        edge_idx_2).transpose_(0, 1)

            data_aug_1 = data_aug_1.to(device)
            data_aug_2 = data_aug_2.to(device)

            optimizer.zero_grad()
            out_aug_1 = model(data_aug_1.x, data_aug_1.edge_index, data_aug_1.edge_attr, data_aug_1.batch)
            out_aug_2 = model(data_aug_2.x, data_aug_2.edge_index, data_aug_2.edge_attr, data_aug_2.batch)

            loss = model.loss_cal(out_aug_1, out_aug_2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += 1

        total_loss = total_loss / total
        logger.info(
            'Epoch: %3d/%3d, Train Loss: %.5f, Time: %.5f' % (epoch + 1, epochs, total_loss, time.time() - start_time))

    train_time = int(time.time() - start_time)

    save_model_path = output_path + '/{}'.format(pooling_type) + ".model"
    logger.info(f"save model to {save_model_path}")
    torch.save(model.state_dict(), save_model_path)
    logger.info(f'train_time: {train_time}')
    result_data = {}
    if os.path.exists('../results.json'):
        with open('../results.json', 'r') as f:
            result_data = json.load(f)
    dataset_results = result_data.setdefault(dataset_name, {})
    algorithm_results = dataset_results.setdefault('TraceCRL', {'total': {}, 'structure': {}, 'latency': {}})
    for key in ['total', 'structure', 'latency']:
        algorithm_results[key]['time'] = train_time
    with open('../results.json', 'w') as f:
        json.dump(result_data, f, indent=4)
