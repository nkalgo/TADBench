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
from evaluate import evaluate
from model import SIMCLR
from aug_dataset import TraceDataset, TestDataset


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.multiprocessing.set_sharing_strategy('file_system')


def tracecrl_test(dataset_name):
    epochs = 1
    batch_size = 32  # 32
    num_workers = 6  # 6
    num_layers = 2
    pooling_type = 'mean'  # mean  add
    gnn_type = 'CGConv'
    aug = 'random'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    fit_dataset = TraceDataset(root='data/%s/train' % dataset_name, aug=aug, fit=True)  # 40000
    test_dataset = TestDataset(root='data/%s/test' % dataset_name)
    fit_loader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    trace_classifier = OneClassSVM(nu=0.0001, kernel='rbf')
    model = SIMCLR(num_layers=num_layers, input_dim=fit_dataset.num_node_features,
                   output_dim=fit_dataset.num_node_features, num_edge_attr=fit_dataset.num_edge_features,
                   pooling_type=pooling_type).to(device)
    output_path = f"results/" + dataset_name + '_{}epoch_{}_{}_{}layers_{}batch_{}'.format(epochs, gnn_type,
                                                                                           pooling_type, num_layers,
                                                                                           batch_size, aug)
    save_model_path = output_path + '/{}'.format(pooling_type) + ".model"
    model.load_state_dict(torch.load(save_model_path, map_location=device))
    model.eval()

    trace_encodes = []
    with torch.no_grad():
        for data in tqdm(fit_loader):
            # data_o, data_aug_1, data_aug_2 = data
            data_o, _, _ = data
            data_o = data_o.to(device)
            pred_1 = model(data_o.x, data_o.edge_index, data_o.edge_attr, data_o.batch)
            trace_encodes.append(pred_1.detach().cpu().numpy())

    trace_encodes = np.concatenate(trace_encodes, axis=0)
    # Fit classifier
    trace_classifier.fit(trace_encodes)
    model.train()
    evaluate(test_loader, model, dataset_name, trace_classifier, device)



