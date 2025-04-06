import torch
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.data.data import Data
from tqdm import tqdm
import json
import os
import os.path as osp
import pandas as pd
from typing import List, Tuple, Union
from copy import deepcopy
from aug_method import *


class TraceDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, aug=None, max_cnt=None, fit=False):
        super(TraceDataset, self).__init__(
            root, transform, pre_transform)
        self.aug = aug
        self.max_cnt = max_cnt
        self.fit = fit
        # Get length
        if self.max_cnt is None:
            self._len = len(self.processed_file_names)
        else:
            self._len = self.max_cnt

        if max_cnt is not None:
            self.sample_list = np.random.choice(self._len, size=max_cnt, replace=False)

    @property
    def kpi_features(self):  # ptd
        return ['duration']

    @property
    def span_features(self):
        return ['ppt', 'qst', 'qet', 'sle', 'ple', 'rsc']

    @property
    def edge_features(self):
        return self.kpi_features + self.span_features

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        file_list = []
        file_list.append(osp.join(self.root, 'preprocessed/train.json'))
        return file_list

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        file_list = []
        for file in os.listdir(self.processed_dir):
            if os.path.splitext(file)[1] == '.pt':
                if file in ['pre_filter.pt', 'pre_transform.pt']:
                    continue
                file_list.append(file)

        return file_list

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, name)

    @property
    def normal_idx(self):
        with open(self.processed_dir + '/data_info.json', 'r') as f:  # file name not list
            data_info = json.load(f)
            normal_idx = data_info['normal']
        return normal_idx

    @property
    def abnormal_idx(self):
        with open(self.processed_dir + '/data_info.json', 'r') as f:  # file name not list
            data_info = json.load(f)
            abnormal_idx = data_info['abnormal']
        return abnormal_idx

    @property
    def trace_classes(self):
        with open(self.processed_dir + '/data_info.json', 'r') as f:  # file name not list
            data_info = json.load(f)
            trace_classes = data_info['trace_classes']
        return trace_classes

    @property
    def url_status_classes(self):
        with open(self.processed_dir + '/data_info.json', 'r') as f:  # file name not list
            data_info = json.load(f)
            url_status_classes = data_info['url_status_classes']
        return url_status_classes

    @property
    def url_classes(self):
        with open(self.processed_dir + '/data_info.json', 'r') as f:  # file name not list
            data_info = json.load(f)
            url_classes = data_info['url_classes']
        return url_classes

    def download(self):
        pass

    def process(self):
        idx = 0
        normal_idx = []
        abnormal_idx = []
        class_list = []
        url_status_class_list = []
        url_class_list = []
        num_features_stat = self._get_num_features_stat()
        operation_embedding = self._operation_embedding()

        print('load preprocessed data file:', self.raw_file_names[0])
        with open(self.raw_file_names[0], 'r') as f:  # file name not list
            raw_data = json.load(f)

        for trace_id, trace in tqdm(raw_data.items()):
            node_feats = self._get_node_features(trace, operation_embedding)  # operation_name_list
            edge_feats, edge_feats_stat = self._get_edge_features(trace, num_features_stat)
            edge_index = self._get_adjacency_info(trace)

            num_edges_edge_feats, _ = edge_feats.size()
            _, num_edges_edge_index = edge_index.size()
            if num_edges_edge_feats != num_edges_edge_index:
                print('Feature dismatch! num_edges_edge_feats: {}, num_edges_edge_index: {}, trace_id: {}'.format(
                    num_edges_edge_feats, num_edges_edge_index, trace_id))

            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_feats,
                trace_id=trace_id,
                time_stamp=trace['edges']['0'][0]['startTime'],
                y=torch.tensor(np.asarray([trace['abnormal']]), dtype=torch.int64),
                root_url=trace['edges']['0'][0]['operation'],
                edge_attr_stat=edge_feats_stat
            )

            if trace['abnormal'] == [0, 0]:
                normal_idx.append(idx)
            else:
                abnormal_idx.append(idx)

            filename = osp.join(self.processed_dir, 'data_{}.pt'.format(idx))
            torch.save(data, filename)
            idx += 1

        datainfo = {'normal': normal_idx,
                    'abnormal': abnormal_idx,
                    'trace_classes': class_list,
                    'url_status_classes': url_status_class_list,
                    'url_classes': url_class_list}

        with open(self.processed_dir + '/data_info.json', 'w', encoding='utf-8') as json_file:
            json.dump(datainfo, json_file)
            print('write data info success')

    def __len__(self):
        return self._len

    def _operation_embedding(self):
        """
        get operation embedding
        """
        with open(self.root + '/preprocessed/embeddings.json', 'r') as f:
            operations_embedding = json.load(f)

        return operations_embedding

    def _get_num_features_stat(self):
        """
        calculate features stat
        """
        operations_stat_map = {}
        with open(self.root + '/preprocessed/operations.json', 'r') as f:
            operations_info = json.load(f)

        for key in operations_info.keys():
            stat_map = {}
            for feature in self.kpi_features:
                ops = operations_info[key][feature]
                ops_mean = np.mean(ops)
                ops_std = np.std(ops)
                stat_map[feature] = [ops_mean, ops_std]
            operations_stat_map[key] = stat_map

        return operations_stat_map

    def _get_node_features(self, trace, operation_embedding):
        """
        node features matrix
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        node_feats = []
        for span_id, attr in trace['vertexs'].items():
            if span_id == '0':
                node_feats.append(operation_embedding[attr])
            else:
                node_feats.append(operation_embedding[attr[1]])

        node_feats = np.asarray(node_feats)
        return torch.tensor(node_feats, dtype=torch.float)

    def _get_edge_features(self, trace, num_features_stat):
        """
        edge features matrix
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        edge_feats = []
        edge_feats_stat = []
        min_startTime = trace['edges']['0'][0]['startTime']
        max_endTime = trace['edges']['0'][0]['startTime'] + trace['edges']['0'][0]['duration']
        for from_id, to_list in trace['edges'].items():
            for to in to_list:
                feat = []
                feat_stat = []
                if from_id == '0':
                    api_pair = 'root--->' + '%s/%s' % (
                        trace['vertexs'][str(to['vertexId'])][0], trace['vertexs'][str(to['vertexId'])][1])
                    # api_pair = 'root--->' + trace['vertexs'][str(to['vertexId'])][1].replace(
                    #     trace['vertexs'][str(to['vertexId'])][0] + '/', '')
                else:
                    api_pair = ('%s/%s' % (trace['vertexs'][from_id][0], trace['vertexs'][from_id][1]) + '--->' +
                                '%s/%s' % (
                                    trace['vertexs'][str(to['vertexId'])][0], trace['vertexs'][str(to['vertexId'])][1]))
                    # api_pair = trace['vertexs'][from_id][1].replace(
                    #     trace['vertexs'][from_id][0] + '/', '') + '--->' + trace['vertexs'][str(to['vertexId'])][
                    #                1].replace(
                    #     trace['vertexs'][str(to['vertexId'])][0] + '/', '')
                for feature in self.kpi_features:
                    feature_num = self._z_score(to[feature], num_features_stat[api_pair][feature])
                    feat.append(feature_num)
                    feat_stat.append(num_features_stat[api_pair][feature][0])
                    feat_stat.append(num_features_stat[api_pair][feature][1])
                """ def span_features(self):
                        return ['ppt', 'qst', 'qet', 'sle', 'ple', 'rsc']"""
                for feature in self.span_features:
                    if feature == 'ppt':
                        if max_endTime - min_startTime == 0:
                            feat.append(1)
                        else:
                            feat.append(to['duration'] / (max_endTime - min_startTime))
                    elif feature == 'qst':
                        if max_endTime - min_startTime == 0:
                            feat.append(1)
                        else:
                            feat.append((to['startTime'] - min_startTime) / (max_endTime - min_startTime))
                    elif feature == 'qet':
                        if max_endTime - min_startTime == 0:
                            feat.append(1)
                        else:
                            feat.append((to['startTime'] + to['duration'] - min_startTime) / (max_endTime - min_startTime))
                    elif feature == 'sle':
                        caller_id = str(to['vertexId'])
                        if caller_id in trace['edges'].keys():
                            feat.append(len(trace['edges'][caller_id]) + 1)
                        else:
                            feat.append(1)
                    elif feature == 'ple':
                        caller_id = str(to['vertexId'])
                        children_durations = 0
                        if caller_id in trace['edges'].keys():
                            for child in trace['edges'][caller_id]:
                                children_durations += child['duration']
                        if to['duration'] != 0:
                            feat.append((to['duration'] - children_durations) / to['duration'])
                        else:
                            feat.append(1.0)
                    elif feature == 'rsc':
                        status_code = int(to['statusCode'])
                        if 100 <= status_code <= 199:
                            feat.extend([1, 0, 0, 0, 0])
                        elif 200 <= status_code <= 299:
                            feat.extend([0, 1, 0, 0, 0])
                        elif 300 <= status_code <= 399:
                            feat.extend([0, 0, 1, 0, 0])
                        elif 400 <= status_code <= 499:
                            feat.extend([0, 0, 0, 1, 0])
                        elif 500 <= status_code <= 599:
                            feat.extend([0, 0, 0, 0, 1])
                        else:
                            feat.extend([0, 0, 0, 0, 0])

                edge_feats.append(feat)
                edge_feats_stat.append(feat_stat)
        edge_feats_stat = np.asarray(edge_feats_stat)
        edge_feats = np.asarray(edge_feats)
        return torch.tensor(edge_feats, dtype=torch.float), torch.tensor(edge_feats_stat, dtype=torch.float)

    def _get_adjacency_info(self, trace):
        """
        adjacency list
        [from1, from2, from3 ...] [to1, to2, to3 ...]
        """
        adj_list = [[], []]
        for from_id, to_list in trace['edges'].items():
            for to in to_list:
                to_id = to['vertexId']
                adj_list[0].append(int(from_id))
                adj_list[1].append(int(to_id))

        return torch.tensor(adj_list, dtype=torch.long)

    def _get_node_labels(self, trace):
        """
        node label
        """
        pass

    def _z_score(self, raw, feature_stat):
        """
        calculate z-score
        """
        if feature_stat[1] == 0:
            z_score = abs(raw - feature_stat[0]) / 1
        else:
            z_score = abs(raw - feature_stat[0]) / feature_stat[1]
        return z_score

    _dispatcher = {}

    def get(self, idx: int):
        if self.max_cnt is not None:
            idx = self.sample_list[idx]

        data = torch.load(
            osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))

        if not self.fit:
            if np.random.random() < 0.5:
                data = self._get_data_aug(deepcopy(data))

        if self.aug == 'permute_edges_for_subgraph':
            data_aug_1 = permute_edges_for_subgraph(deepcopy(data))
            data_aug_2 = permute_edges_for_subgraph(deepcopy(data))
        elif self.aug == 'mask_nodes':
            data_aug_1 = mask_nodes(deepcopy(data))
            data_aug_2 = mask_nodes(deepcopy(data))
        elif self.aug == 'mask_edges':
            data_aug_1 = mask_edges(deepcopy(data))
            data_aug_2 = mask_edges(deepcopy(data))
        elif self.aug == 'mask_nodes_and_edges':
            data_aug_1 = mask_nodes(deepcopy(data))
            data_aug_1 = mask_edges(data_aug_1)
            data_aug_2 = mask_nodes(deepcopy(data))
            data_aug_2 = mask_edges(data_aug_2)
        elif self.aug == 'subgraph':
            data_aug_1 = subgraph(deepcopy(data))
            data_aug_2 = subgraph(deepcopy(data))
        elif self.aug == 'none':
            return data
        elif self.aug == 'random':
            data_aug_1 = self._get_view_aug(data)
            data_aug_2 = self._get_view_aug(data)
        else:
            print('no need for augmentation')
            assert False

        return data, data_aug_1, data_aug_2

    def len(self) -> int:

        return len(self.processed_file_names)

    def _get_view_aug(self, data):
        n = np.random.randint(5)
        if n == 0:
            data_aug = mask_nodes(deepcopy(data))
        elif n == 1:
            data_aug = mask_edges(deepcopy(data))
        elif n == 2:
            data_aug = mask_nodes(deepcopy(data))
            data_aug = mask_edges(data_aug)
        elif n == 3:
            data_aug = permute_edges_for_subgraph(deepcopy(data))
        elif n == 4:
            data_aug = subgraph(deepcopy(data))
        else:
            print('sample error')
            assert False
        return data_aug

    def _get_data_aug(self, data):
        n = np.random.randint(5)
        if n == 0:
            data_aug = status_change(deepcopy(data))
        elif n == 1:
            data_aug = span_order_error_injection(deepcopy(data))
        elif n == 2:
            data_aug = add_nodes(deepcopy(data))
        elif n == 3:
            data_aug = drop_several_nodes(deepcopy(data))
        elif n == 4:
            data_aug = time_error_injection(deepcopy(data), 'duration', self.edge_features)
        else:
            print('sample error')
            assert False
        return data_aug


class TestDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, max_cnt=None):
        super(TestDataset, self).__init__(
            root, transform, pre_transform)
        self.max_cnt = max_cnt
        if self.max_cnt is None:
            self._len = len(self.processed_file_names)
        else:
            self._len = self.max_cnt

        if max_cnt is not None:
            self.sample_list = np.random.choice(self._len, size=max_cnt, replace=False)

    @property
    def kpi_features(self):
        return ['duration']

    @property
    def span_features(self):
        return ['ppt', 'qst', 'qet', 'sle', 'ple', 'rsc']

    @property
    def edge_features(self):
        return self.kpi_features + self.span_features

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        file_list = []
        file_list.append(osp.join(self.root, 'preprocessed/test.json'))
        return file_list

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        file_list = []
        for file in os.listdir(self.processed_dir):
            if os.path.splitext(file)[1] == '.pt':
                if file in ['pre_filter.pt', 'pre_transform.pt']:
                    continue
                file_list.append(file)

        return file_list

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, name)

    @property
    def normal_idx(self):
        with open(self.processed_dir + '/data_info.json', 'r') as f:  # file name not list
            data_info = json.load(f)
            normal_idx = data_info['normal']
        return normal_idx

    @property
    def structure_abnormal_idx(self):
        with open(self.processed_dir + '/data_info.json', 'r') as f:  # file name not list
            data_info = json.load(f)
            structure_abnormal_idx = data_info['structure_abnormal']
        return structure_abnormal_idx

    @property
    def latency_abnormal_idx(self):
        with open(self.processed_dir + '/data_info.json', 'r') as f:  # file name not list
            data_info = json.load(f)
            latency_abnormal_idx = data_info['latency_abnormal']
        return latency_abnormal_idx

    @property
    def trace_classes(self):
        with open(self.processed_dir + '/data_info.json', 'r') as f:  # file name not list
            data_info = json.load(f)
            trace_classes = data_info['trace_classes']
        return trace_classes

    @property
    def url_status_classes(self):
        with open(self.processed_dir + '/data_info.json', 'r') as f:  # file name not list
            data_info = json.load(f)
            url_status_classes = data_info['url_status_classes']
        return url_status_classes

    @property
    def url_classes(self):
        with open(self.processed_dir + '/data_info.json', 'r') as f:  # file name not list
            data_info = json.load(f)
            url_classes = data_info['url_classes']
        return url_classes

    def download(self):
        pass

    def process(self):
        idx = 0
        normal_idx = []
        structure_abnormal_idx = []
        latency_abnormal_idx = []
        class_list = []
        url_status_class_list = []
        url_class_list = []
        num_features_stat = self._get_num_features_stat()
        operation_embedding = self._operation_embedding()

        print('load preprocessed data file:', self.raw_file_names[0])
        with open(self.raw_file_names[0], 'r') as f:  # file name not list
            raw_data = json.load(f)

        for trace_id, trace in tqdm(raw_data.items()):
            node_feats = self._get_node_features(trace, operation_embedding)
            edge_feats, edge_feats_stat = self._get_edge_features(trace, num_features_stat)
            edge_index = self._get_adjacency_info(trace)
            nodeLatencyLabels, traceLabel = self._get_anomaly_labels(trace)

            num_edges_edge_feats, _ = edge_feats.size()
            _, num_edges_edge_index = edge_index.size()
            if num_edges_edge_feats != num_edges_edge_index:
                print('Feature dismatch! num_edges_edge_feats: {}, num_edges_edge_index: {}, trace_id: {}'.format(
                    num_edges_edge_feats, num_edges_edge_index, trace_id))

            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_feats,
                trace_id=trace_id,
                time_stamp=trace['edges']['0'][0]['startTime'],
                y=traceLabel,
                node_latency_labels=nodeLatencyLabels,
                root_url=trace['edges']['0'][0]['operation'],
                edge_attr_stat=edge_feats_stat
            )

            if trace['abnormal'] == [0, 0]:
                normal_idx.append(idx)
            elif trace['abnormal'] == [1, 1]:
                structure_abnormal_idx.append(idx)
                latency_abnormal_idx.append(idx)
            elif trace['abnormal'][0] == 1:
                structure_abnormal_idx.append(idx)
            elif trace['abnormal'][1] == 1:
                latency_abnormal_idx.append(idx)

            filename = osp.join(self.processed_dir, 'data_{}.pt'.format(idx))
            torch.save(data, filename)
            idx += 1

        datainfo = {'normal': normal_idx,
                    'structure_abnormal': structure_abnormal_idx,
                    'latency_abnormal': latency_abnormal_idx,
                    'trace_classes': class_list,
                    'url_status_classes': url_status_class_list,
                    'url_classes': url_class_list}

        with open(self.processed_dir + '/data_info.json', 'w', encoding='utf-8') as json_file:
            json.dump(datainfo, json_file)
            print('write data info success')

    def __len__(self):
        return self._len

    def _operation_embedding(self):
        """
        get operation embedding
        """
        with open(self.root + '/preprocessed/embeddings.json', 'r') as f:
            operations_embedding = json.load(f)

        return operations_embedding

    def _get_num_features_stat(self):
        """
        calculate features stat
        """
        operations_stat_map = {}
        with open(self.root + '/preprocessed/operations.json', 'r') as f:
            operations_info = json.load(f)

        for key in operations_info.keys():
            stat_map = {}
            for feature in self.kpi_features:
                ops = operations_info[key][feature]
                ops_mean = np.mean(ops)
                ops_std = np.std(ops)
                stat_map[feature] = [ops_mean, ops_std]
            operations_stat_map[key] = stat_map

        return operations_stat_map

    def _get_node_features(self, trace, operation_embedding):
        """
        node features matrix
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        node_feats = []
        for span_id, attr in trace['vertexs'].items():
            if span_id == '0':
                node_feats.append(operation_embedding[attr])
            else:
                node_feats.append(operation_embedding[attr[1]])

        node_feats = np.asarray(node_feats)
        return torch.tensor(node_feats, dtype=torch.float)

    def _get_edge_features(self, trace, num_features_stat):
        """
        edge features matrix
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        edge_feats = []
        edge_feats_stat = []
        min_startTime = trace['edges']['0'][0]['startTime']
        max_endTime = trace['edges']['0'][0]['startTime'] + trace['edges']['0'][0]['duration']
        for from_id, to_list in trace['edges'].items():
            for to in to_list:
                feat = []
                feat_stat = []
                if from_id == '0':
                    api_pair = 'root--->' + '%s/%s' % (
                        trace['vertexs'][str(to['vertexId'])][0], trace['vertexs'][str(to['vertexId'])][1])
                    # api_pair = 'root--->' + trace['vertexs'][str(to['vertexId'])][1].replace(
                    #     trace['vertexs'][str(to['vertexId'])][0] + '/', '')
                else:
                    api_pair = ('%s/%s' % (trace['vertexs'][from_id][0], trace['vertexs'][from_id][1]) + '--->' +
                                '%s/%s' % (
                                    trace['vertexs'][str(to['vertexId'])][0], trace['vertexs'][str(to['vertexId'])][1]))
                    # api_pair = trace['vertexs'][from_id][1].replace(
                    #     trace['vertexs'][from_id][0] + '/', '') + '--->' + trace['vertexs'][str(to['vertexId'])][
                    #                1].replace(
                    #     trace['vertexs'][str(to['vertexId'])][0] + '/', '')
                for feature in self.kpi_features:
                    feature_num = self._z_score(to[feature], num_features_stat[api_pair][feature])
                    feat.append(feature_num)
                    feat_stat.append(num_features_stat[api_pair][feature][0])
                    feat_stat.append(num_features_stat[api_pair][feature][1])

                for feature in self.span_features:
                    if feature == 'ppt':
                        if max_endTime - min_startTime == 0:
                            feat.append(1)
                        else:
                            feat.append(to['duration'] / (max_endTime - min_startTime))
                    elif feature == 'qst':
                        if max_endTime - min_startTime == 0:
                            feat.append(1)
                        else:
                            feat.append((to['startTime'] - min_startTime) / (max_endTime - min_startTime))
                    elif feature == 'qet':
                        if max_endTime - min_startTime == 0:
                            feat.append(1)
                        else:
                            feat.append(
                                (to['startTime'] + to['duration'] - min_startTime) / (max_endTime - min_startTime))
                    elif feature == 'sle':
                        caller_id = str(to['vertexId'])
                        if caller_id in trace['edges'].keys():
                            feat.append(len(trace['edges'][caller_id]) + 1)
                        else:
                            feat.append(1)
                    elif feature == 'ple':
                        caller_id = str(to['vertexId'])
                        children_durations = 0
                        if caller_id in trace['edges'].keys():
                            for child in trace['edges'][caller_id]:
                                children_durations += child['duration']
                        if to['duration'] != 0:
                            feat.append((to['duration'] - children_durations) / to['duration'])
                        else:
                            feat.append(1.0)
                    elif feature == 'rsc':
                        status_code = int(to['statusCode'])
                        if 100 <= status_code <= 199:
                            feat.extend([1, 0, 0, 0, 0])
                        elif 200 <= status_code <= 299:
                            feat.extend([0, 1, 0, 0, 0])
                        elif 300 <= status_code <= 399:
                            feat.extend([0, 0, 1, 0, 0])
                        elif 400 <= status_code <= 499:
                            feat.extend([0, 0, 0, 1, 0])
                        elif 500 <= status_code <= 599:
                            feat.extend([0, 0, 0, 0, 1])
                        else:
                            feat.extend([0, 0, 0, 0, 0])

                edge_feats.append(feat)
                edge_feats_stat.append(feat_stat)
        edge_feats_stat = np.asarray(edge_feats_stat)
        edge_feats = np.asarray(edge_feats)
        return torch.tensor(edge_feats, dtype=torch.float), torch.tensor(edge_feats_stat, dtype=torch.float)

    def _get_anomaly_labels(self, trace):
        nodeLatencyLabels = []
        for span_id, attr in trace['vertexs'].items():
            if span_id == '0':
                nodeLatencyLabels.append(0)
            else:
                nodeLatencyLabels.append(attr[2])
        nodeLatencyLabels = np.asarray(nodeLatencyLabels)

        traceLabel = np.asarray([trace['abnormal']])

        return torch.tensor(nodeLatencyLabels, dtype=torch.int64), torch.tensor(traceLabel, dtype=torch.int64)

    def _get_adjacency_info(self, trace):
        """
        adjacency list
        [from1, from2, from3 ...] [to1, to2, to3 ...]
        """
        adj_list = [[], []]
        for from_id, to_list in trace['edges'].items():
            for to in to_list:
                to_id = to['vertexId']
                adj_list[0].append(int(from_id))
                adj_list[1].append(int(to_id))

        return torch.tensor(adj_list, dtype=torch.long)

    def _get_node_labels(self, trace):
        """
        node label
        """
        pass

    def _z_score(self, raw, feature_stat):
        """
        calculate z-score
        """
        if feature_stat[1] == 0:
            z_score = abs(raw - feature_stat[0]) / 1
        else:
            z_score = abs(raw - feature_stat[0]) / feature_stat[1]
        return z_score

    _dispatcher = {}

    def get(self, idx: int):
        if self.max_cnt is not None:
            idx = self.sample_list[idx]

        data = torch.load(
            osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        
        return data

    def len(self) -> int:
        return len(self.processed_file_names)



