import json
import random
from typing import List, Tuple, Union
import torch.backends.cudnn as cudnn
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import Dataset
from utils.time_wrapper import time_used


class MyTraceDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, type="train", anomaly_type="total"):

        self.type = type
        self.anomaly_type = anomaly_type

        super(MyTraceDataset, self).__init__(
            root, transform, pre_transform)


        if type == "train":
            print(f"{type} using {self.processed_paths[0]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif type == "validate":
            print(f"{type} using {self.processed_paths[1]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif type == "test":
            print(f"{type}_{anomaly_type}  using {self.processed_paths[2]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ["train.json", "validate.json", "test_%s.json" % self.anomaly_type]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ["train.pt", "validate.pt", "test_%s.pt" % self.anomaly_type]

    def download(self):
        pass

    def process(self):
        idx = 0
        operation_embedding = self._operation_embedding()

        file_paths = self.raw_paths

        train_path = file_paths[0]
        train_datalist = []
        print(f'train_path={train_path}')
        with open(train_path, "r") as f:
            for line in f:
                if idx % 10000 == 0:
                    print(f"{idx=}", end=', ')
                idx += 1

                trace = json.loads(line)
                node_feats = self._get_node_features(trace, operation_embedding)
                node_feats = torch.tensor(node_feats, dtype=torch.float)

                edge_index = trace.get("edges")
                edge_index = torch.tensor(edge_index, dtype=torch.long)

                data = Data(
                    x=node_feats,
                    edge_index=edge_index,
                    trace_id=trace["trace_id"],  # add trace_id for cluster
                    y=trace['abnormal'],
                )
                train_datalist.append(data)
        print('\n')
        print("collating train data")
        data, slices = self.collate(train_datalist)
        torch.save((data, slices), self.processed_paths[0])

        validate_path = file_paths[1]
        validate_datalist = []
        with open(validate_path, "r") as f:
            for line in f:
                if idx % 10000 == 0:
                    print(f"{idx=}", end=', ')
                idx += 1

                trace = json.loads(line)
                node_feats = self._get_node_features(trace, operation_embedding)
                node_feats = torch.tensor(node_feats, dtype=torch.float)

                edge_index = trace.get("edges")
                edge_index = torch.tensor(edge_index, dtype=torch.long)

                data = Data(
                    x=node_feats,
                    edge_index=edge_index,
                    trace_id=trace["trace_id"],  # add trace_id for cluster
                    y=trace['abnormal'],
                )
                validate_datalist.append(data)
        print('\n')
        print("collating validate data")
        data, slices = self.collate(validate_datalist)
        torch.save((data, slices), self.processed_paths[1])

        test_path = file_paths[2]
        test_datalist = []
        with open(test_path, "r") as f:
            for line in f:
                if idx % 10000 == 0:
                    print(f"{idx=}", end=', ')
                idx += 1
                trace = json.loads(line)
                node_feats = self._get_node_features(trace, operation_embedding)
                node_feats = torch.tensor(node_feats, dtype=torch.float)

                edge_index = trace.get("edges")
                edge_index = torch.tensor(edge_index, dtype=torch.long)

                data = Data(
                    x=node_feats,
                    edge_index=edge_index,
                    trace_id=trace["trace_id"],  # add trace_id for cluster
                    y=trace['abnormal'],
                )
                test_datalist.append(data)
        print('\n')
        print("collating test data")
        data, slices = self.collate(test_datalist)
        torch.save((data, slices), self.processed_paths[2])

    def _operation_embedding(self):
        """
        get operation embedding
        """
        with open(self.root + '/preprocessed/bert_embeddings.json', 'r') as f:
            operations_embedding = json.load(f)

        return operations_embedding

    def _get_node_features(self, trace, operation_embedding):
        """
        node features matrix
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        node_feats = []

        trace_start_timestamp = trace["nodes"][0].get("startTime")
        for span in trace["nodes"]:
            operation = span['operation']
            svc_op = span['service'] + "/" + operation
            # DeepWalk: 20维
            # BERT: 768维
            op_embed = operation_embedding[svc_op]
            # print(len(op_embed))
            op_embed_tensor = torch.tensor(op_embed, dtype=torch.float)

            # 1维 * 4
            span_all_time = span['rawDuration']
            span_all_time_tensor = torch.FloatTensor([span_all_time])

            span_local_time = span['workDuration']
            span_local_time_tensor = torch.FloatTensor([span_local_time])

            span_wait_time = span['rawDuration'] - span['workDuration']
            span_wait_time_tensor = torch.FloatTensor([span_wait_time])

            span_start_tamp = span['startTime'] - trace_start_timestamp
            span_start_tamp_tensor = torch.FloatTensor([span_start_tamp])

            # 5维
            status_code = span['statusCode']
            status_code_tensor = self.embedding_statuscode(status_code)

            span_tensor = torch.cat(
                [op_embed_tensor, span_all_time_tensor, span_local_time_tensor, span_wait_time_tensor,
                 span_start_tamp_tensor, status_code_tensor])
            node_feats.append(span_tensor)

        node_feats = [t.numpy() for t in node_feats]
        return node_feats

    def embedding_statuscode(self, code):
        if isinstance(code, str) or code is None or code == 0:
            code = 200
        else:
            code = int(code)
        status_codes = [
            # 1xx (Informational)
            100, 101, 102, 103,
            # 2xx (Successful)
            200, 201, 202, 203, 204, 205, 206, 207, 208, 226,
            # 3xx (Redirection)
            300, 301, 302, 303, 304, 305, 306, 307, 308,
            # 4xx (Client Error)
            400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410,
            411, 412, 413, 414, 415, 416, 417, 418, 421, 422, 423,
            424, 425, 426, 428, 429, 431, 451,
            # 5xx (Server Error)
            500, 501, 502, 503, 504, 505, 506, 507, 508, 510, 511
        ]
        status_code_dict = {code: index for index, code in enumerate(status_codes)}
        index = status_code_dict[code]
        res = [0 if i is not index else 1 for i in range(len(status_codes))]
        return torch.tensor(res, dtype=torch.int)


def get_my_trace_graph_datas(type, root="./my_trace_data", seed=None, anomaly_type="total"):
    if seed:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    dataset = MyTraceDataset(root=root, type=type, anomaly_type=anomaly_type)

    dataset_ds = []
    dataset_ys = []
    for d in dataset:
        dataset_ds.append(d)
        dataset_ys.append(int(d.y))

    return dataset_ds, dataset_ys


def make_dataset(datasetX, datasetY, n_labeled, n_unlabeled, mode="train", pn=False, seed=None):

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def make_PU_dataset_from_binary_dataset(x, y, labeled=n_labeled, unlabeled=n_unlabeled):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        print(f"{positive=} {negative=}")

        # X, Y = np.asarray(x), np.asarray(y, dtype=np.int32)
        X, Y = x, np.asarray(y, dtype=np.int32)

        # perm = np.random.permutation(len(X))
        #
        # # X, Y = X[perm], Y[perm]
        # Y = Y[perm]
        # X = [X[idx] for idx in perm]

        assert (len(X) == len(Y))

        n_p = (Y == positive).sum()
        n_lp = labeled
        n_u = unlabeled

        if labeled + unlabeled == len(X):
            n_up = n_p - n_lp
        elif unlabeled == len(X):
            n_up = n_p
        else:
            print(f"{labeled=} {unlabeled=} {len(X)=}")
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")

        # lp : label positive
        # up: unlabel positive
        # un: unlabel negative
        # u: unlabled
        # p: labeled
        print(f"{n_p=} {n_lp=} {n_u=} {n_up=}")

        prior = float(n_up) / float(n_u)

        yp_array = list(Y == positive)

        Xp = [X[i] for i in range(len(yp_array)) if yp_array[i]==True]
        Xlp = Xp[:n_lp]
        Xup = (Xp[n_lp:] + Xlp)[:n_up]
        Xun = [X[i] for i in range(len(yp_array)) if yp_array[i]==False]
        X = Xlp + Xup + Xun

        Y = np.asarray(np.concatenate((np.ones(n_lp), -np.ones(n_u))), dtype=np.int32)
        T = np.asarray(np.concatenate((np.ones(n_lp + n_up), -np.ones(n_u - n_up))), dtype=np.int32)

        # Generate ID
        ids = np.array([i for i in range(len(X))])
        return X, Y, T, ids, prior

    def make_PN_dataset_from_binary_dataset(x, y):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]

        X, Y = x, np.asarray(y, dtype=np.int32)

        # if seed is not None:
        #     np.random.seed(seed)
        # perm = np.random.permutation(len(X))

        # Y = Y[perm]
        # X = [X[idx] for idx in perm]

        n_p = (Y == positive).sum()
        n_n = (Y == negative).sum()
        yp_array = list(Y == positive)
        Xp = [X[i] for i in range(len(yp_array)) if yp_array[i]==True]
        Xn = [X[i] for i in range(len(yp_array)) if yp_array[i]==False]
        X = Xp + Xn

        print(f"{n_p=} {n_n=} {len(X)=}")

        Y = np.asarray(np.concatenate((np.ones(n_p), -np.ones(n_n))), dtype=np.int32)
        ids = np.array([i for i in range(len(X))])
        return X, Y, Y, ids

    def make_only_PN_train(x, y, n_labeled=n_labeled, prior=0.5):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = x, np.asarray(y, dtype=np.int32)

        # if seed is not None:
        #     np.random.seed(seed)
        # perm = np.random.permutation(len(X))

        # Y = Y[perm]
        # X = [X[idx] for idx in perm]

        assert (len(X) == len(Y))
        n_n = int(n_labeled * pow(prior / (2 * (1 - prior)), 2))
        print(f"{n_n=}")

        yp_array = list(Y == positive)
        Xp = [X[i] for i in range(len(yp_array)) if yp_array[i]==True]
        Xn = [X[i] for i in range(len(yp_array)) if yp_array[i]==False]
        X = Xp + Xn

        Y = np.asarray(np.concatenate((np.ones(n_labeled), -np.ones(n_n))), dtype=np.int32)
        ids = np.array([i for i in range(len(X))])

        return X, Y, Y, ids

    prior = None
    if mode == 'train':
        if not pn:
            X, Y, T, ids, prior = make_PU_dataset_from_binary_dataset(datasetX, datasetY)
        else:
            X, Y, T, ids = make_only_PN_train(datasetX, datasetY)
    else:
        X, Y, T, ids = make_PN_dataset_from_binary_dataset(datasetX, datasetY)
    return X, Y, T, ids, prior


class GraphDatasetFixSample(Dataset):

    def __init__(self, n_labeled, n_unlabeled, datasetX, datasetY, type="noisy", split="train", mode="A",
                 ids=None, pn=False, increasing=False, replacement=True, top=0.5, flex=0, pickout=True, seed=None):

        self.X, self.Y, self.T, self.oids, self.prior = make_dataset(datasetX, datasetY, n_labeled,
                                                                     n_unlabeled, mode=split, pn=pn, seed=seed)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            cudnn.deterministic = True
            cudnn.benchmark = False

        # print(f">>>\n{self.X=}\n{self.Y=}\n{self.T=}\n{self.oids=}\n{self.prior=}\n<<<")

        assert np.all(self.oids == np.linspace(0, len(self.X) - 1, len(self.X)))
        self.clean_ids = []
        # self.Y_origin = self.Y
        self.P = self.Y.copy()
        self.type = type
        if (ids is None):
            self.ids = self.oids
        else:
            self.ids = np.array(ids)

        self.split = split
        self.mode = mode
        # print(f"{self.Y=}\n{(self.Y == 1)=}")
        # print(f"{self.oids=}")

        self.pos_ids = self.oids[self.Y == 1]

        self.pid = self.pos_ids
        if len(self.ids) != 0:
            self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
        else:
            self.uid = []
        print(len(self.uid))
        print(len(self.pid))
        self.sample_ratio = len(self.uid) // len(self.pid) + 1
        print(f"{self.type=} {self.split=} {self.sample_ratio=}")
        print("origin:", len(self.pos_ids), len(self.ids))
        self.increasing = increasing
        self.replacement = replacement
        self.top = top  # 用top来决定抽取的比例，以此导致两个模型之间的pace不同
        self.flex = flex
        self.pickout = pickout

        self.pick_accuracy = []
        self.result = -np.ones(len(self))

        self.random_count = 0

    def copy(self, dataset):
        ''' Copy random sequence
        '''
        self.X, self.Y, self.T, self.oids = dataset.X.copy(), dataset.Y.copy(), dataset.T.copy(), dataset.oids.copy()
        self.P = self.Y.copy()

    def __len__(self):
        if self.type == 'noisy':

            # return len(self.uid) * 2
            return len(self.pid) * self.sample_ratio
        else:
            return len(self.ids)

    def set_type(self, type):
        self.type = type

    def update_prob(self, result):
        rank = np.empty_like(result)
        rank[np.argsort(result)] = np.linspace(0, 1, len(result))
        # print(rank)
        if (len(self.pos_ids) > 0):
            rank[self.pos_ids] = -1
        self.result = rank

    def shuffle(self):
        perm = np.random.permutation(len(self.uid))
        self.uid = self.uid[perm]

        perm = np.random.permutation(len(self.pid))
        self.pid = self.pid[perm]

    def __getitem__(self, idx):
        # print(idx)
        # self.ids[idx]是真实的行索引
        # 始终使用真实的行索引去获得数据

        # 1901 保持比例
        if self.type == 'noisy':
            '''
            if (idx % 2 == 0):
                index = self.pid[idx % 1000]
            else:
                index = self.uid[idx - (idx // 2 + 1)]

            '''
            if idx % self.sample_ratio == 0:
                index = self.pid[idx // self.sample_ratio]
                id = 0
            else:
                index = self.uid[idx - (idx // self.sample_ratio + 1)]

            # X, Y, _, T, ids, _
            return self.X[index], self.Y[index], self.P[index], self.T[index], index, 0
        else:
            return self.X[self.ids[idx]], self.Y[self.ids[idx]], self.P[self.ids[idx]], self.T[self.ids[idx]], self.ids[
                idx], 0

    def reset_ids(self):
        """
        Using all origin ids
        """
        self.ids = self.oids.copy()

    def set_ids(self, ids):
        """
        Set specific ids
        """
        self.ids = np.array(ids).copy()
        if len(ids) > 0:
            self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
            self.pid = np.intersect1d(self.ids[self.Y[self.ids] == 1], self.ids)
            if len(self.pid) == 0:
                self.sample_ratio = 10000000000
            else:
                self.sample_ratio = int(len(self.uid) / len(self.pid)) + 1

    def reset_labels(self):
        """
        Reset Y labels
        """
        self.P = self.Y.copy()

    @time_used
    def update_ids(self, results, epoch, ratio=None, select="sort", lt=0, ht=0):
        """
        select: "prob" 按照概率抽取  "sort"或None 排序后取top和neg
        """

        if not self.replacement or self.increasing:
            percent = min(epoch / 100, 1)  # 决定抽取数据的比例
        else:
            percent = 1
        if ratio == None:
            ratio = self.prior
        self.reset_labels()
        n_all = int((len(self.oids) - len(self.pos_ids)) * (1 - ratio) * percent * self.top)  # 决定抽取的数量
        confident_num = int(n_all * (1 - self.flex))
        noisy_num = int(n_all * self.flex)

        print(f"【UpdateIds】>>>>>>>>>> {confident_num=} {select=}")
        # print(f"【UpdateIds】{results=}\n{len(results)=}\n{min(results)=} {max(results)=}")
        # print(f"【UpdateIds】{self.ids=}\n{len(self.ids)=}")
        # print(f"【UpdateIds】{self.pos_ids=}\n{len(self.pos_ids)=}")
        # print(f"【UpdateIds】{self.oids=}\n{len(self.oids)=}")

        if self.replacement:
            # h10g: replacement本身就会将不confident的移除出去
            # 如果替换的话，抽取n_pos个
            # print(np.argsort(results))
            # print(np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True))

            if not select or select == "sort":
                al = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)
                neg_ids = al[:confident_num]
                pos_ids = al[-confident_num:]
            elif select == "wsort":
                al = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)
                all_num = 2*confident_num

                pos_num = all_num // 3
                neg_num = all_num - pos_num
                print(f"weighted_sort {pos_num=} {neg_num=}")

                pos_ids = al[-pos_num:]
                neg_ids = al[:neg_num]

            # 使用概率进行数据抽取，而不是直接排序
            elif select == "prob":
                # 分开选取pos和neg，保证数据集选取的均衡
                pos_confident_num = confident_num // 2
                neg_confident_num = confident_num - pos_confident_num

                myal = np.array([i for i in range(len(results))])
                myal_results = results[myal]

                # 去掉已经打标的self.pos_ids的数据计算min,max,mid,mean
                min_val = min(myal_results[self.pos_ids[-1]+1:])
                max_val = max(myal_results[self.pos_ids[-1]+1:])
                mid_val = np.median(myal_results[self.pos_ids[-1]+1:])
                mean_val = np.mean(myal_results[self.pos_ids[-1]+1:])
                # print(f"{min_val=} {max_val=} {mid_val=}")
                # print(f"{myal_results=}")

                # 给每个标签被选取的概率，为了拉开中间的和头尾的数据的差距，使用*10, +10, 再平方的方式进行了加权
                def get_result_prob(x):
                    prob = max(x - min_val, max_val-x)
                    prob *= 10
                    prob += 10
                    prob *= prob
                    return prob

                pos_ids = []
                neg_ids = []

                # 计算每个标签被抽取的概率
                choice_prob = np.array([get_result_prob(xi) for xi in myal_results])
                choice_prob = choice_prob.astype('float64')
                # 去掉已经打标的self.pos_ids的数据
                choice_prob[self.pos_ids] = 0
                choice_prob /= choice_prob.sum()

                # 为了避免无限循环，最多选10轮数据
                select_epoch = 0
                while select_epoch < 10 and len(pos_ids) + len(neg_ids) < confident_num:
                    select_epoch += 1
                    result_ids = np.random.choice(myal, 2*confident_num, p=choice_prob)
                    # print(f"{myal=}")
                    # print(f"{myal_results=}")
                    # print(f"{choice_prob=}")
                    # print(f"{min(choice_prob)=} {max(choice_prob)=} {np.median(choice_prob)=} {np.mean(choice_prob)=}")
                    # print(f"{result_ids=}")
                    for result_id in result_ids:
                        choice_prob[result_id] = 0
                        if results[result_id] >= mean_val and len(pos_ids) < pos_confident_num and result_id not in pos_ids:
                            pos_ids.append(result_id)
                            # 避免已经选取的id影响接下来的选取过程, 所以重新计算被选取的概率，将被选取过得数据的概率设为0
                            # choice_prob[result_id] = 0
                        elif results[result_id] < mean_val and len(neg_ids) < neg_confident_num and result_id not in neg_ids:
                            neg_ids.append(result_id)
                            # 避免已经选取的id影响接下来的选取过程, 所以重新计算被选取的概率，将被选取过得数据的概率设为0
                            # choice_prob[result_id] = 0
                    # 重新计算概率，避免加起来不等于1
                    choice_prob /= choice_prob.sum()
                # 得到最终数据
                pos_ids = np.array(pos_ids, dtype=int)
                neg_ids = np.array(neg_ids, dtype=int)
        else:
            # 否则抽取n_pos - #ids
            al = np.setdiff1d(np.argsort(results), self.ids, assume_unique=True)
            neg_ids = al[:(confident_num - len(self.ids) // 2)]
            pos_ids = al[-(confident_num - len(self.ids) // 2):]

        # 变成向量
        pos_ids = np.array(pos_ids)
        pos_label = self.T[pos_ids]  # 获得neg_ids的真实标签
        pcorrect = np.sum(pos_label == 1)  # 抽取N的时候真实标签为-1

        neg_ids = np.array(neg_ids)
        neg_label = self.T[neg_ids]  # 获得neg_ids的真实标签
        ncorrect = np.sum(neg_label < 1)

        self.P[pos_ids] = 1  # 将他们标注为1
        print("P Correct: {}/{}".format(pcorrect, len(pos_ids)))  # 打印
        print("N Correct: {}/{}".format(ncorrect, len(neg_ids)))

        self.pick_accuracy.append((pcorrect + ncorrect) * 1.0 / (len(pos_ids) * 2))
        if self.replacement:
            # self.ids = np.concatenate([self.pos_ids, pos_ids, neg_ids]) # 如果置换的话，在ids的基础上加上neg_ids
            self.ids = np.concatenate([pos_ids, neg_ids])
        else:
            # if len(self.ids) == 0: self.ids = np.concatenate([self.ids, self.pos_ids]) # 如果为空的话则首先加上pos_ids
            # self.ids = np.concatenate([self.ids, pos_ids, neg_ids])
            self.ids = np.concatenate([self.ids, pos_ids, neg_ids])

        self.ids = self.ids.astype(int)  # 为了做差集
        if self.pickout:
            out = np.setdiff1d(self.oids, self.ids)  # 计算剩下的ids的数量并返回
        else:
            out = self.oids
        if noisy_num > 0:
            noisy_select = out[np.random.permutation(len(out))][:noisy_num]
            self.P[np.intersect1d(results >= 0.5, noisy_select)] = 1
            self.ids = np.concatenate([self.ids, noisy_select], 0)
            if self.pickout:
                out = np.setdiff1d(self.oids, self.ids)
        if self.pickout:
            assert len(np.intersect1d(self.ids, out)) == 0  # 要求两者不能有重合
        return out
