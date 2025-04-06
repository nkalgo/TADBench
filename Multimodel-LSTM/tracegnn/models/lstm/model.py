import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tracegnn.constants import *

import numpy as np
from tqdm import tqdm as tq
from torch.nn.utils import clip_grad_norm_
import time
import os

from typing import *


# Hyper Parameters
batch_size = 512    # 原来512
max_epochs = 400  # 原来400
initial_lr = 0.001  # 原来0.001


def torch_seed(seed=2020):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


torch_seed()


class traceSet(Dataset):
    def __init__(self, D, Nt, Nl):
        self.x = D
        self.Nt = Nt
        self.Nl = Nl
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, i):
        a = self.x[i]
        assert a.shape[0] == self.Nt
        assert a.shape[1] == self.Nl + 1
        assert len(a.shape) == 2
        b = a[1:]  # (Nt-1, Nl + 1)
        end = np.zeros(self.Nl + 1, dtype=np.float32)
        end[0] = 1
        return a, np.concatenate([b, np.reshape(end, (1, -1))], axis=0)


class Multimodal(nn.Module):
    def __init__(self, Nl, h1_num, h2_num, hiddens):
        super(Multimodal, self).__init__()
        self.lstm01 = nn.LSTM(Nl, h1_num, 1, batch_first=True)
        self.lstm02 = nn.LSTM(1, h2_num, 1, batch_first=True)
        
        start = h1_num +  h2_num
        self.hidden_layers  = len(hiddens)
        
        self.lstms = []
        for i in range(self.hidden_layers):
            self.lstms.append(nn.LSTM(start, hiddens[i], 1, batch_first=True))
            start = hiddens[i]
        self.lstms = nn.ModuleList(self.lstms)

        self.lstm03 = nn.LSTM(start, Nl + 1, 1, batch_first=True)
        self.act1 = nn.Softmax(dim=2)
        
    def forward(self, x):
        x1 = x[:, :, :-1]
        x2 = x[:, :, -1:]
        y1, _ = self.lstm01(x1)
        y2, _ = self.lstm02(x2)
        
        x3 = torch.cat([y1, y2], dim=2)
        for n in range(self.hidden_layers):
             x3, _ = self.lstms[n](x3)
        
        res, _ = self.lstm03(x3)
        res1 = self.act1(res[:, :, :-1])
        res2 = res[:, :, -1:]

        return (res1, res2)


# x: torch [batch, len, Nl]
# y: ground truth
class Loss1(nn.Module):
    def __init__(self):
        super(Loss1, self).__init__()

    def forward(self, x, y):
        return -torch.mean(torch.sum(torch.log(x) * y, dim=2))


class Loss2(nn.Module):
    def __init__(self):
        super(Loss2, self).__init__()

    def forward(self, x, y):
        res = x - y
        return torch.mean(res * res)


class LossF(nn.Module):
    def __init__(self, a1=0.5, a2=0.5):
        super(LossF, self).__init__()
        self.l1 = Loss1()
        self.l2 = Loss2()
        self.a1 = a1
        self.a2 = a2

    def forward(self, x1, y1, x2, y2):
        ls1 = self.l1(x1, y1)
        ls2 = self.l2(x2, y2)
        return ls1 * self.a1 + ls2 * self.a2


def get_all(t):
    return '-'.join(t['labels'])


def clear(traces, Nt):
    n_traces = []
    for t in traces:
        nt = {
            'trace_id': t['trace_id'],
            'latency': t['latency'][:Nt],
            'labels': t['labels'][:Nt],
            'node_anomaly': t['node_anomaly'][:Nt],
            'anomaly': t['anomaly']
        }
        n_traces.append(nt)
    return n_traces


def update_idx(traces, idx):
    for t in traces:
        ms = t['labels']
        for m in ms:
            idx.setdefault(m, 0)
            idx[m] += 1
    return idx


def calculate_score(train_traces: List[dict],
                    test_traces: List[dict],
                    dataset: str,
                    model_label: str,
                    Nt: int=MAX_NODE_COUNT, 
                    device: str='cpu',
                    test_only: bool=False,
                    no_bias: bool=False):
    # Preprocess
    print('LSTM: Processing data...')
    a_no = {}
    a_len = {}
    for t in train_traces:
        a = get_all(t)
        if not a_no.__contains__(a):
            a_no[a] = len(a_no)
            a_len[a_no[a]] = len(t['latency'])
        t['shape'] = a_no[a]

    ta_d = {}
    for t in train_traces:
        ta_d.setdefault(t['shape'], 0)
        ta_d[t['shape']] += 1

    train_traces = clear(train_traces, Nt)

    print()
    test_traces = clear(test_traces, Nt)

    train_idx = update_idx(train_traces, {})
    train_idx = update_idx(test_traces, train_idx)

    idx = []
    for k in train_idx:
        idx.append((k, train_idx[k]))
    idx.sort(key=lambda x: -x[1])
    train_idx = idx

    label_no = {'0!': 0}  # o! 表示次数较少的label的统一形式
    for i in range(len(train_idx)):
        if train_idx[i][1] < 800:
            break
        label_no[train_idx[i][0]] = i + 1
    label_no['o!'] = len(label_no)
    Nl = len(label_no)

    mi_d = {}
    mx_d = {}
    for k in train_idx:
        mi_d[k[0]] = np.inf
        mx_d[k[0]] = 0

    def update_m_i_x(traces):
        for t in traces:
            rs = t['latency']
            for i, x in enumerate(t['labels']):
                if rs[i] < mi_d[x]:
                    mi_d[x] = rs[i]
                if rs[i] > mx_d[x]:
                    mx_d[x] = rs[i]

    def update_0_1(traces):
        for t in traces:
            rs = t['latency']
            for i, x in enumerate(t['labels']):
                mi = mi_d[x]
                mx = mx_d[x]
                if mx > mi:
                    rs[i] = (rs[i] - mi) / (mx - mi)
                elif rs[i] == mi:
                    rs[i] = 1
                else:
                    rs[i] = 6
            t['latency'] = rs
    update_m_i_x(train_traces)
    update_0_1(train_traces)
    update_0_1(test_traces)

    # Check
    label_set = {}
    print(f'Nt={Nt}')
    for t in train_traces:
        lbs = t['labels']
        for i, lb in enumerate(lbs):
            if i == Nt:
                break
            if label_no.__contains__(lb):
                label_set[lb] = True
    print(f'len(label_set)={len(label_set)}')
        
    for t in test_traces:
        lbs = t['labels']
        for i, lb in enumerate(lbs):
            if i == Nt:
                break
            if label_no.__contains__(lb):
                if not label_set.__contains__(lb):
                    # print('不包含', lb)
                    break

    # Trans Raw
    def trans_raw_d1(traces, idx, Nt):
        d1 = []
        for trace in tq(traces):
            t = []
            lbs = trace['labels']
            lts = trace['latency']
            for i, x in enumerate(lbs):
                if len(t) == Nt:
                    break
                if idx.__contains__(x):
                    t.append((idx[x], lts[i]))
                else:
                    t.append((idx['o!'], lts[i]))
            while len(t) < Nt:
                t.append((0, 0))
            assert len(t) == Nt
            nt = []
            for tp in t:
                y = np.zeros(len(idx) + 1)
                y[tp[0]] = 1
                y[-1] = tp[1]
                nt.append(y)
            d1.append(nt)
        return np.array(d1, dtype=np.float32)

    print('LSTM: Generating train dataset...')
    train_d1 = trans_raw_d1(train_traces, label_no, Nt)
    print(f'train_d1.shape={train_d1.shape}')
    print('LSTM: Generating test dataset...')
    test_d1 = trans_raw_d1(test_traces, label_no, Nt)

    print('LSTM: Building model...')
    # Dataset
    train_set = traceSet(train_d1, Nt, Nl)

    # Model
    net = Multimodal(Nl, 64, 8, [128])
    cret = LossF(0.01, 1.5)
    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.75)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Start training
    if not test_only:
        net.to(device)
        print('LSTM: Start training...')
        train_start = time.time()
        train_losses = []

        best_loss = np.inf

        for e in range(1, max_epochs + 1):
            print(f'LSTM: Epoch {e} ----------------->')
            net.train()
            train_loss = 0.0
            
            for x, y_gt in tq(train_loader):
                x, y_gt = x.to(device), y_gt.to(device)
                optimizer.zero_grad()
                y1_pr, y2_pr = net(x)
                loss = cret(y1_pr, y_gt[:, :, :-1], y2_pr, y_gt[:, :, -1:])
                loss.backward()
                # clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()
                train_loss += loss.item() * x.shape[0]
            
            train_losses.append(train_loss / train_set.__len__())
            print('epoch :%s, train_loss:%s' % (e,  train_losses[-1]))
            # scheduler.step()
            
            if train_losses[-1] < best_loss:
                best_loss = train_losses[-1]
                os.makedirs('tracegnn/models/lstm/model_dat', exist_ok=True)
                torch.save(net.state_dict(), f'tracegnn/models/lstm/model_dat/{model_label}')
        train_end = time.time()
        train_time = train_end - train_start
        print(f'LSTM: Training finished. train_time={train_time}')

    # Evaluate
    test_set = traceSet(test_d1, Nt, Nl)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f'LSTM: Evaluate...')
    test_start_time = time.time()
    net.load_state_dict(torch.load(f'tracegnn/models/lstm/model_dat/{model_label}', map_location=device))
    net = net.to(device)
    net.eval()

    def get_res(net, data_loader):
        res = []
        with torch.no_grad():
            for x, y_gt in tq(data_loader):  # 真实标签 y_gt
                x = x.to(device)
                y_gt = y_gt.numpy()
                y1_pr, y2_pr = net(x)
                y1_pr = y1_pr.cpu().detach().numpy()
                y2_pr = y2_pr.cpu().detach().numpy()
                for i in range(y1_pr.shape[0]):
                    res.append((y1_pr[i], y2_pr[i], y_gt[i]))
        return res

    res_test = get_res(net, test_loader)

    rt_d = {}

    print('LSTM: Calculating res_test...')
    for tp in tq(res_test):
        # y1就是lbs (event_id)的重构
        y1 = tp[0]
        # y2就是latency的重构
        y2 = tp[1]
        # Event one-hot label
        y1_g = tp[2][:, :-1]
        # Event latency
        y2_g = tp[2][:, -1:]
    
        # Calculate latency loss (of each event)
        rt = np.reshape((y2 - y2_g) * (y2 - y2_g), (-1))

        # argmax (one-hot -> int)
        lbs = np.argmax(y1_g, axis=-1).tolist()
        
        for i, lb in enumerate(lbs):
            rt_d.setdefault(lb, [])
            rt_d[lb].append(rt[i])

    # calculate mean and std for rt_d
    mean_d = {}
    std_d = {}
    for lb in rt_d:
        tmp = np.array(rt_d[lb], dtype=np.float32)
        mean_d[lb]= np.mean(tmp)
        std_d[lb]= np.std(tmp)

    def get_anomaly(res, trace_lengths):
        anomaly_idx = []
        all_node_lat_loss = []
        # for tid, tp in enumerate(tq(res)):
        for tid, (tp, length) in enumerate(zip(tq(res), trace_lengths)):  # Match trace length
            y1 = tp[0]
            y2 = tp[1]
            y1_g = tp[2][:, :-1]
            y2_g = tp[2][:, -1:]
            
            # 和上面一样
            rt = np.reshape((y2 - y2_g) * (y2 - y2_g), (-1))  # 自加：每个事件的 latency reconstruction loss
            # rt[i] 表示第 i 个事件的延迟预测误差，mean_d[lb] 和 std_d[lb] 分别是标签为 lb 的事件的延迟误差的均值和标准差。
            lbs = np.argmax(y1_g, axis=-1).tolist()  # 自加：每个事件的实际标签索引
            assert len(lbs) == rt.shape[0]
            rank_m = 0
            ok = 0
            max_nll = 0.0
            
            cnt = 0

            node_lat_loss = []

            for i in range(min(length, len(lbs))):
                lb = lbs[i]
                if mean_d.__contains__(lb):
                    tmp = np.fabs((rt[i] - mean_d[lb]) / std_d[lb])
                    node_lat_loss.append(tmp)
                else:
                    print('不包含', rt[i])
                    node_lat_loss.append(rt[i])

            for i, lb in enumerate(lbs):
                rank = int(np.sum(y1[i] > y1[i][lb]))
                rank_m = max(rank, rank_m)

                if mean_d.__contains__(lb):
                    tmp = np.fabs((rt[i] - mean_d[lb]) / std_d[lb])
                    ok = max(ok, tmp)

                # Calculate biased nll
                cur_nll = -np.log(y1[i][lb]+1e-8)

                if not no_bias:
                    if y1[i][lb] < 2.0 / Nl:
                        cnt += 1
                        cur_nll *= Nt
                
                max_nll += cur_nll

            anomaly_idx.append((ok, rank_m, max_nll))
            all_node_lat_loss.extend(node_lat_loss)

        return anomaly_idx, all_node_lat_loss

    print('LSTM: Ranking...')
    trace_lengths = [len(trace['node_anomaly']) for trace in test_traces]
    test_result, node_lat_loss = get_anomaly(res_test, trace_lengths)
    test_time = time.time() - test_start_time
    node_latency = node_lat_loss
    nll_latency = [i[0] for i in test_result]
    nll_drop = [i[2] for i in test_result]
    labels = [i['anomaly'] for i in test_traces]
    node_labels = []
    for i in test_traces:
        for j in i['node_anomaly']:
            node_labels.append(j)
    print('sum(labels)', sum(labels))
    print('set(labels)', set(labels))
    print('len(node_labels) == len(node_latency)', len(node_labels) == len(node_latency))
    print('len(node_labels)', len(node_labels))
    print('len(node_latency)', len(node_latency))
    return nll_latency, nll_drop, labels, node_latency, node_labels, test_time
