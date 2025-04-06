import argparse
import copy
import csv
import json
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from final_nnpu_datasets import get_my_trace_graph_datas, GraphDatasetFixSample
from functions import stats, plot_curve
from models import MyConfig
from utils.util import PULoss, FocalLoss
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, GAT, GATConv, CGConv, GCN, GINConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_sort_pool, global_add_pool, GCNConv
from torch.nn import Sequential, ReLU, Linear, LeakyReLU, Tanh, ModuleList
from sklearn import metrics
import time


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--batch-size', '-b', type=int, default=128, help='batch-size')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--modeldir', type=str, default="model/", help="Model path")
parser.add_argument('--epochs', type=int, default=50)  # 50
parser.add_argument('--loss', type=str, default='nnPU')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('-j', '--workers', default=4, type=int, help='workers')

parser.add_argument('--weight', type=float, default=1.0)

parser.add_argument('--self-paced', type=boolean_string, default=False)
parser.add_argument('--self-paced-start', type=int, default=10)
parser.add_argument('--self-paced-stop', type=int, default=50)
parser.add_argument('--self-paced-frequency', type=int, default=10)
parser.add_argument('--self-paced-type', type=str, default="A")

parser.add_argument('--increasing', type=boolean_string, default=True)
parser.add_argument('--replacement', type=boolean_string, default=True)

parser.add_argument('--mean-teacher', type=boolean_string, default=False)
parser.add_argument('--ema-start', type=int, default=50)
parser.add_argument('--ema-decay', type=float, default=0.999)
parser.add_argument('--consistency', type=float, default=0.3)
parser.add_argument('--consistency-rampup', type=int, default=400)

parser.add_argument('--evaluation', action="store_true")
parser.add_argument('--top1', type=float, default=0.4)
parser.add_argument('--top2', type=float, default=0.6)
parser.add_argument('--soft-label', action="store_true")

parser.add_argument('--type', type=str, default="mu")
parser.add_argument('--alpha', type=float, default=0.1)

parser.add_argument("--gnn", type=str, default="gat")
parser.add_argument("--gnnlayer", type=int, default=3)
parser.add_argument("--labeled", type=float, default=0.05)
parser.add_argument("--compare", type=str, default="default")
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument('--pratio', type=float, default=1)
parser.add_argument("--beta", type=float, default=0)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--anomaly_type", type=str, default="total")

step = 0
results = np.zeros(61000)
switched = False
results1 = None
results2 = None
args = None

DATASET_ALL_LENGTH = 0
DATASET_LABEL_LENGTH = 0
DATASET_UNLABEL_LENGTH = 0


def create_graph_model(ema=False):
    print(f"{args.gnn=}")

    config = MyConfig()

    # Only BERT
    # config.hidden_dims = 785
    # config.embedding_size = 785

    # BERT + NodeVec
    print(f"{args.gnn=} {args.compare=}")

    dropout = args.dropout

    if args.gnn == "gat":
        if args.compare == "default":
            config.embedding_size = 931
            config.hidden_dims = 931
            config.num_layers = args.gnnlayer
            config.num_classes = 1

            print(f"{config.hidden_dims=}, {config.embedding_size=} {config.num_layers=}")

            class MyGAT(nn.Module):

                def __init__(self, config: MyConfig):
                    super(MyGAT, self).__init__()
                    self.time_linear = nn.Linear(4, 40)
                    self.time_linear2 = nn.Linear(40, 100, bias=False)

                    self.convs = ModuleList()
                    self.batch_norms = ModuleList()

                    if config.num_layers <= 1:
                        conv_0 = GATConv(in_channels=config.embedding_size, out_channels=config.embedding_size,
                                         dropout=dropout)
                        bn_0 = BatchNorm(config.embedding_size)
                        self.convs.append(conv_0)
                        self.batch_norms.append(bn_0)
                    else:
                        conv_0 = GATConv(in_channels=config.embedding_size, out_channels=config.embedding_size,
                                         heads=3, dropout=dropout)
                        conv_1 = GATConv(in_channels=config.embedding_size * 3, out_channels=config.embedding_size,
                                         heads=1, dropout=dropout)
                        bn_0 = BatchNorm(config.embedding_size * 3)
                        bn_1 = BatchNorm(config.embedding_size)

                        self.convs.append(conv_0)
                        self.batch_norms.append(bn_0)

                        config.num_layers -= 2
                        if config.num_layers > 0:
                            for i in range(config.num_layers):
                                conv = GATConv(in_channels=config.embedding_size * 3,
                                               out_channels=config.embedding_size, heads=3)
                                bn = BatchNorm(config.embedding_size * 3)
                                self.convs.append(conv)
                                self.batch_norms.append(bn)

                        self.convs.append(conv_1)
                        self.batch_norms.append(bn_1)

                    self.mlp = nn.Sequential(Linear(config.hidden_dims, 8),
                                             Tanh(),
                                             Linear(8, config.num_classes))

                def forward(self, x, edge_index, batch):

                    time_expand = self.time_linear(x[:, 768:772])
                    time_expand = F.softmax(time_expand)
                    time_expand = self.time_linear2(time_expand)

                    x = torch.cat([x[:, :768].clone().detach(),
                                   time_expand,
                                   x[:, 772:].clone().detach()], dim=-1)

                    for conv, batch_norm in zip(self.convs, self.batch_norms):
                        x = torch.tanh(conv(x, edge_index))
                        x = batch_norm(x)

                    x = global_mean_pool(x, batch)
                    x = F.relu(x)
                    x = self.mlp(x)
                    return x

            model = MyGAT(config)

    else:
        raise Exception(f"{args.gnn=}")

    if ema:
        for param in model.parameters():
            param.detach_()
    return model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count == 0:
            self.avg = 0
        else:
            self.avg = self.sum / self.count


def accuracy(output, target):
    with torch.no_grad():

        batch_size = float(target.size(0))

        output = output.view(-1)
        correct = torch.sum(output == target).float()

        pcorrect = torch.sum(output[target == 1] == target[target == 1]).float()
        ncorrect = correct - pcorrect

    ptotal = torch.sum(target == 1).float()

    if ptotal == 0:
        return torch.tensor(0.).cuda(args.gpu), ncorrect / (
                    batch_size - ptotal) * 100, correct / batch_size * 100, ptotal
    elif ptotal == batch_size:
        return pcorrect / ptotal * 100, torch.tensor(0.).cuda(args.gpu), correct / batch_size * 100, ptotal
    else:
        return pcorrect / ptotal * 100, ncorrect / (batch_size - ptotal) * 100, correct / batch_size * 100, ptotal


def validate(val_loader, model1, dataset_name, epoch=None, final=True, anomaly_type="total"):
    if epoch is None:
        epoch = -1

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pacc = AverageMeter()
    nacc = AverageMeter()
    pnacc1 = AverageMeter()
    model1.eval()
    end = time.time()

    all_labels = []
    all_predict_s1 = []

    if final:
        final_records = []

    with torch.no_grad():
        for i, (X, Y, _, T, ids, _) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            X = X.cuda(args.gpu)
            Y = Y.cuda(args.gpu).float()
            T = T.cuda(args.gpu).long()

            output1 = model1(X.x, X.edge_index, X.batch)

            predictions1 = torch.sign(output1).long()
            predict_s1 = predictions1.view(-1).tolist()
            all_predict_s1 += predict_s1

            labels = T.tolist()
            all_labels += labels

            if final:
                for i in range(len(ids)):
                    trace_id = X[i].trace_id
                    predict = predict_s1[i]
                    label = labels[i]
                    record = {"trace_id": trace_id, "predict": predict, "label": label}
                    final_records.append(record)

            pacc_, nacc_, pnacc_, psize = accuracy(predictions1, T)
            pacc.update(pacc_, Y.size(0))
            nacc.update(nacc_, Y.size(0))
            pnacc1.update(pnacc_, Y.size(0))

    all_labels = np.array(all_labels)
    all_predict_s1 = np.array(all_predict_s1)

    TP = np.sum((all_labels == 1) & (all_predict_s1 == 1))
    FP = np.sum((all_labels == -1) & (all_predict_s1 == 1))
    FN = np.sum((all_labels == 1) & (all_predict_s1 == -1))
    TN = np.sum((all_labels == -1) & (all_predict_s1 == -1))
    print(f'{TP=}, {FN=}, {FP=}, {TN=}')

    precision_s1 = round(metrics.precision_score(all_labels, all_predict_s1), 4)
    recall_s1 = round(metrics.recall_score(all_labels, all_predict_s1), 4)
    f1_s1 = round(metrics.f1_score(all_labels, all_predict_s1), 4)
    acc_s1 = round(metrics.accuracy_score(all_labels, all_predict_s1), 4)
    print(
        'Finish testing, p={:.4f}%, r={:.4f}%, f1={:.4f}%, acc={:.4f}%'.format(100. * precision_s1,
                                                                               100. * recall_s1, 100. * f1_s1,
                                                                               100. * acc_s1))

    print('Test [{0}]: \t'
          'PNACC1 {pnacc1.val:.3f} ({pnacc1.avg:.3f})\t'.format(
        epoch, pnacc1=pnacc1, ))

    if final:
        result_data = {}
        if os.path.exists('../results.json'):
            with open('../results.json', 'r') as f:
                result_data = json.load(f)

        dataset_results = result_data.setdefault(dataset_name, {})
        algorithm_results = dataset_results.setdefault('PUTraceAD', {'total': {}, 'structure': {}, 'latency': {}})
        algorithm_results[anomaly_type].update({
            'p': precision_s1, 'r': recall_s1, 'f1': f1_s1, 'acc': acc_s1
        })
        with open('../results.json', 'w') as f:
            json.dump(result_data, f, indent=4)

    print("=====================================")


def graph_collate_fn(batch):
    # data_length = len(batch)
    datas = [row[0] for row in batch]
    data_batch = Batch.from_data_list(datas)

    ys = [row[1] for row in batch]
    ys = torch.IntTensor(ys)

    ps = [row[2] for row in batch]
    ps = torch.IntTensor(ps)

    ts = [row[3] for row in batch]
    ts = torch.IntTensor(ts)

    ids = [row[4] for row in batch]
    ids = torch.IntTensor(ids)

    temps = [row[5] for row in batch]
    temps = torch.Tensor(temps)

    return data_batch, ys, ps, ts, ids, temps


def putracead_test(dataset_name, anomaly_type):
    global args, switched, DATASET_LABEL_LENGTH, DATASET_UNLABEL_LENGTH, DATASET_ALL_LENGTH
    args = parser.parse_args()
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    print(f"{args.soft_label=}")

    testX, testY = get_my_trace_graph_datas(type="test", root="data/%s" % dataset_name, seed=args.seed,
                                            anomaly_type=anomaly_type)
    dataset_test = GraphDatasetFixSample(DATASET_LABEL_LENGTH, DATASET_UNLABEL_LENGTH,
                                         testX, testY, split='test',
                                         increasing=args.increasing, replacement=args.replacement,
                                         mode=args.self_paced_type, type="clean", seed=args.seed)
    print(f"{dataset_test=} {len(dataset_test)=}")
    dataloader_test = DataLoader(dataset_test, collate_fn=graph_collate_fn, batch_size=args.batch_size,
                                 num_workers=args.workers, shuffle=False, pin_memory=True)
    model1 = create_graph_model()

    checkpoint = torch.load(
        'model/model_%s.pth.tar' % dataset_name,
        map_location=torch.device(args.gpu if args.gpu else 'cpu')
    )
    model1.load_state_dict(checkpoint['state_dict'])

    if args.gpu is not None:
        model1 = model1.cuda(args.gpu)
    else:
        model1 = model1.cuda(args.gpu)

    model1.eval()
    validate(dataloader_test, model1, dataset_name, anomaly_type=anomaly_type)
