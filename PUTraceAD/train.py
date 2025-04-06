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
import shutil
from tqdm import tqdm


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

def get_criterion():
    positive_probability = 0.25  # 0.15

    if args.pratio != 1:
        positive_probability *= args.pratio
    print(f"{positive_probability=}")

    weights = [float(args.weight), 1.0]
    class_weights = torch.FloatTensor(weights)

    class_weights = class_weights.cuda(args.gpu)
    if args.loss == 'Xent':
        criterion = PULoss(Probability_P=positive_probability, loss_fn="Xent")
    elif args.loss == 'nnPU':
        criterion = PULoss(Probability_P=positive_probability, BETA=args.beta)
    elif args.loss == 'Focal':
        class_weights = torch.FloatTensor(weights).cuda(args.gpu)
        criterion = FocalLoss(gamma=0, weight=class_weights, one_hot=False)
    elif args.loss == 'uPU':
        criterion = PULoss(Probability_P=positive_probability, nnPU=False)
    elif args.loss == 'Xent_weighted':
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    return criterion


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


def check_mean_teacher(epoch):
    if not args.mean_teacher:
        return False
    elif epoch < args.ema_start:
        return False
    else:
        return True


def check_self_paced(epoch):
    if not args.self_paced:
        return False
    elif args.self_paced and epoch >= args.self_paced_stop:
        return False
    elif args.self_paced and epoch < args.self_paced_start:
        return False
    else:
        return True


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


def train(noisy1_loader, model1, criterion, optimizer1, scheduler1, epoch):
    global step, switched
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    pacc1 = AverageMeter()
    nacc1 = AverageMeter()
    pnacc1 = AverageMeter()
    count_clean = AverageMeter()
    count_noisy = AverageMeter()
    model1.train()
    end = time.time()

    # print(f"{model1.state_dict()=}")
    entropy_clean = AverageMeter()
    entropy_noisy = AverageMeter()

    scheduler1.step()

    epoch_train_cnt = 0
    batch_cnt = 0

    for i, (X, Y, _, T, ids, _) in enumerate(noisy1_loader):
        # print(f"{ids=}\n{ids.view(-1)=}\n{ids.view(-1).numpy()=}\n")
        epoch_train_cnt += len(ids)
        batch_cnt += 1

        X = X.cuda(args.gpu)
        # if args.dataset == 'mnist':
        #     X = X.view(X.shape[0], 1, -1)
        Y = Y.cuda(args.gpu).float()
        T = T.cuda(args.gpu).long()

        output1 = model1(X.x, X.edge_index, X.batch)

        _, loss = criterion(output1, Y)
        predictions1 = torch.sign(output1).long()

        smx1 = torch.sigmoid(output1)
        smx1 = torch.cat([1 - smx1, smx1], dim=1)
        aux1 = - torch.sum(smx1 * torch.log(smx1 + 1e-10)) / smx1.shape[0]
        entropy_noisy.update(aux1, 1)

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        pacc_1, nacc_1, pnacc_1, psize = accuracy(predictions1, T)  # 使用T来计算预测准确率
        pacc1.update(pacc_1, psize)
        nacc1.update(nacc_1, Y.size(0) - psize)
        pnacc1.update(pnacc_1, Y.size(0))

    print('Epoch Noisy : [{0}]\t'
          'PACC1 {pacc1.val:.3f} ({pacc1.avg:.3f})\t'
          'NACC1 {nacc1.val:.3f} ({nacc1.avg:.3f})\t'
          'PNACC1 {pnacc1.val:.3f} ({pnacc1.avg:.3f})\t'.format(
        epoch, pacc1=pacc1, nacc1=nacc1, pnacc1=pnacc1))
    print(f"{epoch_train_cnt=} {batch_cnt=}")
    return pacc1.avg, nacc1.avg, pnacc1.avg


def validate(val_loader, model1, dataset_name, epoch=None, final=False, anomaly_type="total"):
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

    print("=====================================")

    return pacc.avg, nacc.avg, pnacc1.avg, f1_s1


def putracead_train(dataset_name):
    global args, switched, DATASET_LABEL_LENGTH, DATASET_UNLABEL_LENGTH, DATASET_ALL_LENGTH
    args = parser.parse_args()
    print(args)
    criterion = get_criterion()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    print(f"{args.soft_label=}")

    trainX, trainY = get_my_trace_graph_datas(type="train", root="data/%s" % dataset_name, seed=args.seed)
    validateX, validateY = get_my_trace_graph_datas(type="validate", root="data/%s" % dataset_name, seed=args.seed)
    print(f"{len(trainX)=} {len(validateX)=} ")
    DATASET_ALL_LENGTH = len(trainX)
    real_pp = 0.25  # 0.1526
    DATASET_LABEL_LENGTH = int(DATASET_ALL_LENGTH * real_pp * args.labeled)
    print(f"{args.labeled=} {DATASET_LABEL_LENGTH=}")
    DATASET_UNLABEL_LENGTH = DATASET_ALL_LENGTH - DATASET_LABEL_LENGTH

    dataset_train1_noisy = GraphDatasetFixSample(DATASET_LABEL_LENGTH, DATASET_UNLABEL_LENGTH,
                                                 trainX, trainY, split='train',
                                                 increasing=args.increasing, replacement=args.replacement,
                                                 mode=args.self_paced_type, top=args.top1, type="noisy",
                                                 seed=args.seed)
    print(f"{dataset_train1_noisy=} {len(dataset_train1_noisy)=}")
    dataset_train1_noisy.reset_ids()

    dataset_validate = GraphDatasetFixSample(DATASET_LABEL_LENGTH, DATASET_UNLABEL_LENGTH,
                                             validateX, validateY, split='test',
                                             increasing=args.increasing, replacement=args.replacement,
                                             mode=args.self_paced_type, type="clean", seed=args.seed)
    print(f"{dataset_validate=} {len(dataset_validate)}")

    dataloader_train1_noisy = DataLoader(dataset_train1_noisy, collate_fn=graph_collate_fn, batch_size=args.batch_size,
                                         num_workers=args.workers, shuffle=False, pin_memory=True)
    dataloader_validate = DataLoader(dataset_validate, collate_fn=graph_collate_fn, batch_size=args.batch_size,
                                     num_workers=args.workers, shuffle=False, pin_memory=True)

    model1 = create_graph_model()

    if args.gpu is not None:
        model1 = model1.cuda(args.gpu)
    else:
        model1 = model1.cuda(args.gpu)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    stats_ = stats(args.modeldir, 0)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, args.epochs)

    best_acc1 = 0
    best_f1 = 0
    best_model = None

    time_train_start = time.time()

    all_f1 = []
    models = []

    for epoch in range(args.epochs):
        print("Self paced status: {}".format(check_self_paced(epoch)))
        print("Mean Teacher status: {}".format(check_mean_teacher(epoch)))

        trainPacc, trainNacc, trainPNacc = train(dataloader_train1_noisy, model1, criterion, optimizer1, scheduler1,
                                                 epoch)
        valPacc, valNacc, valPNacc1, f1_1 = validate(dataloader_validate, model1, dataset_name, epoch, final=False)
        stats_._update(trainPacc, trainNacc, trainPNacc, valPacc, valNacc, valPNacc1)

        best_acc1 = max(valPNacc1, best_acc1)

        print(f'{f1_1=}')
        all_f1.append(f1_1)
        models.append(model1)

        modelname = os.path.join('model', 'model_%s.pth.tar' % dataset_name)

        if max(all_f1) > best_f1:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': models[all_f1.index(max(all_f1))].state_dict(),
                'best_prec1': best_acc1,
            }, modelname)
            best_f1 = max(all_f1)
            print("new best_f1={:.4f}%".format(100 * best_f1))

    train_seconds = int(time.time() - time_train_start)
    print(f"{best_acc1=}")
    print("\nbest validate f1={:.4f}%".format(100 * best_f1))
    print(f"{train_seconds=}")

    result_data = {}
    if os.path.exists('../results.json'):
        with open('../results.json', 'r') as f:
            result_data = json.load(f)
    dataset_results = result_data.setdefault(dataset_name, {})
    algorithm_results = dataset_results.setdefault('PUTraceAD', {'total': {}, 'structure': {}, 'latency': {}})
    for key in ['total', 'structure', 'latency']:
        algorithm_results[key]['time'] = train_seconds
    with open('../results.json', 'w') as f:
        json.dump(result_data, f, indent=4)



