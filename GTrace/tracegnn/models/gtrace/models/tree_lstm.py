# https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py
"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import time
import itertools
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import ExpConfig
import dgl


class ChildSumTreeLSTMOp(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMOp, self).__init__()
        self.W_iouf = nn.Linear(x_size, 4 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(nodes.data['f'].unsqueeze(1) + self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'sum': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + nodes.data['sum'] + self.b_iou
        # print(f'{iou.shape=}')
        i, o, u = torch.chunk(iou, 3, -1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}

class TreeLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int):
        super(TreeLSTM, self).__init__()

        self.x_size = input_size
        self.cell = ChildSumTreeLSTMOp(self.x_size, self.x_size)
        self.linear_o = nn.Linear(self.x_size, output_size)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor):
        # remove self-loop & reverse
        g = dgl.reverse(dgl.remove_self_loop(g))
        '''
        dgl.reverse(g)：反转图的方向，即将所有的边方向翻转。反转图有时有助于实现某些图操作
        （例如，树形LSTM中常常反转树的边，以便进行自上而下的计算）。
        '''
        # feed embedding
        g.ndata['iou'], g.ndata['f'] = torch.split(self.cell.W_iouf(x), [3 * self.x_size, self.x_size], dim=-1)
        # 通过 self.cell.W_iouf(x) 计算节点的输入门、遗忘门、输出门以及候选状态。
        # self.cell.W_iouf(x)：这会生成一个形状为 (batch_size, num_nodes, 4 * self.x_size) 的张量，
        # 其中每个节点的特征被分成 4 个部分，用于计算输入门（i）、遗忘门（f）、输出门（o）和候选细胞状态（u）。
        g.ndata['sum'] = torch.zeros_like(g.ndata['iou'])
        # 初始化节点的 sum 特征，这用于存储节点的子节点汇聚的结果。
        g.ndata['h'] = torch.zeros_like(x)
        g.ndata['c'] = torch.zeros_like(x)
        # propagate
        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        # compute output
        h = g.ndata.pop('h')
        output = self.linear_o(torch.relu(h))

        return output
