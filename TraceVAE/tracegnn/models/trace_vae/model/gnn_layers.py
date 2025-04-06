from enum import Enum
from typing import *

import mltk
import tensorkit as tk
from dgl import nn as gnn
from tensorkit import tensor as T

__all__ = [
    'GNNLayerType',
    'GNNLayerConfig',
    'make_gnn_layers',
    'apply_gnn_layer',
    'GNNSequential',
    'GATConvAgg',
    'GraphConv',
]


class GNNLayerType(str, Enum):
    GAT = 'GAT'
    GraphConv = 'GraphConv'


class GNNLayerConfig(mltk.Config):
    type: GNNLayerType = GNNLayerType.GAT

    # whether to use batch norm?
    use_batch_norm: bool = True

    # config for GAT
    class gat(mltk.Config):
        num_attention_heads: int = 2


def make_gnn_layers(config: GNNLayerConfig,
                    input_dim: int,
                    gnn_layers: List[int],
                    ):
    if config.use_batch_norm:
        normalization_factory = tk.layers.BatchNorm
    else:
        normalization_factory = lambda num_inputs: None

    layers = []
    for size in gnn_layers:
        if config.type == GNNLayerType.GAT:
            layers.append(GATConvAgg(
                input_dim,
                size,
                config.gat.num_attention_heads,
                activation=tk.layers.LeakyReLU(),
                normalization_factory=normalization_factory,
            ))
        elif config.type == GNNLayerType.GraphConv:
            layers.append(GraphConv(
                input_dim,
                size,
                activation=tk.layers.LeakyReLU(),
                normalization_factory=normalization_factory,
            ))
        else:
            raise ValueError(f'Unsupported GNN type: {config.type!r}')
        input_dim = layers[-1].output_dim

    return input_dim, layers


def apply_gnn_layer(layer, g, h):
    if isinstance(g, (list, tuple)):
        if len(h.shape) == 3:
            if len(g) != h.shape[0]:
                raise ValueError(f'len(g) != h.shape[0]: {len(g)} vs {h.shape[0]}')
            return T.stack(
                [
                    layer(g[i], h[i])
                    for i in range(len(g))
                ],
                axis=0
            )
        else:
            return T.stack(
                [
                    layer(g[i], h)
                    for i in range(len(g))
                ],
                axis=0
            )
    else:
        if len(h.shape) == 3:
            return T.stack(
                [
                    layer(g, h[i])
                    for i in range(h.shape[0])
                ],
                axis=0
            )
        else:
            return layer(g, h)

'''
apply_gnn_layer方法
该方法将指定的GNN层应用到图g和节点特征h上。
处理逻辑：
如果输入的图数据g是列表或元组（即批次的多个图），则对每个图分别应用GNN层，并将结果堆叠到一起。
如果图g是单个图但特征h是批量形式（即三维张量），则对每个特征批次逐一应用GNN层。
最后返回堆叠后的结果或单个图的处理结果。
'''


class GNNSequential(tk.layers.BaseLayer):

    def __init__(self, layers):
        super().__init__()
        self.gnn = gnn.Sequential(*layers)

    def forward(self, g, h):
        return apply_gnn_layer(self.gnn, g, h)

'''
GNNSequential类
继承自tensorkit的BaseLayer，该类用于创建顺序GNN模型，即将多层GNN串联起来。
forward方法：在图g和节点特征h上应用多层GNN，实际使用时会调用apply_gnn_layer来处理批量图。
'''


class GATConvAgg(tk.layers.BaseLayer):
    """First apply `dgl.nn.GATConv` then aggregate the multi attention heads."""

    aggregate_mode: str
    output_dim: int

    def __init__(self, input_dim: int, output_dim: int, num_heads: int,
                 aggregate_mode: str = 'concat', activation=None,
                 normalization_factory=None):
        super().__init__()

        if aggregate_mode == 'concat':
            self.output_dim = output_dim * num_heads
        elif aggregate_mode in ('mean', 'avg'):
            self.output_dim = output_dim
        else:
            raise ValueError(f'Unsupported aggregate_mode: {aggregate_mode!r}')

        self.activation = activation
        self.normalization = None if normalization_factory is None else \
            normalization_factory(self.output_dim)

        self.gnn = gnn.GATConv(
            input_dim,
            output_dim,
            num_heads,
            activation=None,
        )
        self.aggregate_mode = aggregate_mode

    def forward(self, g, h):
        h = self.gnn(g, h)
        if self.aggregate_mode == 'concat':
            h = T.concat(
                [h[..., i, :] for i in range(h.shape[-2])],
                axis=-1
            )
        else:
            h = T.reduce_mean(h, axis=[-2])

        if self.normalization is not None:
            h = self.normalization(h)
        if self.activation is not None:
            h = self.activation(h)

        return h
'''
GATConvAgg类
图注意力卷积（GAT）层的实现。GAT层通过多个注意力头对节点特征进行加权求和，并根据aggregate_mode汇总这些注意力头的输出。
forward方法：在图g和节点特征h上应用GAT层。
使用GAT层生成多头注意力的输出，然后根据aggregate_mode对这些头进行汇总。
如果设置了批量归一化和激活函数，会进一步对汇总结果进行处理。
'''


class GraphConv(tk.layers.BaseLayer):

    output_dim: int

    def __init__(self, input_dim: int, output_dim: int, activation=None,
                 normalization_factory=None):
        super().__init__()
        self.output_dim = output_dim

        self.activation = activation
        self.normalization = None if normalization_factory is None else \
            normalization_factory(self.output_dim)

        self.gnn = gnn.GraphConv(
            input_dim,
            output_dim,
            norm='both',
            weight=self.normalization is None,
            bias=self.normalization is None,
            activation=None,
        )

    def forward(self, g, h):
        h = self.gnn(g, h)

        if self.normalization is not None:
            h = self.normalization(h)
        if self.activation is not None:
            h = self.activation(h)

        return h

'''
GraphConv类
基于图卷积（GraphConv）的图神经网络层实现。主要用于对节点特征进行卷积操作。
forward方法：在图g和节点特征h上应用图卷积。
通过dgl.nn.GraphConv层对图进行卷积操作。
如果设置了批量归一化和激活函数，会对卷积结果进一步处理。
'''