from typing import *

import dgl
import mltk
import tensorkit as tk
import torch
from tensorkit import tensor as T

from ..constants import *
from ..distributions import *
from ..tensor_utils import *
from .gnn_layers import *
from .model_utils import *
from .operation_embedding import *
from .pooling import *
from .realnvp_flow import *

__all__ = [
    'TraceStructVAEConfig',
    'TraceStructVAE',
]


class TraceStructVAEConfig(mltk.Config):
    # the dimension of z (to encode adj & node_type)
    z_dim: int = 3

    # the config of posterior / prior flow
    realnvp: RealNVPFlowConfig = RealNVPFlowConfig()

    # whether to use BatchNorm?
    use_batch_norm: bool = True

    class encoder(mltk.Config):
        # ===============
        # h(G) for q(z|G)
        # ===============
        # the gnn layer config
        gnn: GNNLayerConfig = GNNLayerConfig()

        # the gnn layer sizes for q(z|...)
        gnn_layers: List[int] = [500, 500, 500, 500]

        # =============
        # graph pooling
        # =============
        pool_type: PoolingType = PoolingType.AVG
        pool_config: PoolingConfig = PoolingConfig()

        # ======
        # q(z|G)
        # ======
        z_logstd_min: Optional[float] = -7
        z_logstd_max: Optional[float] = 2

        # whether to use realnvp posterior flow?
        use_posterior_flow: bool = False

    class decoder(mltk.Config):
        # ====================
        # decoder architecture
        # ====================
        use_prior_flow: bool = False

        # whether to use `z` directly as context, instead of passing through
        # the graph embedding layers?
        z_as_context: bool = False

        # whether to use `node_depth` and `node_idx` as extra information?
        use_depth: bool = False
        use_idx: bool = True

        # =========
        # structure
        # =========
        # gnn layer config
        gnn: GNNLayerConfig = GNNLayerConfig()

        # the node types from node embedding e
        gnn_layers: List[int] = [500, 500, 500, 500]

        # hidden layers for p(node_count|z)
        node_count_layers: List[int] = [500]

        # hidden layers for graph embedding from z
        graph_embedding_layers: List[int] = [500, 500]

        # size of the latent embedding e
        latent_embedding_size: int = 40


class TraceStructVAE(tk.layers.BaseLayer):

    config: TraceStructVAEConfig
    num_operations: int

    def __init__(self,
                 config: TraceStructVAEConfig,
                 operation_embedding: OperationEmbedding,
                 ):
        super().__init__()

        # ===================
        # memorize the config
        # ===================
        self.config = config

        # =============================
        # node embedding for operations
        # =============================
        self.operation_embedding = operation_embedding
        self.num_operations = operation_embedding.num_operations

        # ========================
        # standard layer arguments
        # ========================
        layer_args = tk.layers.LayerArgs()
        layer_args.set_args(['dense'], activation=tk.layers.LeakyReLU)
        if config.use_batch_norm:
            layer_args.set_args(['dense'], normalizer=tk.layers.BatchNorm)

        # ==================
        # q(z|adj,node_type)
        # ==================
        # 用于创建编码器的GNN层
        output_size, gnn_layers = make_gnn_layers(
            config.encoder.gnn,
            self.operation_embedding.embedding_dim,
            config.encoder.gnn_layers,
        )
        self.qz_gnn_layers = GNNSequential(
            gnn_layers + [
                make_graph_pooling(
                    output_size,
                    config.encoder.pool_type,
                    config.encoder.pool_config
                ),
            ]
        )
        '''
        GNNSequential 将gnn_layers以及一个池化层组合为一个顺序网络。
        make_graph_pooling 创建图池化层，用于对图结构数据进行降维。
        '''
        self.qz_mean = tk.layers.Linear(output_size, config.z_dim)
        self.qz_logstd = tk.layers.Linear(output_size, config.z_dim)

        if config.encoder.use_posterior_flow:
            self.qz_flow = make_realnvp_flow(config.z_dim, config.realnvp)

        # ====
        # p(z)
        # ====
        if config.decoder.use_prior_flow:
            self.pz_flow = make_realnvp_flow(config.z_dim, config.realnvp).invert()

        # ===============
        # p(node_count|z)
        # ===============
        node_count_builder = tk.layers.SequentialBuilder(
            config.z_dim,
            layer_args=layer_args
        )
        for size in config.decoder.node_count_layers:
            node_count_builder.dense(size)
        self.pG_node_count_logits = node_count_builder. \
            linear(MAX_NODE_COUNT + 1). \
            build(flatten_to_ndims=True)

        # ========
        # p(adj|z)
        # ========
        # graph embedding from z
        graph_embedding_builder = tk.layers.SequentialBuilder(
            config.z_dim,
            layer_args=layer_args,
        )
        for size in config.decoder.graph_embedding_layers:
            graph_embedding_builder.dense(size)
        self.pG_graph_embedding = graph_embedding_builder.build(flatten_to_ndims=True)

        # node embedding (akka, `e`) from the graph embedding
        self.pG_node_embedding = tk.layers.Linear(
            graph_embedding_builder.out_shape[-1],
            MAX_NODE_COUNT * config.decoder.latent_embedding_size,
        )
        '''
        self.pG_node_embedding 是线性层，用于从图嵌入生成每个节点的嵌入。
        输出形状为MAX_NODE_COUNT * latent_embedding_size，表示所有节点的嵌入。
        '''

        # note: p(adj) = outer-dot(e)

        # ==================
        # p(node_type|e,adj)
        # ==================
        if config.decoder.z_as_context:
            input_size = (
                config.z_dim +
                int(config.decoder.use_idx) * MAX_NODE_COUNT +  # node_idx
                int(config.decoder.use_depth) * (MAX_SPAN_COUNT + 1)  # node_depth
            )
        else:
            input_size = config.decoder.latent_embedding_size

        output_size, gnn_layers = make_gnn_layers(
            config.decoder.gnn,
            input_size,
            config.decoder.gnn_layers,
        )
        self.pG_node_type_logits = GNNSequential(
            gnn_layers +
            [
                GraphConv(output_size, self.num_operations),  # p(node_type|e)
            ]
        )

    def _is_attr_included_in_repr(self, attr: str, value: Any) -> bool:
        if attr == 'config':
            return False
        return super()._is_attr_included_in_repr(attr, value)

    def q(self,
          net: tk.BayesianNet,
          g: dgl.DGLGraph,
          n_z: Optional[int] = None):
        config = self.config

        # embedding lookup
        h = self.operation_embedding(g.ndata['node_type'])

        # feed into gnn and get node embeddings
        h = self.qz_gnn_layers(g, h)

        # mean and logstd for q(z|G)
        z_mean = T.maybe_clip(
            self.qz_mean(h),
            # min_val=-5,
            # max_val=5,
        )
        z_logstd = T.maybe_clip(
            self.qz_logstd(h),
            min_val=config.encoder.z_logstd_min,
            max_val=config.encoder.z_logstd_max,
        )
        # 定义了一个正态分布（高斯分布）
        # add 'z' random variable
        qz = tk.Normal(mean=z_mean, logstd=z_logstd, event_ndims=1)
        if config.encoder.use_posterior_flow:
            qz = tk.FlowDistribution(qz, self.qz_flow)
        z = net.add('z', qz, n_samples=n_z)

    def p(self,
          net: tk.BayesianNet,
          g: Optional[dgl.DGLGraph] = None,
          n_z: Optional[int] = None,
          use_biased: bool = False):
        config = self.config

        # sample z ~ p(z)
        pz = tk.UnitNormal([1, config.z_dim], event_ndims=1)
        if config.decoder.use_prior_flow:
            pz = tk.FlowDistribution(pz, self.pz_flow)
        z = net.add('z', pz, n_samples=n_z)

        # p(node_count|z)
        node_count_logits = self.pG_node_count_logits(z.tensor)
        if use_biased:
            p_node_count = BiasedCategorical(
                alpha=MAX_NODE_COUNT * MAX_NODE_COUNT,
                threshold=0.5,
                logits=node_count_logits,
            )
        else:
            p_node_count = tk.Categorical(logits=node_count_logits)
        node_count = net.add('node_count', p_node_count)

        # graph embedding
        h = z.tensor
        h = self.pG_graph_embedding(h)
        h = self.pG_node_embedding(h)
        h = T.reshape(
            h,
            T.shape(h)[:-1] + [
                MAX_NODE_COUNT,
                config.decoder.latent_embedding_size
            ]
        )

        # p(A|e)
        edge_logits = edge_logits_by_dot_product(h)
        edge_logits = dense_to_triu(edge_logits, MAX_NODE_COUNT)

        if use_biased:
            p_adj = BiasedBernoulli(
                alpha=MAX_NODE_COUNT,
                threshold=0.5,
                logits=edge_logits,
                event_ndims=1,
            )
        else:
            p_adj = tk.Bernoulli(logits=edge_logits, event_ndims=1)

        adj = net.add('adj', p_adj)

        if g is None:
            # construct the `g` from the `adj`, assuming full MAX_NODE_COUNT adj
            def make_graph(triu_adj):
                adj = triu_to_dense(triu_adj, MAX_NODE_COUNT)

                # make graph
                u, v = T.where(adj)
                g = dgl.graph((u, v), num_nodes=MAX_NODE_COUNT)
                g = dgl.add_reverse_edges(g)
                g = dgl.add_self_loop(g)

                # make `node_idx`
                node_idx = T.maximum(
                    T.reduce_max(
                        (adj * torch.cumsum(adj, dim=-1)),
                        axis=[-2]
                    ) - 1,
                    T.int_scalar(0, dtype=T.int64),
                )
                g.ndata['node_idx'] = node_idx
                return g

            adj_shape = T.shape(adj.tensor)
            if len(adj_shape) == 3:
                g = [
                    dgl.batch([
                        make_graph(adj.tensor[i, j])
                        for j in range(adj_shape[1])
                    ])
                    for i in range(adj_shape[0])
                ]
            elif len(adj_shape) == 2:
                g = dgl.batch([
                    make_graph(adj.tensor[i])
                    for i in range(adj_shape[0])
                ])
            else:
                raise RuntimeError(f'Unsupported adj.shape: {adj_shape}')

        else:
            # expand the node_count of each graph to MAX_NODE_COUNT
            sub_graphs = []
            for sub_g in dgl.unbatch(g):
                # struct
                sub_u, sub_v = sub_g.edges()
                sub_node_idx = sub_g.ndata['node_idx']
                mask = sub_u < sub_v
                sub_u = sub_u[mask]
                sub_v = sub_v[mask]
                sub_g = dgl.graph((sub_u, sub_v), num_nodes=MAX_NODE_COUNT)
                sub_g = dgl.add_reverse_edges(sub_g)
                sub_g = dgl.add_self_loop(sub_g)

                # feature
                if sub_node_idx.shape[0] < MAX_NODE_COUNT:
                    sub_node_idx = T.concat(
                        [
                            sub_node_idx,
                            T.zeros([MAX_NODE_COUNT - sub_node_idx.shape[0]], dtype=T.int64)
                        ],
                        axis=0
                    )
                sub_g.ndata['node_idx'] = sub_node_idx

                # add this graph
                sub_graphs.append(sub_g)
            g = dgl.batch(sub_graphs)

        net.meta['g'] = g

        # p(node_type|e)
        if config.decoder.z_as_context:
            # z as context
            z_shape = T.shape(z.tensor)
            h = T.repeat(
                T.reshape(z.tensor, z_shape[:-1] + [1, z_shape[-1]]),
                [1] * (len(z_shape) - 1) + [MAX_NODE_COUNT, 1]
            )

            # h = []
            # for i, node_count in enumerate(g.batch_num_nodes()):
            #     h.append(
            #         T.repeat(
            #             z.tensor[..., i: i+1, :],
            #             [1] * (len(h_shape) - 1) + [int(T.to_numpy(node_count)), 1],
            #         )
            #     )
            # h = T.concat(h, axis=-2)

            # node_depth and node_idx
            h2 = decoder_use_depth_and_idx(
                g,
                config.decoder.use_depth,
                config.decoder.use_idx,
            )
            if h2 is not None:
                h = T.broadcast_concat(h, h2, axis=-1)

        h_shape = T.shape(h)
        h = T.reshape(
            h,
            h_shape[:-3] + [h_shape[-3] * h_shape[-2], h_shape[-1]]
        )

        node_type_logits = self.pG_node_type_logits(g, h)
        node_type_logits = T.reshape(node_type_logits, h_shape[:-1] + [self.num_operations])

        # if use_biased:
        #     p_node_type = BiasedCategorical(
        #         alpha=MAX_NODE_COUNT,
        #         threshold=0.5,
        #         logits=node_type_logits,
        #         event_ndims=1
        #     )
        # else:
        #     p_node_type = tk.Categorical(logits=node_type_logits, event_ndims=1)

        p_node_type = tk.Categorical(logits=node_type_logits, event_ndims=1)
        node_type = net.add('node_type', p_node_type)

'''
TraceStructVAE类
TraceStructVAE类实现了VAE模型的核心架构，包括编码器q(z|G)和解码器p(G|z)。模型结构如下：

初始化：在初始化中，将配置、操作嵌入等保存到实例中，并配置标准层属性、编码器和解码器的GNN层、图嵌入层等。

q(z|G)（编码器）：

使用图神经网络（GNN）从图中提取节点嵌入。
使用池化操作对GNN的输出进行图级别的聚合，获得编码器输出的均值和方差（qz_mean和qz_logstd）。
若启用了后验流（posterior flow），则进一步使用RealNVPFlow将q(z|G)的分布变换为一个复杂分布。
p(G|z)（解码器）：

生成潜在变量z的先验分布p(z)。如果使用先验流（prior flow），则流会作用在p(z)上。
生成节点数量的分布p(node_count|z)，通过一个分类分布（Categorical或BiasedCategorical）来建模。
根据潜在变量z，通过一系列线性层构建图嵌入。
生成邻接矩阵的分布p(adj|e)，通过节点嵌入的点积计算出边的存在性分布。
最后，p(node_type|e, adj)基于图嵌入生成节点类型的分布。
3. 编码器和解码器的细节
编码器 q方法
嵌入查找：获取每个节点类型的嵌入向量。
GNN传递：通过GNN层计算节点嵌入。
均值和方差：输出潜在变量的均值和方差，并通过流模型生成潜在变量的后验分布。
解码器 p方法
采样z：从标准正态分布采样或通过先验流变换生成z。
节点数量：通过分类分布预测节点数量。
邻接矩阵：计算每对节点的边的概率，并生成上三角形式的邻接矩阵。
节点类型：基于节点嵌入和邻接矩阵生成节点类型分布。
'''
'''
MAX_NODE_COUNT 的影响
pG_node_count_logits：
在生成节点数的概率分布时，MAX_NODE_COUNT + 1 被用作输出大小，意味着模型预计最多可能会有 MAX_NODE_COUNT 个节点。
此时，pG_node_count_logits 的输出 logits 表示每个节点数量的概率分布。
如果 trace 中的节点数少于 MAX_NODE_COUNT，模型会在推理时使用 MAX_NODE_COUNT 的可能性，这个设置确保了模型在生成图时有一致的节点数
。
pG_node_embedding：
MAX_NODE_COUNT * latent_embedding_size 作为输出尺寸，用来表示所有节点的嵌入。
即使 trace 中的节点数量小于 MAX_NODE_COUNT，模型仍然会生成一个包含 MAX_NODE_COUNT 个节点嵌入的张量。

pG_node_type_logits：
MAX_NODE_COUNT 影响 node_type_logits 的生成。即使 trace 中的节点数不足，模型仍然会生成大小为 MAX_NODE_COUNT 的节点类型预测结果。
这个设计允许模型处理不同规模的图，并且使得模型的输出在结构上是一致的。

图的构造：
在 p 方法中，如果输入图（g）为 None，模型会基于 adj 创建一个包含 MAX_NODE_COUNT 个节点的图。
即使实际的节点数少于 MAX_NODE_COUNT，也会将节点数补充至 MAX_NODE_COUNT。
'''