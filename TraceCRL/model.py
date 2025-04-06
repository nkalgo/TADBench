import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch_geometric
from torch_geometric.nn import BatchNorm, GATConv, global_mean_pool, TransformerConv, CGConv, global_add_pool
import numpy as np
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, num_layers=2, input_dim=20, output_dim=20, num_edge_attr=25, pooling_type='mean'):
        super(Encoder, self).__init__()

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.num_layer = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling_type = pooling_type

        # CGConv
        conv_0 = CGConv(channels=input_dim, dim=num_edge_attr)
        conv_1 = CGConv(channels=input_dim, dim=num_edge_attr)
        bn_0 = BatchNorm(input_dim)
        bn_1 = BatchNorm(input_dim)
        if num_layers == 3:
            conv_2 = CGConv(channels=input_dim, dim=num_edge_attr)
            bn_2 = BatchNorm(input_dim)

        self.convs.append(conv_0)
        self.convs.append(conv_1)

        self.batch_norms.append(bn_0)
        self.batch_norms.append(bn_1)
        if num_layers == 3:
            self.convs.append(conv_2)
            self.batch_norms.append(bn_2)

    def forward(self, x, edge_index, edge_attr, batch):
        xs = []
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = batch_norm(conv(x, edge_index, edge_attr))
            xs.append(x)

        if self.pooling_type == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling_type == 'add':
            x = global_add_pool(x, batch)
        else:
            print('pooling type error')
            assert False

        return x

    def get_embeddings(self, dataloader):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        trace_ids = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                data = data.to(device)
                x = self.forward(data.x, data.edge_index, data.edge_attr, data.batch)
                trace_ids.extend(data.trace_id)
                ret.append(x.detach().cpu().numpy())
                y.append(data.y.detach().cpu().numpy())

        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y, trace_ids


class SIMCLR(nn.Module):
    def __init__(self, num_layers=2, input_dim=20, output_dim=20, num_edge_attr=25, pooling_type='mean'):
        super(SIMCLR, self).__init__()

        self.num_layer = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling_type = pooling_type

        self.encoder = Encoder(num_layers=num_layers, input_dim=input_dim, output_dim=output_dim,
                               num_edge_attr=num_edge_attr, pooling_type=pooling_type)

        self.proj_head = nn.Sequential(nn.Linear(output_dim, output_dim), nn.LeakyReLU(),
                                       nn.Linear(output_dim, output_dim))
        self.init_emb()

    def init_emb(self):  # 模型的权重被初始化为服从 Xavier 均匀分布的随机值。此外，偏置（bias）被初始化为全零。
        initrange = -1.5 / self.output_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_attr, batch):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y = self.encoder(x, edge_index, edge_attr, batch)
        y = self.proj_head(y)

        return y

    def loss_cal(self, x: torch.Tensor, x_aug: torch.Tensor):
        batch_size, feat_num = x.size()
        batch_size_aug, feat_num_aug = x_aug.size()
        min_batch_size = min(batch_size, batch_size_aug)
        x = x[:min_batch_size, range(feat_num)]
        x_aug = x_aug[:min_batch_size, range(feat_num_aug)]

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss
