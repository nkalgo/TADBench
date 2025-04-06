import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
from tracegnn.data.trace_graph import TraceGraphIDManager
from sklearn.manifold import TSNE
import os
import sys
import click

@click.command()
@click.option('-i', '--input-dir', default='dataset/resample/')
@click.option('-c', '--color-type')
def main(input_dir, color_type):
    assert color_type in ['node_cnt', 'root_service', None]

    # Define id_manager
    id_manager = TraceGraphIDManager(input_dir)

    # Load data
    Z = np.load('tracegnn/models/gvae_tf/result/z.npy')
    DV = np.load('tracegnn/models/gvae_tf/result/DV.npy')
    DE = np.load('tracegnn/models/gvae_tf/result/DE.npy')

    # TSNE for z
    print('TSNE training...')
    tsne = TSNE(n_components=2, verbose=1)
    z = tsne.fit_transform(Z)

    node_color = None

    if color_type == 'node_cnt':
        # Color
        node_cnt = np.sum(np.sum(DV, axis=-1), axis=-1)
        cmap = cm.get_cmap('hsv', np.max(node_cnt)+1)
        node_color = [cmap(int(i)) for i in node_cnt]
    elif color_type == 'root_service':
        root_service = np.argmax(DV[:,0,:], axis=-1)
        cmap = cm.get_cmap('hsv', np.max(root_service)+1)
        node_color = [cmap(int(i)) for i in root_service]
    else:
        node_color = None

    # Plot Cluster
    plt.scatter(z[:, 0], z[:, 1], c=node_color)
    plt.title(str(color_type))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
