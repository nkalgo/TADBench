from pprint import pprint
from tempfile import TemporaryDirectory

import mltk
import tensorkit as tk
from tensorkit import tensor as T
from tqdm import tqdm

import torch
import torch.nn as nn

from tracegnn.data import *
from tracegnn.models.trace_vae.dataset import TraceGraphDataStream
from tracegnn.models.trace_vae.evaluation import *
from tracegnn.models.trace_vae.graph_utils import *
from tracegnn.models.trace_vae.rca import *
from tracegnn.models.trace_vae.search_tree import *
from tracegnn.models.trace_vae.test_utils import *
from tracegnn.models.trace_vae.types import TraceGraphBatch
from tracegnn.utils import *
from tracegnn.visualize import *
from tracegnn.models.trace_vae.model import TraceVAE


def get_z(vae: TraceVAE,
          test_stream: TraceGraphDataStream,
          id_manager: TraceGraphIDManager,
          latency_range: TraceGraphLatencyRangeFile) -> torch.Tensor:
    def eval_step(trace_graphs):
        G = TraceGraphBatch(
            id_manager=id_manager,
            latency_range=latency_range,
            trace_graphs=trace_graphs,
        )
        net = vae.q(G)
        print(net)

    test_loop = mltk.TestLoop()
    with tk.layers.scoped_eval_mode(vae), T.no_grad():
        test_loop.run(eval_step, test_stream)


@click.command()
@click.option('-D', '--data-dir', required=True)
@click.option('-M', '--model-path', required=True)
@click.option('--device', required=False, default=None)
@click.option('--batch-size', type=int, default=128)
@click.option('--drop2', is_flag=True, default=False, required=False)
@click.option('--ltest', is_flag=True, default=False, required=False)
@click.argument('extra_args', nargs=-1, type=click.UNPROCESSED)
def main(data_dir, model_path, device, batch_size, drop2, ltest, extra_args):
    N_LIMIT = None

    with mltk.Experiment(mltk.Config, args=[]) as exp:
        # check parameters
        with T.use_device(device or T.first_gpu_device()):
            # load the dataset
            if ltest:
                data_names = ['test', 'ltest-drop', 'ltest-latency']
            else:
                if drop2:
                    data_names = ['test', 'test-drop-anomaly2', 'test-latency-anomaly2']
                else:
                    data_names = ['test', 'test-drop-anomaly', 'test-latency-anomaly2']
            test_db, id_manager = open_trace_graph_db(
                data_dir,
                names=data_names
            )
            latency_range = TraceGraphLatencyRangeFile(
                id_manager.root_dir,
                require_exists=True,
            )
            test_stream = TraceGraphDataStream(
                test_db, id_manager=id_manager, batch_size=batch_size,
                shuffle=True, skip_incomplete=False, data_count=N_LIMIT,
            )

            # load the model
            vae, train_config = load_model(
                model_path=model_path,
                id_manager=id_manager,
                strict=False,
                extra_args=extra_args,
            )
            mltk.print_config(vae.config, title='Model Config')
            vae = vae.to(T.current_device())

            get_z(vae, test_stream, id_manager, latency_range)


if __name__ == '__main__':
    main()
