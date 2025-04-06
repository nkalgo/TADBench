import pickle
import time
from itertools import chain
from pprint import pprint
from tempfile import TemporaryDirectory

import os
import mltk
import click
import pandas as pd
import tensorkit as tk
import numpy as np
from tensorkit import tensor as T
from tensorkit.examples.utils import print_experiment_summary

from tracegnn.data import *
from tracegnn.models.trace_vae.dataset import TraceGraphDataStream
from tracegnn.models.trace_vae.evaluation import compute_span_nll
# from tracegnn.models.trace_vae.evaluation import *
from tracegnn.models.trace_vae.graph_utils import *
from tracegnn.models.trace_vae.test_utils import *
from tracegnn.models.trace_vae.types import TraceGraphBatch
from tracegnn.utils import *
from tracegnn.utils.fscore_utils import new_fscore

from .model import TraceVAE
from typing import *
from tqdm import tqdm
from .graph_utils import p_net_to_trace_graphs, trace_graph_key
from .tensor_utils import *
import math

@click.group()
def main():
    pass

# _dataset = "total_train_ratio_none"


def do_evaluate_nll(test_stream: mltk.DataStream,
                    vae: TraceVAE,
                    _dataset: str,
                    id_manager: TraceGraphIDManager,
                    latency_range: TraceGraphLatencyRangeFile,
                    n_z: int,
                    use_biased: bool = True,
                    use_latency_biased: bool = True,
                    no_latency: bool = False,
                    no_struct: bool = False,
                    std_limit: Optional[T.Tensor] = None,
                    latency_log_prob_weight: bool = False,
                    latency_logstd_min: Optional[float] = None,
                    test_threshold: Optional[float] = None,
                    test_loop=None,
                    summary_writer=None,
                    clip_nll=None,
                    use_embeddings: bool = False,
                    num_embedding_samples=None,
                    nll_output_file=None,
                    proba_cdf_file=None,
                    auc_curve_file=None,
                    latency_hist_file=None,
                    operation_id_dict_out=None,  # corresponding to latency_std_dict_out
                    latency_std_dict_out=None,
                    latency_reldiff_dict_out=None,
                    p_node_count_dict_out=None,
                    p_edge_dict_out=None,
                    latency_dict_prefix='',
                    ):
    # check params
    if std_limit is not None:
        std_limit = T.as_tensor(std_limit, dtype=T.float32)

    # result buffer
    nll_list = []
    label_list = []
    trace_id_list = []
    graph_key_list = []

    # span_nll_list = []
    span_label_list = []

    z_buffer = []  # the z embedding buffer of the graph
    z2_buffer = []  # the z2 embedding buffer of the graph
    z_label = []  # the label for z and z2
    latency_samples = {}
    result_dict = {}

    if operation_id_dict_out is not None:
        for key in ('normal', 'drop', 'latency', 'both'):
            if key not in operation_id_dict_out:
                operation_id_dict_out[latency_dict_prefix + key] = ArrayBuffer(81920, dtype=np.int)

    if latency_std_dict_out is not None:
        for key in ('normal', 'drop', 'latency', 'both'):
            if key not in latency_std_dict_out:
                latency_std_dict_out[latency_dict_prefix + key] = ArrayBuffer(81920)

    if latency_reldiff_dict_out is not None:
        for key in ('normal', 'drop', 'latency', 'both'):
            if key not in latency_reldiff_dict_out:
                latency_reldiff_dict_out[latency_dict_prefix + key] = ArrayBuffer(81920)

    if p_node_count_dict_out is not None:
        for key in ('normal', 'drop', 'latency', 'both'):
            if key not in p_node_count_dict_out:
                p_node_count_dict_out[latency_dict_prefix + key] = ArrayBuffer(81920)

    if p_edge_dict_out is not None:
        for key in ('normal', 'drop', 'latency', 'both'):
            if key not in p_edge_dict_out:
                p_edge_dict_out[latency_dict_prefix + key] = ArrayBuffer(81920)

    def add_embedding(buffer, label, tag, limit=None):
        if limit is not None:
            indices = np.arange(len(buffer))
            np.random.shuffle(indices)
            indices = indices[:limit]
            buffer = buffer[indices]
            label = label[indices]
        summary_writer.add_embedding(
            buffer,
            metadata=label,
            tag=tag,
        )

    eval_step_outside = 0

    # run evaluation
    def eval_step(trace_graphs: List[TraceGraph]):
        nonlocal eval_step_outside
        eval_step_outside_start1 = time.time()

        G = TraceGraphBatch(
            id_manager=id_manager,
            latency_range=latency_range,
            trace_graphs=trace_graphs,
        )
        '''
        .chain(...) 是一个方法，用于链接两个模型的推理过程，通常在变分推理（Variational Inference）中使用。
        它将 q 和 p 两个分布连接起来，形成一个完整的变分推理过程。'''
        chain = vae.q(G, n_z=n_z, no_latency=no_latency).chain(
            vae.p,
            latent_axis=0,
            G=G,
            use_biased=use_biased,
            use_latency_biased=use_latency_biased,
            no_latency=no_latency,
            latency_logstd_min=latency_logstd_min,
            latency_log_prob_weight=latency_log_prob_weight,
            std_limit=std_limit,
        )

        # step_span_nll_list = []
        step_span_label_list = []
        for i, trace_graph in enumerate(trace_graphs):
        #     span_latencys = list(trace_graph.get_span_latency().values())
        #     print(span_latencys)
            step_span_label_list.append(trace_graph.get_span_anomaly())
        #     for span_latency in span_latencys:
        #         # 计算每个 span 的 NLL，假设使用链的 p 分布
        #         span_nll = compute_span_nll(chain.p, span_latency)
        #         step_span_nll_list.append(span_nll)
        #     print(step_span_nll_list)

        # # 将 span 级别的 NLL 记录下来
        # step_span_nll_list = np.array(step_span_nll_list)
        # step_span_label_list = np.array(step_span_label_list)
        #
        # span_nll_list.extend(step_span_nll_list)
        span_label_list.extend(step_span_label_list)

        eval_step_outside += time.time() - eval_step_outside_start1

        if no_struct:
            q, p = chain.q, chain.p
            del q['z']
            del p['z']
            del p['adj']
            del p['node_count']
            del p['node_type']
            chain = q.chain(lambda *args, **kwargs: p, latent_axis=0)

        loss = chain.vi.training.sgvb()
        nll = -chain.vi.evaluation.is_loglikelihood()

        # clip the nll, and treat 'NaN' or 'Inf' nlls as `config.test.clip_nll`
        if clip_nll is not None:
            clip_limit = T.float_scalar(clip_nll)
            loss = T.where(loss < clip_limit, loss, clip_limit)
            nll = T.where(nll < clip_limit, nll, clip_limit)

        eval_step_outside_start2 = time.time()

        # the nlls and labels of this step
        step_label_list = np.array([
            0 if not g.data.get('is_anomaly') else (
                1 if g.data['anomaly_type'] == 'drop' else (
                    2 if g.data['anomaly_type'] == 'latency' else 3))
            for g in trace_graphs
        ])

        # Load the graph_key
        step_graph_key_list = [trace_graph_key(g) for g in trace_graphs]
        step_trace_id_list = [g.trace_id for g in trace_graphs]

        if not no_struct:
            # collect operation id
            if operation_id_dict_out is not None:
                collect_operation_id(operation_id_dict_out[f'{latency_dict_prefix}normal'], chain, step_label_list == 0)
                collect_operation_id(operation_id_dict_out[f'{latency_dict_prefix}drop'], chain, step_label_list == 1)
                collect_operation_id(operation_id_dict_out[f'{latency_dict_prefix}latency'], chain, step_label_list == 2)
                collect_operation_id(operation_id_dict_out[f'{latency_dict_prefix}both'], chain, step_label_list == 3)

            # collect latency
            if latency_std_dict_out is not None:
                collect_latency_std(latency_std_dict_out[f'{latency_dict_prefix}normal'], chain, step_label_list == 0)
                collect_latency_std(latency_std_dict_out[f'{latency_dict_prefix}drop'], chain, step_label_list == 1)
                collect_latency_std(latency_std_dict_out[f'{latency_dict_prefix}latency'], chain, step_label_list == 2)
                collect_latency_std(latency_std_dict_out[f'{latency_dict_prefix}both'], chain, step_label_list == 3)

            # collect relative diff
            if latency_reldiff_dict_out is not None:
                collect_latency_reldiff(latency_reldiff_dict_out[f'{latency_dict_prefix}normal'], chain, step_label_list == 0)
                collect_latency_reldiff(latency_reldiff_dict_out[f'{latency_dict_prefix}drop'], chain, step_label_list == 1)
                collect_latency_reldiff(latency_reldiff_dict_out[f'{latency_dict_prefix}latency'], chain, step_label_list == 2)
                collect_latency_reldiff(latency_reldiff_dict_out[f'{latency_dict_prefix}both'], chain, step_label_list == 3)

            # collect p node count
            if p_node_count_dict_out is not None:
                collect_p_node_count(p_node_count_dict_out[f'{latency_dict_prefix}normal'], chain, step_label_list == 0)
                collect_p_node_count(p_node_count_dict_out[f'{latency_dict_prefix}drop'], chain, step_label_list == 1)
                collect_p_node_count(p_node_count_dict_out[f'{latency_dict_prefix}latency'], chain, step_label_list == 2)
                collect_p_node_count(p_node_count_dict_out[f'{latency_dict_prefix}both'], chain, step_label_list == 3)

            # collect p edge
            if p_edge_dict_out is not None:
                collect_p_edge(p_edge_dict_out[f'{latency_dict_prefix}normal'], chain, step_label_list == 0)
                collect_p_edge(p_edge_dict_out[f'{latency_dict_prefix}drop'], chain, step_label_list == 1)
                collect_p_edge(p_edge_dict_out[f'{latency_dict_prefix}latency'], chain, step_label_list == 2)
                collect_p_edge(p_edge_dict_out[f'{latency_dict_prefix}both'], chain, step_label_list == 3)

            # inspect the internals of every trace graph
            if 'latency' in chain.p:
                p_latency = chain.p['latency'].distribution.base_distribution
                p_latency_mu, p_latency_std = p_latency.mean, p_latency.std
                if len(T.shape(p_latency.mean)) == 4:
                    p_latency_mu = p_latency_mu[0]
                    p_latency_std = p_latency_std[0]

                latency_sample = T.to_numpy(T.random.normal(p_latency_mu, p_latency_std))

                for i, tg in enumerate(trace_graphs):
                    assert isinstance(tg, TraceGraph)
                    if step_label_list[i] == 0:
                        for j in range(tg.node_count):
                            node_type = int(T.to_numpy(G.dgl_graphs[i].ndata['node_type'][j]))
                            if node_type not in latency_samples:
                                latency_samples[node_type] = []
                            mu, std = latency_range[node_type]
                            latency_samples[node_type].append(latency_sample[i, j, 0] * std + mu)

            if use_embeddings:
                for i in range(len(trace_graphs)):
                    if step_label_list[i] == 0:
                        node_type = trace_graphs[i].root.operation_id
                        node_label = id_manager.operation_id.reverse_map(node_type)
                        z_label.append(node_label)
                        z_buffer.append(T.to_numpy(chain.q['z'].tensor[0, i]))
                        if 'z2' in chain.q:
                            z2_buffer.append(T.to_numpy(chain.q['z2'].tensor[0, i]))

        # memorize the outputs
        nll_list.extend(T.to_numpy(nll))
        label_list.extend(step_label_list)
        trace_id_list.extend(step_trace_id_list)
        graph_key_list.extend(step_graph_key_list)

        # return a dict of the test result
        ret = {}
        normal_losses = T.to_numpy(loss)[step_label_list == 0]
        if len(normal_losses) > 0:
            test_loss = np.nanmean(normal_losses)
            if not math.isnan(test_loss):
                ret['loss'] = test_loss

        eval_step_outside += time.time() - eval_step_outside_start2

        return ret

    eval_step_all_start = time.time()
    with T.no_grad():
        # run test on test set
        if test_loop is not None:
            with test_loop.timeit('eval_time'):
                r = test_loop.run(eval_step, test_stream)
                if 'loss' in r:
                    r['test_loss'] = r['loss']
                if 'test_loss' in r:
                    result_dict['test_loss'] = r['test_loss']
        else:
            test_losses = []
            test_weights = []
            for [trace_graphs] in tqdm(test_stream, total=test_stream.batch_count):
                r = eval_step(trace_graphs)
                if 'loss' in r:
                    test_losses.append(r['loss'])
                    test_weights.append(len(trace_graphs))
            test_weights = np.asarray(test_weights)
            result_dict['test_loss'] = np.sum(
                np.asarray(test_losses) *
                (test_weights / np.sum(test_weights))
            )

        eval_step_all = time.time() - eval_step_all_start
        print('eval_step_all', eval_step_all)
        print('eval_step_outside', eval_step_outside)
        print('eval_step_inside', eval_step_all - eval_step_outside)

        if not os.path.exists(f'results_{_dataset}'):
            os.mkdir(f'results_{_dataset}')
        with open(f"results_{_dataset}/scores_{_dataset}.pkl", 'bw') as f:
            pickle.dump(nll_list, f)
        with open(f"results_{_dataset}/labels_{_dataset}.pkl", 'bw') as f:
            pickle.dump(label_list, f)

        print('len(nll_list)', len(nll_list))
        print('len(span_label_list)', len(span_label_list))
        # print('len(span_label_list[0])', len(span_label_list[0]))
        # print('len(span_label_list[1])', len(span_label_list[1]))
        # print('span_label_list[2]', span_label_list[2])
        # print('span_label_list[3]', span_label_list[3])

        span_nll_list = [[nll_list[i]] * len(span_label_list[i]) for i in range(len(nll_list))]
        # print('nll_list[:4]', nll_list[:4])
        # print('span_nll_list[0]', span_nll_list[0])
        # print('span_nll_list[2]', span_nll_list[2])
        # print('span_nll_list[3]', span_nll_list[3])
        flattened_span_nll_list = list(chain(*span_nll_list))
        flattened_span_label_list = list(chain(*span_label_list))
        print('len(flattened_span_nll_list)', len(flattened_span_nll_list))
        print('len(flattened_span_label_list)', len(flattened_span_label_list))
        flattened_span_nll_list = np.asarray(flattened_span_nll_list)
        flattened_span_label_list = np.asarray(flattened_span_label_list)

        # save the evaluation results
        nll_list = np.asarray(nll_list)
        label_list = np.asarray(label_list)
        graph_key_list = np.asarray(pickle.dumps(graph_key_list))

        # span_nll_list = np.asarray(span_nll_list)


        # label_drop_list = []
        # label_latency_list = []
        #
        # for i in range(len(label_list)):
        #     if label_list[i] != 1:
        #         nll_latency_list.append(nll_list[i])
        #
        #
        # print(len(nll_drop_list), len(nll_latency_list))
        return nll_list, label_list, flattened_span_nll_list, flattened_span_label_list
        # return nll_list, label_list, span_nll_list, span_label_list


def analyze_anomaly_nll(nll_list: np.ndarray,
                        label_list: np.ndarray,
                        span_nll_list: np.ndarray,
                        span_label_list: np.ndarray,
                        save_path: str,
                        method: Optional[str] = None,
                        dataset: Optional[str] = None,
                        save_dict: bool = False,
                        ) -> Dict[str, float]:
    # prepare for analyze
    result_dict = {}
    is_anomaly_list = label_list != 0

    is_span_anomaly_list = span_label_list != 0

    # drop_anomaly_list = label_list == 1
    # latency_anomaly_list = label_list == 2
    #
    # # nll_drop
    # nll_drop_normal = float(np.mean(nll_drop[label_list == 0]))
    # nll_drop_anomaly = float(np.mean(nll_drop[label_list == 1]))
    #
    # # nll_latency
    # nll_latency_normal = float(np.mean(nll_latency[label_list == 0]))
    # nll_latency_anomaly = float(np.mean(nll_latency[label_list == 2]))

    # separated nlls for different labels
    # nll_normal is not right, just for test.
    # result_dict['nll_normal'] = nll_drop_normal
    # result_dict['nll_drop'] = nll_drop_anomaly
    # result_dict['nll_latency'] = nll_latency_anomaly

    result_dict['nll_normal'] = float(np.mean(nll_list[label_list == 0]))
    result_dict['nll_drop'] = float(np.mean(nll_list[label_list == 1]))
    result_dict['nll_latency'] = float(np.mean(nll_list[label_list == 2]))
    result_dict['nll_p99'] = float(np.quantile(nll_list, 0.99))

    # drop and latency auc score
    # drop_auc = float(auc_score(nll_drop, drop_anomaly_list))
    # latency_auc = float(auc_score(nll_drop, latency_anomaly_list))
    # # This auc value is not right, just for test.
    # result_dict['auc'] = (drop_auc + latency_auc) / 2

    # auc score
    result_dict['auc'] = float(auc_score(nll_list, is_anomaly_list))

    # Find threshold
    # (best_fscore_drop, thresh_drop, precision_drop, recall_drop,
    #  TP_drop, TN_drop, FN_drop, FP_drop,
    #  p_drop, r_drop, f1_drop, acc_drop) = best_fscore(
    #     nll_drop, drop_anomaly_list)
    #
    # (best_fscore_latency, thresh_latency, precision_latency, recall_latency,
    #  TP_latency, TN_latency, FN_latency, FP_latency,
    #  p_latency, r_latency, f1_latency, acc_latency) = best_fscore(
    #     nll_latency, latency_anomaly_list)
    #
    # # Total f1_score (anomaly or not, if one is anomaly, then it is anomaly)
    # nll_list = (nll_drop > thresh_drop) | (nll_latency > thresh_latency)
    # (best_fscore_total, threshold, precision, recall, TP, TN, FN, FP,
    #  p, r, f1, acc) = best_fscore(nll_list.astype(np.float32), is_anomaly_list)

    (best_fscore_total, threshold, precision, recall, TP, TN, FN, FP,
     p, r, f1, acc) = best_fscore(nll_list, is_anomaly_list)

    (best_span_fscore_total, span_threshold, span_precision, span_recall, span_TP, span_TN, span_FN, span_FP,
     span_p, span_r, span_f1, span_acc) = best_fscore(span_nll_list, is_span_anomaly_list)

    # (best_span_fscore_new, span_threshold_new, span_precision_new, span_recall_new, span_TP_new, span_TN_new, span_FN_new, span_FP_new,
    #  span_p_new, span_r_new, span_f1_new, span_acc_new) = new_fscore(span_nll_list, is_span_anomaly_list)

    mask_drop = (label_list != 2) & (label_list != 3)
    (best_fscore_drop, thresh_drop, precision_drop, recall_drop,
     TP_drop, TN_drop, FN_drop, FP_drop,
     p_drop, r_drop, f1_drop, acc_drop) = best_fscore(nll_list[mask_drop], label_list[mask_drop] != 0)

    mask_latency = (label_list != 1) & (label_list != 3)
    (best_fscore_latency, thresh_latency, precision_latency, recall_latency,
     TP_latency, TN_latency, FN_latency, FP_latency,
     p_latency, r_latency, f1_latency, acc_latency) = best_fscore(nll_list[mask_latency], label_list[mask_latency] != 0)

    # (best_fscore_node_latency, thresh_node_latency, precision_node_latency, recall_node_latency,
    #  TP_node_latency, TN_node_latency, FN_node_latency, FP_node_latency,
    #  p_node_latency, r_node_latency, f1_node_latency, acc_node_latency) = best_fscore(node_nll_list, is_node_anomaly_list)

    result_dict.update({
        # 'best_fscore': round(float(best_fscore_total), 6),
        # 'precision': round(float(precision), 6),
        # 'recall': round(float(recall), 6),
        'TP': TP,
        'TN': TN,
        'FN': FN,
        'FP': FP,
        'p': round(p, 6),
        'r': round(r, 6),
        'f1': round(f1, 6),
        'acc': round(acc, 6),

        # 'best_fscore_drop': round(float(best_fscore_drop), 6),
        # 'precision_drop': round(float(precision_drop), 6),
        # 'recall_drop': round(float(recall_drop), 6),
        'TP_drop': TP_drop,
        'TN_drop': TN_drop,
        'FN_drop': FN_drop,
        'FP_drop': FP_drop,
        'p_drop': round(p_drop, 6),
        'r_drop': round(r_drop, 6),
        'f1_drop': round(f1_drop, 6),
        'acc_drop': round(acc_drop, 6),

        # 'best_fscore_latency': round(float(best_fscore_latency), 6),
        # 'precision_latency': round(float(precision_latency), 6),
        # 'recall_latency': round(float(recall_latency), 6),
        'TP_latency': TP_latency,
        'TN_latency': TN_latency,
        'FN_latency': FN_latency,
        'FP_latency': FP_latency,
        'p_latency': round(p_latency, 6),
        'r_latency': round(r_latency, 6),
        'f1_latency': round(f1_latency, 6),
        'acc_latency': round(acc_latency, 6),

        'span_TP': span_TP,
        'span_TN': span_TN,
        'span_FN': span_FN,
        'span_FP': span_FP,
        'span_p': round(span_p, 6),
        'span_r': round(span_r, 6),
        'span_f1': round(span_f1, 6),
        'span_acc': round(span_acc, 6),

        # 'span_TP_new': span_TP_new,
        # 'span_TN_new': span_TN_new,
        # 'span_FN_new': span_FN_new,
        # 'span_FP_new': span_FP_new,
        # 'span_p_new': round(span_p_new, 6),
        # 'span_r_new': round(span_r_new, 6),
        # 'span_f1_new': round(span_f1_new, 6),
        # 'span_acc_new': round(span_acc_new, 6),
    })

    if save_dict and method and dataset:
        # dataset = dataset.rstrip('/')

        result_to_save = result_dict.copy()
        result_to_save['dataset'] = dataset
        result_to_save['method'] = method

        if os.path.exists(save_path):
            df = pd.read_csv(save_path)

            if not df[(df['dataset'] == dataset) & (df['method'] == method)].empty:
                df.iloc[df[(df['dataset'] == dataset) & (df['method'] == method)].index[0]] = result_to_save
            else:
                df = df.append(result_to_save, ignore_index=True)
        else:
            df = pd.DataFrame()
            df = df.append(result_to_save, ignore_index=True)

        # os.makedirs('lstm_res', exist_ok=True)
        df.to_csv(save_path, index=False)

    return result_dict


@main.command(context_settings=dict(
    ignore_unknown_options=True,
    help_option_names=[],
))
@click.option('-D', '--data-dir', required=False)
@click.option('-M', '--model-path', required=True)
@click.option('-N', '--dataset-name', required=True)
@click.option('-o', '--nll-out', required=False, default=None)
@click.option('--proba-out', default=None, required=False)
@click.option('--auc-out', default=None, required=False)
@click.option('--latency-out', default=None, required=False)
@click.option('--gui', is_flag=True, default=False, required=False)
@click.option('--device', required=False, default=None)
@click.option('--n_z', type=int, required=False, default=10)
@click.option('--batch-size', type=int, default=128)
@click.option('--clip-nll', type=float, default=100_000)
@click.option('--no-biased', is_flag=True, default=False, required=False)
@click.option('--no-latency-biased', is_flag=True, default=False, required=False)
@click.option('--no-latency', is_flag=True, default=False, required=False)
@click.option('--use-train-val', is_flag=True, default=False, required=False)
@click.option('--infer-bias-std', is_flag=True, default=False, required=False)
@click.option('--bias-std-normal-p', type=float, default=0.995, required=False)
@click.option('--infer-threshold', is_flag=True, default=False, required=False)
@click.option('--threshold-p', type=float, default=0.995, required=False)
@click.option('--threshold-amplify', type=float, default=1.0, required=False)
@click.option('--no-latency-log-prob-weight', is_flag=True, default=False, required=False)
@click.option('--use-std-limit', is_flag=True, default=False, required=False)
@click.option('--std-limit-global', is_flag=True, default=False, required=False)
@click.option('--std-limit-fixed', type=float, default=None, required=False)
@click.option('--std-limit-p', type=float, default=0.99, required=False)
@click.option('--std-limit-amplify', type=float, default=1.0, required=False)
@click.argument('extra_args', nargs=-1, type=click.UNPROCESSED)
def evaluate_nll(data_dir, model_path, dataset_name, nll_out, proba_out, auc_out, latency_out, gui, device,
                 n_z, batch_size, clip_nll, no_biased, no_latency_biased, no_latency,
                 use_train_val, infer_bias_std, bias_std_normal_p, infer_threshold,
                 threshold_p, threshold_amplify, no_latency_log_prob_weight,
                 use_std_limit, std_limit_global, std_limit_fixed, std_limit_p, std_limit_amplify,
                 extra_args):
    _dataset = dataset_name
    N_LIMIT = None

    if infer_bias_std or infer_threshold or use_std_limit:
        use_train_val = True

    with mltk.Experiment(mltk.Config, args=[]) as exp:
        # check parameters
        if gui:
            proba_out = ':show:'
            auc_out = ':show:'
            latency_out = ':show:'

        with T.use_device(device or T.first_gpu_device()):
            # load the config
            train_config = load_config(
                model_path=model_path,
                strict=False,
                extra_args=extra_args,
            )
            if data_dir is None:
                data_dir = train_config.dataset.root_dir

            # load the dataset
            data_names = ['test', 'test-drop', 'test-latency', 'test-both']
            test_db, id_manager = open_trace_graph_db(
                data_dir,
                names=data_names
            )
            print('Test DB:', test_db)
            latency_range = TraceGraphLatencyRangeFile(
                id_manager.root_dir,
                require_exists=True,
            )
            test_stream = TraceGraphDataStream(
                test_db, id_manager=id_manager, batch_size=batch_size,
                shuffle=False, skip_incomplete=False, data_count=N_LIMIT,
            )

            # also load train / val
            if use_train_val:
                train_db, _ = open_trace_graph_db(
                    data_dir,
                    names=['train'],
                )
                print('Train DB:', train_db)
                val_db, _ = open_trace_graph_db(
                    data_dir,
                    names=['val']
                )
                print('Val DB:', val_db)
                train_stream = TraceGraphDataStream(
                    train_db, id_manager=id_manager, batch_size=batch_size,
                    shuffle=True, skip_incomplete=False, data_count=N_LIMIT,
                )
                val_stream = TraceGraphDataStream(
                    val_db, id_manager=id_manager, batch_size=batch_size,
                    shuffle=True, skip_incomplete=False, data_count=N_LIMIT,
                )
            else:
                train_stream = val_stream = None

            print_experiment_summary(exp, train_stream, val_stream, test_stream)

            # load the model
            vae = load_model2(
                model_path=model_path,
                train_config=train_config,
                id_manager=id_manager,
            )
            mltk.print_config(vae.config, title='Model Config')
            vae = vae.to(T.current_device())

            # do evaluation
            operation_id = {}
            latency_std = {}
            latency_reldiff = {}
            p_node_count = {}
            p_edge = {}
            nll_result = {}
            thresholds = {}
            std_group_limit = np.full([id_manager.num_operations], np.nan, dtype=np.float32)

            def F(stream, category, n_z, threshold=None, std_limit=None):
                # the save files kw
                kw = dict(
                    nll_output_file=ensure_parent_exists(nll_out),
                    proba_cdf_file=ensure_parent_exists(proba_out),
                    auc_curve_file=ensure_parent_exists(auc_out),
                    latency_hist_file=ensure_parent_exists(latency_out),
                )
                differ_set = set()

                for k in kw:
                    if kw[k] is not None:
                        s = kw[k].replace('test', category)
                        if category == 'test' or s != kw[k]:
                            differ_set.add(k)
                        kw[k] = s
                kw = {k: v for k, v in kw.items() if k in differ_set}

                # the output temp dir
                with TemporaryDirectory() as temp_dir:
                    if 'nll_output_file' not in kw:
                        kw['nll_output_file'] = ensure_parent_exists(
                            os.path.join(temp_dir, 'nll.npz')
                        )

                    s_time = time.time()
                    # do evaluation
                    nll_list, labels, span_nll_list, span_labels = do_evaluate_nll(
                        test_stream=stream,
                        vae=vae,
                        _dataset=dataset_name,
                        id_manager=id_manager,
                        latency_range=latency_range,
                        n_z=n_z,
                        use_biased=(not no_biased) and (category == 'test'),
                        use_latency_biased=not no_latency_biased,
                        no_latency=no_latency,
                        no_struct=False,
                        latency_log_prob_weight=not no_latency_log_prob_weight,
                        std_limit=std_limit,
                        test_threshold=threshold,
                        clip_nll=clip_nll,
                        use_embeddings=False,
                        operation_id_dict_out=operation_id,
                        latency_std_dict_out=latency_std,
                        p_node_count_dict_out=p_node_count,
                        p_edge_dict_out=p_edge,
                        latency_reldiff_dict_out=latency_reldiff,
                        latency_dict_prefix=f'{category}_',
                        **kw,
                    )
                    print('do_evaluate_nll_time', time.time() - s_time)
                    print(analyze_anomaly_nll(
                        nll_list=np.array(nll_list, dtype=np.float32),
                        label_list=np.array(labels, dtype=np.int32),
                        span_nll_list=span_nll_list,
                        span_label_list=span_labels,
                        method='TraceVAE',
                        dataset=_dataset,
                        save_dict=True,
                        save_path=f'results_{_dataset}/test-result.csv'
                    ))

            tk.layers.set_eval_mode(vae)
            with T.no_grad():
                start_time = time.time()
                F(test_stream, 'test', n_z)
                end_time = time.time()
                print("test time: {}s".format(end_time - start_time))

    # end_time = time.time()
    # print("test time: {}s".format(end_time-start_time))


if __name__ == '__main__':
    # _dataset='AIOps2022'

    main()
    # with open(f"results_{_dataset}/labels_{_dataset}.pkl", 'rb') as f:
    #     data = pickle.load(f)
    # print(len(data))
    # with open(f"results_{_dataset}/scores_{_dataset}.pkl", 'rb') as f:
    #     data = pickle.load(f)
    # print(len(data))

