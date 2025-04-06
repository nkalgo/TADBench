import json
import os
import time
from readdata import *
import pandas as pd
import tensorflow as tf
import tfsnippet as spt
from sklearn.model_selection import train_test_split
from tensorflow.contrib.framework import arg_scope, add_arg_scope
from tfsnippet.examples.utils import (print_with_title, collect_outputs)
from tfsnippet.ops import log_mean_exp
from MLConfig import (MLConfig, global_config as config, config_options)
from readdata import get_data_vae


class ExpConfig(MLConfig):
    debug_level = -1  # -1: disable all checks;
    #  0: assertions only
    #  1: assertions + numeric check

    # model parameters
    z_dim = 10
    x_dim = 100

    flow_type = None  # None: no flow
    # planar_nf
    # rnvp
    n_planar_nf_layers = 10
    n_rnvp_layers = 10
    n_rnvp_hidden_layers = 1

    # training parameters
    write_summary = False
    max_epoch = 200  # 原来 200
    max_step = None
    batch_size = 256

    l2_reg = 0.0001
    initial_lr = 0.001
    # l2_reg = 0.001
    # initial_lr = 0.01

    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 100
    lr_anneal_step_freq = None

    # evaluation parameters
    test_n_z = 500
    test_batch_size = 128

    norm_clip = 10


config = ExpConfig()


@spt.global_reuse
@add_arg_scope
def q_net(x, posterior_flow, observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)

    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = tf.to_float(x)
        h_x = spt.layers.dense(h_x, 500)
        h_x = spt.layers.dense(h_x, 500)

    # sample z ~ q(z|x)
    z_mean = spt.layers.dense(h_x, config.z_dim, name='z_mean')
    z_std = 1e-4 + tf.nn.softplus(
        spt.layers.dense(h_x, config.z_dim, name='z_std'))
    z = net.add('z', spt.Normal(mean=z_mean, std=z_std), n_samples=n_z,
                group_ndims=1, flow=posterior_flow)
    return net


@spt.global_reuse
@add_arg_scope
def p_net(observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)

    # sample z ~ p(z)
    z = net.add('z', spt.Normal(mean=tf.zeros([1, config.z_dim]),
                                logstd=tf.zeros([1, config.z_dim])),
                group_ndims=1, n_samples=n_z)

    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = z
        h_z = spt.layers.dense(h_z, 500)
        h_z = spt.layers.dense(h_z, 500)

    # sample x ~ p(x|z)
    x_mean = spt.layers.dense(h_z, config.x_dim, name='x_mean')
    x_std = 1e-4 + tf.nn.softplus(
        spt.layers.dense(h_z, config.x_dim, name='x_std'))
    x = net.add('x', spt.Normal(mean=x_mean, std=x_std), group_ndims=1)

    return net


def coupling_layer_shift_and_scale(x1, n2):
    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h = x1
        for _ in range(config.n_rnvp_hidden_layers):
            h = spt.layers.dense(h, 500)

    # compute shift and scale
    shift = spt.layers.dense(
        h, n2, kernel_initializer=tf.zeros_initializer(),
        bias_initializer=tf.zeros_initializer(), name='shift'
    )
    scale = spt.layers.dense(
        h, n2, kernel_initializer=tf.zeros_initializer(),
        bias_initializer=tf.zeros_initializer(), name='scale'
    )
    return shift, scale


def cal(dataset_name, filepath, anomaly_type):
    TP, FP, TN, FN = 0, 0, 0, 0
    scores = []
    error_trace_num = 0
    with open(filepath, 'r') as f:
        for line in f.readlines()[1:]:
            try:
                scores.append(float(line.strip().split(',')[2]))
            except:
                error_trace_num += 1
    scores.sort()
    threshold = scores[len(scores) // 100 * 15]
    original = []
    predict = []
    with open(filepath, 'r')as f:
        for info in f.readlines()[1:]:
            try:
                predict.append(0 if float(info.strip().split(',')[2]) > threshold else 1)
            except:
                continue
            original.append(int(info.strip().split(',')[1]))
    for i in range(len(original)):
        if original[i] is 1 and predict[i] is 1:
            TP += 1
        elif original[i] is 0 and predict[i] is 0:
            TN += 1
        elif original[i] is 0 and predict[i] is 1:
            FP += 1
        else:
            FN += 1
    precision = round(TP / (TP + FP), 4)
    recall = round(TP / (TP + FN), 4)
    F1_score = round(2 * precision * recall / (precision + recall), 4)
    ACC = round((TP + TN) / (TP + FN + FP + TN), 4)

    print("TP: {}".format(TP), end=', ')
    print("TN: {}".format(TN), end=', ')
    print("FP: {}".format(FP), end=', ')
    print("FN: {}".format(FN), end='\n')
    print("precision: {}".format(precision), end=', ')
    print("recall: {}".format(recall), end=', ')
    print("F1_score: {}".format(F1_score), end=', ')
    print('ACC:{}'.format(ACC))

    result_data = {}
    if os.path.exists('../results.json'):
        with open('../results.json', 'r') as f:
            result_data = json.load(f)

    dataset_results = result_data.setdefault(dataset_name, {})
    algorithm_results = dataset_results.setdefault('CRISP', {'total': {}, 'structure': {}, 'latency': {}})
    algorithm_results[anomaly_type].update({
        'p': precision,
        'r': recall,
        'f1': F1_score,
        'acc': ACC
    })

    with open('../results.json', 'w') as f:
        json.dump(result_data, f, indent=4)


def crisp_test(dataset_name, anomaly_type):
    if config.debug_level == -1:
        spt.utils.set_assertion_enabled(False)
    elif config.debug_level == 1:
        spt.utils.set_check_numerics(True)

    # print the config
    print_with_title('Configurations', config.format_config(), after='\n')

    train_file = os.path.join('data', '%s_SCPV' % dataset_name, 'train_normal')
    normal_file = os.path.join('data', '%s_SCPV' % dataset_name, 'test_normal')
    if anomaly_type == 'total':
        abnormal_file = os.path.join('data', '%s_SCPV' % dataset_name, 'abnormal')
    elif anomaly_type == 'structure':
        abnormal_file = os.path.join('data', '%s_SCPV' % dataset_name, 'only_structure_abnormal')
    elif anomaly_type == 'latency':
        abnormal_file = os.path.join('data', '%s_SCPV' % dataset_name, 'only_latency_abnormal')
    else:
        print('Please input correct anomaly_type.')
        exit()
    # ['only_latency_abnormal', 'only_structure_abnormal', 'abnormal']
    model_path = os.path.join('results', 'md_{}_result_{}.model'.format(config.flow_type or 'vae', dataset_name))
    output_file = os.path.join('results', 'test_{}_result_{}.csv'.format(config.flow_type or 'vae', dataset_name))
    (x_train, y_train), (x_test, y_test), flows_test = get_data_vae(train_file, normal_file, abnormal_file)
    config.x_dim = x_train.shape[1]
    input_x = tf.placeholder(dtype=tf.float32, shape=(None, config.x_dim), name='input_x')

    # build the posterior flow
    if config.flow_type is None:
        posterior_flow = None  # 后验流
    elif config.flow_type == 'planar_nf':
        posterior_flow = spt.layers.planar_normalizing_flows(config.n_planar_nf_layers)
    else:
        assert (config.flow_type == 'rnvp')
        with tf.variable_scope('posterior_flow'):
            flows = []
            for i in range(config.n_rnvp_layers):
                flows.append(spt.layers.ActNorm())
                flows.append(spt.layers.CouplingLayer(
                    tf.make_template(
                        'coupling',
                        coupling_layer_shift_and_scale,
                        create_scope_now_=True
                    ),
                    scale_type='sigmoid'
                ))
                flows.append(spt.layers.InvertibleDense(strict_invertible=True))
            posterior_flow = spt.layers.SequentialFlow(flows=flows)

    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, posterior_flow, n_z=config.test_n_z)
        test_chain = test_q_net.chain(p_net, latent_axis=0, observed={'x': input_x})
        test_logp = test_chain.vi.evaluation.is_loglikelihood()
        test_nll = -tf.reduce_mean(test_logp)
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())

    test_flow = spt.DataFlow.arrays([x_test], config.test_batch_size)

    with spt.utils.create_session().as_default() as session:
        var_dict = spt.utils.get_variables_as_dict()
        saver = spt.VariableSaver(var_dict, model_path)

        print('Loading trained model...')
        saver.restore()

        print('Getting testing results...')
        test_ans = collect_outputs([test_logp], [input_x], test_flow)[0] / config.x_dim
        pd.DataFrame({'id': flows_test, 'label': y_test, 'score': test_ans}).to_csv(output_file, index=False)
        cal(dataset_name, output_file, anomaly_type)
