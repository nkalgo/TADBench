import argparse
import shutil
import pandas as pd
from loguru import logger
from pathlib import Path
from typing import Dict, Union, List

from sdk_new.example import *


def main():
    parser = argparse.ArgumentParser(description="Unified Trace Anomaly Detection Framework")
    subparsers = parser.add_subparsers(dest='algorithm', help='Select algorithm')

    # 注册所有算法解析器
    register_traceanomaly(subparsers)
    register_crisp(subparsers)
    register_tracecrl(subparsers)
    register_putracead(subparsers)
    register_tracevae(subparsers)
    register_multimodal_lstm(subparsers)
    register_gtrace(subparsers)

    args = parser.parse_args()

    if not args.algorithm:
        parser.print_help()
        return

    # 路由到对应算法处理
    algorithm_handlers = {
        'traceanomaly': handle_traceanomaly,
        'crisp': handle_crisp,
        'tracecrl': handle_tracecrl,
        'putracead': handle_putracead,
        'tracevae': handle_tracevae,
        'lstm': handle_multimodal_lstm,
        'gtrace': handle_gtrace
    }

    if args.algorithm in algorithm_handlers:
        algorithm_handlers[args.algorithm](args)
    else:
        logger.error(f"Unsupported algorithm: {args.algorithm}")


def register_traceanomaly(subparsers):
    parser = subparsers.add_parser('traceanomaly', help='TraceAnomaly Algorithm')
    sub = parser.add_subparsers(dest='mode', required=True)

    # Train
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--train_path', type=Path, required=True)
    train_parser.add_argument('--output_path', type=Path, required=True)

    # Test
    test_parser = sub.add_parser('test')
    test_parser.add_argument('--test_normalpath', type=Path, required=True)
    test_parser.add_argument('--test_abnormalpath', type=Path, required=True)
    test_parser.add_argument('--output_path', type=Path, required=True)


def handle_traceanomaly(args):
    algo = TraceAnomaly()
    if args.mode == 'train':
        algo.train(
            train_path=args.train_path,
            output_path=args.output_path
        )
    elif args.mode == 'test':
        algo.test(
            test_normalpath=args.test_normalpath,
            test_abnormalpath=args.test_abnormalpath,
            output_path=args.output_path
        )


def register_crisp(subparsers):
    parser = subparsers.add_parser('crisp', help='CRISP algorithm')
    sub = parser.add_subparsers(dest='mode', required=True)

    # Preprocess
    pre_parser = sub.add_parser('preprocess')
    pre_parser.add_argument('--service_name', required=True)
    pre_parser.add_argument('--operation_name', required=True)
    pre_parser.add_argument('--trace_files', type=Path, nargs='+')
    pre_parser.add_argument('--root_trace', required=True)
    pre_parser.add_argument('--parallelism', type=int, default=4)

    # Train
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--train_path', type=Path, required=True)
    train_parser.add_argument('--test_nomalpath', type=Path, required=True)
    train_parser.add_argument('--test_abnormalpath', type=Path, required=True)
    train_parser.add_argument('--output_path', type=Path, required=True)

    # Test
    test_parser = sub.add_parser('test')
    test_parser.add_argument('--test_nomalpath', type=Path, required=True)
    test_parser.add_argument('--test_abnormalpath', type=Path, required=True)
    test_parser.add_argument('--output_path', type=Path, required=True)


def handle_crisp(args):
    algo = CRISP()
    if args.mode == 'preprocess':
        algo.preprocess(
            service_name=args.service_name,
            operation_name=args.operation_name,
            trace_files=args.trace_files,
            root_trace=args.root_trace,
            parallelism=args.parallelism
        )
    elif args.mode == 'train':
        algo.train(
            train_path=args.train_path,
            test_nomalpath=args.test_nomalpath,
            test_abnormalpath=args.test_abnormalpath,
            output_path=args.output_path
        )
    elif args.mode == 'test':
        algo.test(
            test_nomalpath=args.test_nomalpath,
            test_abnormalpath=args.test_abnormalpath,
            output_path=args.output_path
        )


def register_tracecrl(subparsers):
    parser = subparsers.add_parser('tracecrl', help='TraceCRL algorithm')
    sub = parser.add_subparsers(dest='mode', required=True)

    # Preprocess
    pre_parser = sub.add_parser('preprocess')
    pre_parser.add_argument('--dataset_dir', type=Path, required=True)
    pre_parser.add_argument('--dataset_name', required=True)

    # Train
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--dataset_name', required=True)
    train_parser.add_argument('--embedding_dim', type=int, default=128)


def handle_tracecrl(args):
    algo = TraceCRL()
    if args.mode == 'preprocess':
        algo.preprocess_data(
            dataset_dir=args.dataset_dir,
            dataset_name=args.dataset_name
        )
    elif args.mode == 'train':
        algo.train_and_test(
            dataset_name=args.dataset_name
        )



def register_putracead(subparsers):
    parser = subparsers.add_parser('putracead', help='PUTraceAD algorithm')
    sub = parser.add_subparsers(dest='mode', required=True)

    pre_parser = sub.add_parser('preprocess')
    pre_parser.add_argument('--mode', choices=['train', 'test'], required=True)
    pre_parser.add_argument('--input_path', type=Path, required=True)
    pre_parser.add_argument('--output_dir', type=Path, required=True)
    pre_parser.add_argument('--dataset', default='gaia')
    pre_parser.add_argument('--label_dir', type=Path)
    pre_parser.add_argument('--test_ratio', type=float, default=0.2)

    train_parser = sub.add_parser('train')
    train_parser.add_argument('--normal_path', type=Path, required=True)
    train_parser.add_argument('--train_path', type=Path, required=True)
    train_parser.add_argument('--abnormal_path', type=Path, required=True)
    train_parser.add_argument('--output_path', type=Path, required=True)

    test_parser = sub.add_parser('test')
    test_parser.add_argument('--normal_path', type=Path, required=True)
    test_parser.add_argument('--train_path', type=Path, required=True)
    test_parser.add_argument('--abnormal_path', type=Path, required=True)
    test_parser.add_argument('--output_path', type=Path, required=True)


def handle_putracead(args):
    algo = PUTraceAD()
    if args.mode == 'preprocess':
        algo.preprocess_data(
            mode=args.mode,
            input_path=args.input_path,
            output_dir=args.output_dir,
            dataset=args.dataset,
            label_dir=args.label_dir,
            test_ratio=args.test_ratio
        )
    elif args.mode == 'train':
        algo.train_and_test(
            normalpath=args.normal_path,
            trainpath=args.train_path,
            abnormalpath=args.abnormal_path,
            outputpath=args.output_path
        )
    elif args.mode == 'test':
        algo.test(
            normalpath=args.normal_path,
            trainpath=args.train_path,
            abnormalpath=args.abnormal_path,
            outputpath=args.output_path
        )



def register_tracevae(subparsers):
    parser = subparsers.add_parser('tracevae', help='TraceVAE algorithm')
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('data'))
    parser.add_argument('--data_path', type=Path)


def handle_tracevae(args):
    algo = TraceVAE()
    algo.preprocess_data(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir
    )
    if args.data_path:
        algo.execute_full_pipeline(data_path=args.data_path)



def register_multimodal_lstm(subparsers):
    parser = subparsers.add_parser('lstm', help='Multimodal LSTM')
    sub = parser.add_subparsers(dest='mode', required=True)

    train_parser = sub.add_parser('train')
    train_parser.add_argument('--dataset', required=True)
    train_parser.add_argument('--device', default='cuda:0')
    train_parser.add_argument('--nt', type=int, default=20)
    train_parser.add_argument('--base_path', default='./Datasets')
    train_parser.add_argument('--save_filename')
    train_parser.add_argument('--model_label')

    test_parser = sub.add_parser('test')
    test_parser.add_argument('--dataset', required=True)
    test_parser.add_argument('--device', default='cuda:0')
    test_parser.add_argument('--nt', type=int, default=20)


def handle_multimodal_lstm(args):
    algo = Multimodal_LSTM()
    if args.mode == 'train':
        algo.train(
            dataset=args.dataset,
            device=args.device,
            nt=args.nt,
            base_path=args.base_path,
            save_filename=args.save_filename,
            model_label=args.model_label
        )
    elif args.mode == 'test':
        algo.test(
            dataset=args.dataset,
            device=args.device,
            nt=args.nt
        )



def register_gtrace(subparsers):
    parser = subparsers.add_parser('gtrace', help='Gtrace algorithm')
    sub = parser.add_subparsers(dest='mode', required=True)

    train_parser = sub.add_parser('train')
    train_parser.add_argument('--data_path', type=Path, required=True)

    test_parser = sub.add_parser('test')
    test_parser.add_argument('--model_path', type=Path, required=True)


def handle_gtrace(args):
    algo = GTrace()
    if args.mode == 'train':
        algo.train(data_path=args.data_path)
    elif args.mode == 'test':
        algo.test(model_path=args.model_path)


if __name__ == '__main__':
    main()