
from loguru import logger
from sdk_new.base import *
from Multimodal_LSTM.models.lstm.test_for_sdk import LSTM_test
from Multimodal_LSTM.models.lstm.train_for_sdk import *
from PUTraceAD.process_for_sdk import *
from PUTraceAD.test_for_sdk import *
from PUTraceAD.train_for_sdk import *
from TraceCRL.generate_operations import *
from TraceCRL.node_embedding import *
from TraceCRL.process_data import *
from TraceCRL.train_for_sdk import *
from TraceAnomaly.f1_for_sdk import *
from TraceAnomaly.traceanomaly.main_for_sdk import *
from CRISP.process_for_sdk import *
from TraceVAE.data.csv_for_sdk import *
from TraceVAE.tracegnn.cli.process_for_sdk import *
from GTrace.tracegnn.models.gtrace.main_for_sdk import *
from PUTraceAD.process_for_sdk import *


class TraceAnomaly(TADTemplate):
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return 0

    def train(self) -> object:
        Ta_train(self.train_path, self.output_path)
        return 0

    def test(self) -> (object, dict):
        Ta_test(self.test_normalpath, self.test_abnormalpath, self.output_path)

    def calculate_metrics(self):
        return 0


class CRISP(TADTemplate):
    def preprocess(
        self,
        service_name,
        operation_name,
        trace_files,
        root_trace,
        parallelism,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        Cr_preprocess_data(service_name, operation_name, trace_files, root_trace, parallelism)
        return 0

    def train(self) -> object:
        Ta_train(self.train_path, self.test_nomalpath, self.test_abnormalpath, self.output_path)
        return 0

    def test(self) -> (object, dict):
        Ta_test(self.test_nomalpath, self.test_abnormalpath, self.output_path)
        return 0


class TraceCRL(TADTemplate):
    def preprocess_data(self, dataset_dir: str, dataset_name: str) -> pd.DataFrame:
        process_data_uniform(dataset_dir, dataset_name)
        generate_operations('train')
        generate_operations('test')
        for dataset_type in ['train', 'test']:
            file_name = f'data/{dataset_type}/preprocessed/{dataset_type}.json'
            weighted = False
            name = '_weighted' if weighted else ''
            graph = build_graph(file_name)
            edgelist_file = make(graph, dataset_type, weighted=weighted)
            deepwalk_embedding(
                edgelist_filename=edgelist_file,
                output_filename=f'./experiment/{dataset_type}/node_embedding/embedding{name}_deepwalk.json'
            )
            shutil.copyfile(
                f'./experiment/{dataset_type}/node_embedding/embedding{name}_deepwalk.json',
                f'data/{dataset_type}/preprocessed/embeddings.json'
            )
        return 0

    def train_and_test(self, dataset_name) -> object:
        train_and_test(dataset_name)
        return 0


class PUTraceAD(TADTemplate):
    def preprocess_data(
        self,
        mode: str,
        input_path: str,
        output_dir: str,
        dataset: str = "gaia",
        label_dir: str = None,
        test_ratio: float = 0.2,
    ) -> Dict[str, Union[int, List[str]]]:
        TracePreprocessor.PU_preprocess(mode, input_path, output_dir, dataset, label_dir, test_ratio)
        return 0

    def train(
        self,
        seed=None,
        batch_size=256,
        lr=5e-4,
        epochs=200,
        weight_decay=5e-3,
        momentum=0.9,
        gnn="gat",
        gnn_layers=3,
        dropout=0.0,
        dataset="test_trace_graph",
        labeled_ratio=0.05,
        pratio=1.0,
        data_root="./data",
        gpu=None,
        workers=4,
        self_paced=False,
        self_paced_type="A",
        use_ema=False,
        ema_decay=0.999,
        model_dir="models",
        model_name="best_model.pth",
        save_results=True,
        verbose=True
    ):
        PU_train(
            seed,
            batch_size,
            lr,
            epochs,
            weight_decay,
            momentum,
            gnn,
            gnn_layers,
            dropout,
            dataset,
            labeled_ratio,
            pratio,
            data_root,
            gpu,
            workers,
            self_paced,
            self_paced_type,
            use_ema,
            ema_decay,
            model_dir,
            model_name,
            save_results,
            verbose
        )
        return 0

    def test(
        self,
        labeled_ratio=0.05,
        pratio=1,
        seed=2,
        data_root="./data",
        model_path="model_best.pth",
        gnn_type="gat",
        gnn_layers=3,
        dropout=0.0,
        in_channels=64,
        batch_size=128,
        loss_type='nnPU',
        device="cuda:0",
        workers=4,
        **kwargs
    ):
        PU_test(
            labeled_ratio,
            pratio,
            seed,
            data_root,
            model_path,
            gnn_type,
            gnn_layers,
            dropout,
            in_channels,
            batch_size,
            loss_type,
            device,
            workers,
        )
        return 0


class TraceVAE(TADTemplate):
    def preprocess_data(self, dataset_name: str, dataset_dir: str, output_dir: str = 'data') -> None:
        Vae_preprocess_data(dataset_name, dataset_dir, output_dir)

    def execute_full_pipeline(self, data_path, df: pd.DataFrame):
        vae_execute_full_pipeline(data_path)


class Multimodal_LSTM(TADTemplate):
    def preprocess_data(self, dataset_name: str, dataset_dir: str, output_dir: str = 'data') -> None:
        Vae_preprocess_data(dataset_name, dataset_dir, output_dir)

    def execute_full_pipeline(self, data_path, df: pd.DataFrame):
        vae_execute_full_pipeline(data_path)

    def test(
        self,
        dataset: str,
        device: str = 'cuda',
        nt: int = 20,
        data3: bool = False,
        data4: bool = False,
        drop2: bool = False,
        ltest: bool = False,
        no_biased: bool = False
    ) -> object:
        LSTM_test(dataset, device, nt, data3, data4, drop2, ltest, no_biased)
        return 0

    def train(
        self,
        dataset: str,
        device: str = 'cuda:1',
        nt: int = 20,
        data3: bool = False,
        data4: bool = False,
        drop2: bool = False,
        ltest: bool = False,
        base_path: str = './Datasets',
        save_filename: str = None,
        model_label: str = None
    ) -> (object, dict):
        LSTM_test(dataset, device, nt, data3, data4, drop2, ltest, base_path, save_filename, model_label)
        return 0


class GTrace(TADTemplate):
    def preprocess_data(self, dataset_name: str, dataset_dir: str, output_dir: str = 'data') -> None:
        Vae_preprocess_data(dataset_name, dataset_dir, output_dir)

    def execute_full_pipeline(self, data_path, df: pd.DataFrame):
        vae_execute_full_pipeline(data_path)

    def train(self) -> object:
        G_train()
        return 0

    def test(self) -> (object, dict):
        G_test()
        return 0
