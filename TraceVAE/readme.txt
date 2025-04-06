env: TraceVAE1

(原始数据格式为规定的统一格式，如果不是，可以参考data/generate_csv.py中的process_data函数，处理成对应的csv文件就可以)
1. python data/generate_csv.py -n xxx -dir xxx
生成train.csv, test.csv, val.csv

2. 准备文件
mkdir data/xxx/id_manager

3. 生成中间文件
python3 -m tracegnn.cli.data_process make-status-id-train-and-val-and-test -i data/xxx -o data/xxx/id_manager
python3 -m tracegnn.cli.data_process make-service-id-train-and-val-and-test -i data/xxx -o data/xxx/id_manager
python3 -m tracegnn.cli.data_process make-operation-id-train-and-val-and-test -i data/xxx -o data/xxx/id_manager
python3 -m tracegnn.cli.data_process preprocess -i data/xxx -o data/xxx
python3 -m tracegnn.cli.data_process make-latency-range -i data/xxx --names train

4. 确保results文件夹中没有文件，如果有，要把上一个数据集生成的results移走
mkdir results_xxx
cd results_xxx
mkdir train_results
mv ../results/* ./train_results
cd ..
rm -r results
5. 训练
bash train.sh data/xxx

6. 评估
bash test.sh results/train/models/final.pt data/xxx xxx



