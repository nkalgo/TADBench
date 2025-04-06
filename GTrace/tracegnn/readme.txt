env: gtrace

1. 在shaominyi路径下 source .bashrc

2. 在TraceVAE处理完数据的情况下，可以拷贝部分数据过来
cd dataset
mkdir train_ratio_10
cd train_ratio_10
mkdir raw
mkdir processed
cd raw
cp -r ../../../../TraceVAE/data/train_ratio_10/* ./
rm -r processed
cp -r id_manager/* ../processed/
cd ../../../

python3 -m tracegnn.data.data_process preprocess -i dataset/train_ratio_10/raw -o dataset/train_ratio_10

3. 训练
!!!!!!记得修改tracegnn/models/gtrace/config.py里的dataset name!!!!!
python3 -m tracegnn.models.gtrace.main