# Multimodal LSTM
## Usage
### Train
```
python3 -m tracegnn.models.lstm.train --dataset data1 --device cuda
```

### Test
```
python3 -m tracegnn.models.lstm.test --dataset data1 --device cuda
```
### 数据迁移
'''
cd ../tracegnn/Datasets
rm -r *
mkdir data1
cd data1
mkdir after_process_db
cd after_process_db
cp -r ../../../../TraceVAE/data/data1/processed/* ./
cp -r ../../../../TraceVAE/data/data1/id_manager/* ./
cd ../../..
'''
