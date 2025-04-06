# GTrace
Code and dataset for GTrace.

## Evaluation of Accuracy
We provide dataset B for evaluation. The dataset is under the `dataset` folder.
- Install Python 3.8+ on your system.
- Run `pip3 install -r requirements.txt` to install the dependencies.
- Run `python3 -m tracegnn.models.gtrace.main` to start training. The evaluation will automatically starts after training.
- If you want to run on GPU, you can modify the `device` in `tracegnn/models/config.py`.

## Evaluation of Time Efficiency
We provide the code for the `Anomaly Detection` module and `Graph Building` module.

To evaluate the time efficiency, we provide a minimal example and a trained model that can be run directly on your local device without deployment:
- Run `cd deployment`.
- Install `GCC 9.3.0+`, `make` and `CMake 3.2+` on your device. Run `bash build.sh` to download and build the dependencies.
- Run `sh run_local.sh` to evaluate the time efficiency.
- Install `Intel SVML` to get better performance on Intel CPU. (See https://numba.readthedocs.io/en/stable/user/performance-tips.html#intel-svml).

## Visualization Tool
- Run `python3 -m tracegnn.visualization.webviewer_server`.
- Visit `http://localhost:12312/0` or `http://localhost:12312/1` to see the visualization results for two example cases.

# Reference
- LRUCache11: https://github.com/mohaps/lrucache11.git
- Kubernetes: https://kubernetes.io/
- PyTorch: https://pytorch.org/
- DGL: https://dgl.ai/
- CMake: https://cmake.org/
