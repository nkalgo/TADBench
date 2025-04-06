echo "Usage: bash test.sh [model_path] [dataset_path] [dataset_name]"
echo "MODEL: $1"
echo "DATASET: $2"
echo "NAME: $3"
# python3 -m tracegnn.models.trace_vae.test evaluate-nll -M "$1" --use-train-val -D "$2" --device cpu --use-std-limit --std-limit-global
python3 -m tracegnn.models.trace_vae.test_evl evaluate-nll -M "$1" --use-train-val -D "$2" -N "$3" --device cpu --use-std-limit --std-limit-global
