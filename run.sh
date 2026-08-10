device="$1"
dataset="$2"

echo "Device is ${device}. Running on dataset ${dataset}";
shift 2
CUDA_VISIBLE_DEVICES=${device} uv run python -u ./code/main.py --config "./config/${dataset}.yaml" "$@"