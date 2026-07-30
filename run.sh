echo "Device is ${1}. Running on dataset ${2}";
CUDA_VISIBLE_DEVICES=${1} uv run python -u ./code/main.py --config "./config/${2}.yaml"