set -ex

PROMPT_TYPE="${1:-qwen25-math-cot}"
MODEL_NAME_OR_PATH="${2:?MODEL_NAME_OR_PATH is required}"
OUTPUT_DIR="${3:-./output}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${EVAL_ROOT}/../../../data}"

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
# minerva_math,
DATA_NAME="olympiadbench"
TOKENIZERS_PARALLELISM=false \
python3 -u "${EVAL_ROOT}/math_eval.py" \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --data_dir "${DATA_DIR}" \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 3 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --apply_chat_template \







