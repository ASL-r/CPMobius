# Evaluate Qwen2.5-Math-Instruct
set -x
PROMPT_TYPE="qwen25-math-cot"

# Qwen2.5-Math-1.5B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# # Qwen2.5-Math-7B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_NAME_OR_PATH="${1:-Qwen/Qwen2.5-Math-7B-Instruct}"
OUTPUT_DIR="${2:-./output}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/eval.sh" "$PROMPT_TYPE" "$MODEL_NAME_OR_PATH" "$OUTPUT_DIR"
