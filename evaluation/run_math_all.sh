#!/usr/bin/env bash
# =============================================================================
#
# Unified evaluation script for math benchmarks.
# Supported datasets:
#   math500, amc, aime, aime2025, minerva_math, olympiadbench
#
# All benchmarks run through the shared parquet-based chat evaluator
# (utils/chat_eval.py + utils/parquet_loader.py). The parquet files under
# ``data/`` are pre-replicated (suffix ``_xN``); a single greedy sample per row
# yields self-consistency over N rollouts.
#
# Usage:
#   bash run_math_all.sh --model <MODEL_PATH> [options]
#
# Examples:
#   # Run all benchmarks with default sampling on a single model
#   bash run_math_all.sh --model /path/to/checkpoint
#
#   # Run only a subset
#   bash run_math_all.sh --model /path/to/ckpt --tasks "math500,aime,aime2025"
#
#   # Custom sampling
#   bash run_math_all.sh --model /path/to/ckpt \
#       --temperature 0.7 --top-p 0.9 --repetition-penalty 1.05
#
#   # Override output directory and GPU devices
#   OUTPUT_ROOT=./my_results CUDA_VISIBLE_DEVICES=0,1 \
#       bash run_math_all.sh --model /path/to/ckpt
# =============================================================================

set -uo pipefail
shopt -s nullglob


REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results}"

ALL_TASKS="math500,amc,aime,aime2025,minerva_math,olympiadbench"

MODEL=""
TASKS="${ALL_TASKS}"
RUN_NAME=""
TEMPERATURE="0.7"
TOP_P="0.9"
REPETITION_PENALTY="1.05"
MAX_TOKENS="4096"

# ----------------------------- Helpers ---------------------------------------

usage() {
  sed -n '/^# ====/,/^# ====/p' "$0" | sed 's/^# \{0,1\}//'
  cat <<EOF

Options:
  --model PATH                 HuggingFace model path or name (required)
  --tasks LIST                 Comma-separated tasks to run.
                               Default: ${ALL_TASKS}
  --run-name NAME              Sub-directory under \$OUTPUT_ROOT.
                               Default: derived from --model basename
  --temperature FLOAT          Sampling temperature (default: ${TEMPERATURE})
  --top-p FLOAT                Top-p (default: ${TOP_P})
  --repetition-penalty FLOAT   Repetition penalty (default: ${REPETITION_PENALTY})
  --max-tokens INT             Per-sample max new tokens (default: ${MAX_TOKENS})
  -h, --help                   Show this help and exit

Environment overrides:
  DATA_ROOT       Root of evaluation data (default: <repo>/data)
  OUTPUT_ROOT     Root of result outputs   (default: <repo>/results)
  CUDA_VISIBLE_DEVICES  GPUs to use (passed through to subprocesses)
EOF
}

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }


while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)               MODEL="$2"; shift 2 ;;
    --tasks)               TASKS="$2"; shift 2 ;;
    --run-name)            RUN_NAME="$2"; shift 2 ;;
    --temperature)         TEMPERATURE="$2"; shift 2 ;;
    --top-p)               TOP_P="$2"; shift 2 ;;
    --repetition-penalty)  REPETITION_PENALTY="$2"; shift 2 ;;
    --max-tokens)          MAX_TOKENS="$2"; shift 2 ;;
    -h|--help)             usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

[[ -z "${MODEL}" ]] && { usage; die "--model is required"; }

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="$(basename "$(echo "${MODEL}" | sed 's:/*$::')")"
fi


TEMP_TAG="${TEMPERATURE//./p}"
TOPP_TAG="${TOP_P//./p}"
REP_TAG="${REPETITION_PENALTY//./p}"
TAG="${TEMP_TAG}-${TOPP_TAG}-${REP_TAG}"

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

log "Model:       ${MODEL}"
log "Tasks:       ${TASKS}"
log "Output dir:  ${OUTPUT_DIR}"
log "Sampling:    temperature=${TEMPERATURE} top_p=${TOP_P} repetition_penalty=${REPETITION_PENALTY} max_tokens=${MAX_TOKENS}"


run_eval() {
  # $1: evaluator script  $2: parquet path  $3: save subdir
  local script="$1" parquet="$2" save_sub="$3"
  local save="${OUTPUT_DIR}/${save_sub}/${TAG}"
  if [[ ! -f "${script}" ]]; then
    warn "Script not found, skip: ${script}"; return
  fi
  if [[ ! -f "${parquet}" ]]; then
    warn "Data not found, skip: ${parquet}"; return
  fi
  mkdir -p "${save}"
  TOKENIZERS_PARALLELISM=false \
  python3 -u "${script}" \
    --model "${MODEL}" \
    --data_path "${parquet}" \
    --save_dir "${save}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --max_tokens "${MAX_TOKENS}"
}


IFS=',' read -ra TASK_LIST <<< "${TASKS}"
for task in "${TASK_LIST[@]}"; do
  task="$(echo "${task}" | xargs)"  # trim
  [[ -z "${task}" ]] && continue
  log "=== Running task: ${task} ==="

  case "${task}" in
    math500)
      run_eval "${REPO_ROOT}/Math/math/evaluate_math.py" \
               "${DATA_ROOT}/math_x5.parquet" "math500" ;;
    amc)
      run_eval "${REPO_ROOT}/Math/amc/evaluate_amc.py" \
               "${DATA_ROOT}/amc_x10.parquet" "amc" ;;
    aime)
      run_eval "${REPO_ROOT}/Math/aime/evaluate_aime.py" \
               "${DATA_ROOT}/aime_x32.parquet" "aime2024" ;;
    aime2025)
      run_eval "${REPO_ROOT}/Math/aime/evaluate_aime_2025.py" \
               "${DATA_ROOT}/aimo-validation-aime2025_x32.parquet" "aime2025" ;;
    minerva_math)
      run_eval "${REPO_ROOT}/Math/minerva/evaluate_minerva.py" \
               "${DATA_ROOT}/minerva_6x.parquet" "minerva_math" ;;
    olympiadbench)
      run_eval "${REPO_ROOT}/Math/olympiad/evaluate_olympiad.py" \
               "${DATA_ROOT}/olympiad_bench_3x.parquet" "olympiadbench" ;;
    *)
      warn "Unknown task: ${task} (skipped)" ;;
  esac
done

log "All evaluations finished. Results -> ${OUTPUT_DIR}"
