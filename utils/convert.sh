#!/bin/bash
set -x
shopt -s nullglob


    # "/home/test/test04/liran/LLM-GAN/examples/grpo_trainer/model_train/checkpoints/model_train/09_12_octothinker_hybrid_zero_tbs16_prompt768_response8192_mbs256_kl0.001_n16_t0.7_topp0.9_proposer1_repetition1.05_metric_val_reward_100percentreward_nok12_largeval_sympy_100_pt0.7_ptopp1.0_baseline0.5_online_format_entropy-0.01_saveload_15/best_actor_checkpoint"
declare -a paths=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for path in "${paths[@]}"; do
    echo "Processing path: $path"
    python "$SCRIPT_DIR/model_merger.py" --local_dir "$path/actor"
done

echo "done!"