#!/bin/bash
set -x
shopt -s nullglob

declare -a paths=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for path in "${paths[@]}"; do
    echo "Processing path: $path"
    python "$SCRIPT_DIR/model_merger.py" --local_dir "$path/actor"
done

echo "done!"