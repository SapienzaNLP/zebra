#!/bin/bash
model=$1
current_dir=$(dirname "$(realpath "$0")")
for script in "$current_dir"/*.sh; do
    if [[ "$script" != "$current_dir/zebra.sh" ]]; then
        bash "$script" "$model"
    fi
done