#!/bin/bash

# Retriever output paths
retriever_output_path="data/zebra_retriever/outputs/arc/challenge/arc_challenge_test.arc_train.jsonl"

# Dataset path
data_path="data/datasets/arc/challenge/arc-challenge-test.jsonl"

dataset_tag="arc"
# Fewshot data path
fewshot_data_path="data/datasets/arc/challenge/arc-challenge-train.jsonl"
# Explanation path
explanations_path="sapienzanlp/zebra-kb-explanations"
explanations_split=(
    "arc-train-gemini"
)
# Num kg examples
num_kg_examples=(1 3 5 10 20)
# Output dir
output_dir="results/zebra/arc/challenge/test"

# Model name
model=$1
# Path to the Python script
python_script="zebra/run_zebra.py"

for split in "${explanations_split[@]}"; do
    for num_examples in "${num_kg_examples[@]}"; do
        echo "Running Zebra with: model=$model, explanations=$split and num_kg_examples=$num_examples"
        python "$python_script" \
            --retriever_output_path "$retriever_output_path" \
            --data_path "$data_path" \
            --fewshot_data_path "$fewshot_data_path" \
            --explanations_path "$explanations_path" \
            --explanations_split "$split" \
            --model_name "$model" \
            --dataset_tag "$dataset_tag" \
            --output_dir "$output_dir" \
            --plain \
            --oracle \
            --examples \
            --num_qa_examples 0 \
            --num_kg_examples "$num_examples"
    done
done