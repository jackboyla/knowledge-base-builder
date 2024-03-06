#!/bin/bash

# Define arrays for each parameter if they vary between experiments
# For this example, let's vary the datasets and random seeds
datasets=("FB13" "FB15K-237-N" "CoDeX-S" "Wiki27K" "YAGO3-10" "UMLS" "WN11" "WN18RR")

# Fixed parameters can just be set once
model="gpt-3.5-turbo-0125"
context_type="WikidataWebKGValidator"
seed=11
num_examples=100

# Loop through all combinations of datasets and seeds
for dataset in "${datasets[@]}"; do
    echo "Running experiment with dataset ${dataset} and seed ${seed}"
    python assess_validator.py \
        --dataset "$dataset" \
        --num-examples "$num_examples" \
        --random-seed "$seed" \
        --model "$model" \
        --context-type "$context_type"
done
