#!/bin/bash

# bash assess_validators.sh 2>&1 | tee logs/run_$(date "+%Y-%m-%d_%H-%M-%S").log


# Define arrays for each parameter if they vary between experiments
# For this example, let's vary the datasets and random seeds
datasets=("UMLS" "FB15K-237-N" "Wiki27K" "FB13" "CoDeX-S" "YAGO3-10")

# Fixed parameters can just be set once
model="gpt-3.5-turbo-0125"
context_types=('WebKGValidator' 'WorldKnowledgeKGValidator' 'WikidataKGValidator' 'WikidataWebKGValidator' 'WikipediaWikidataKGValidator')
seed=17
num_examples=4
embedding_model="all-MiniLM-L6-v2" #  all-MiniLM-L6-v2 / mixedbread-ai/mxbai-embed-large-v1 / text-embedding-3-small

# Loop through all combinations of datasets and seeds
for dataset in "${datasets[@]}"; do
    for context_type in "${context_types[@]}"; do
        echo "Running experiment with dataset ${dataset} and seed ${seed}"
        python assess_validator.py \
            --dataset "$dataset" \
            --num-examples "$num_examples" \
            --random-seed "$seed" \
            --model "$model" \
            --context-type "$context_type" \
            --embedding-model "$embedding_model"
    done
done
