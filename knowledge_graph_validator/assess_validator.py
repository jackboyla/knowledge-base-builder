import click
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Any, Optional, Literal
import sys
sys.path.insert(0, '../../knowledge_graph_validator')
import utils
import validators
import os

logger = utils.create_logger(__name__)


def compute_metrics(pos_results, neg_results):
    tp = 0
    fp = 0
    tn = 0
    fn = 0 
    for val in pos_results[0].model_dump()['validated_triples']:
        if val['triple_is_valid']:    # property is correctly marked as valid
            tp += 1
        else:                           # property is incorrectly marked as invalid
            fn += 1
    for val in neg_results[0].model_dump()['validated_triples']:
        if val['triple_is_valid']:    # property is incorrectly marked as valid
            fp += 1
        else:                           # property is correctly marked as invalid
            tn += 1

    metrics = utils.calc_metrics(tp, fp, tn, fn)
    return metrics

def sample_triples(triples, num_examples, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)  
        return np.random.choice(triples, num_examples, replace=False)
    else:
        return triples[:num_examples]

@click.command()
@click.option('--dataset', required=True, type=click.Choice(['FB13', 'WN11', 'WN18RR', 'YAGO3-10', 'FB15K-237N', 'CoDeX-S'], case_sensitive=False), help='Dataset name; one of [FB13, WN11, WN18RR, YAGO3-10]')
@click.option('--num-examples', default=10, type=int, help='Number of examples to evaluate')
@click.option('--random-seed', required=False, default=None, type=int, help='Random seed to select examples')
@click.option('--reference-context', required=False, type=click.Path(), help='The path to a custom reference context if the `RefKG` or `RefDocs` validator is chosen.')
@click.option('--context-type', required=True, type=click.Choice(['WorldKnowledgeKGValidator', 'ReferenceKGValidator', 'WikidataKGValidator', 'WebKGValidator'], case_sensitive=False), help='Model name')
def main(dataset, num_examples, random_seed, reference_context, context_type):
    """Evaluate a model on a dataset.

    usage:
            python assess_validator.py \
                --dataset CoDeX-S \
                --num-examples 10 \
                --random-seed 42 \
                --context-type WikidataKGValidator
    
    """

    if context_type in ['RefKG', 'RefDocs']:
        assert reference_context is not None, "You must provide a path to a reference context if you choose a reference context validator."

    positive_triples, negative_triples = utils.read_dataset(dataset)

    evaluators = {
        'WorldKnowledgeKGValidator': validators.WorldKnowledgeKGValidator,
        # 'ReferenceKGValidator': validators.ReferenceKGValidator,
        'WikidataKGValidator': validators.WikidataKGValidator,
        'WebSearch': validators.WebKGValidator,
    }
    v = evaluators[context_type]

    positive_samples = sample_triples(positive_triples, num_examples//2, random_seed)
    negative_samples = sample_triples(negative_triples, num_examples//2, random_seed)

    pos_results = []
    pos_results.append(v(**{'triples': positive_samples}))
    neg_results = []
    neg_results.append(v(**{'triples': negative_samples}))

    # saving
    results_json = [r.model_dump() for r in neg_results] + [r.model_dump() for r in pos_results]
    save_dir = Path('../data/results') / f"{dataset}" / f"{context_type}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / f"{num_examples}_{context_type}_{dataset}_seed{random_seed}.jsonl"
    utils.save_jsonl(results_json, str(save_path))
    
    metrics = compute_metrics(pos_results, neg_results)
    logger.info(f"-------------\nMETRICS:\n\n {metrics}\n-------------")

if __name__ == "__main__":
    main()
