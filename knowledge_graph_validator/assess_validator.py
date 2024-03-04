import click
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Any, Optional, Literal
import sys
sys.path.insert(0, '../../knowledge_graph_validator')
import utils
import validators

logger = utils.create_logger(__name__)


def compute_metrics(pos_results, neg_results):
    tp = 0
    fp = 0
    tn = 0
    fn = 0 
    for val in pos_results[0].model_dump()['validated_triples']:
        if val['property_is_valid']:    # property is correctly marked as valid
            tp += 1
        else:                           # property is incorrectly marked as invalid
            fn += 1
    for val in neg_results[0].model_dump()['validated_triples']:
        if val['property_is_valid']:    # property is incorrectly marked as valid
            fp += 1
        else:                           # property is correctly marked as invalid
            tn += 1

    metrics = utils.calc_metrics(tp, fp, tn, fn)
    return metrics

def sample_triples(triples, num_examples, random):
    if random:
        return np.random.choice(triples, num_examples, replace=False)
    else:
        return triples[:num_examples]

@click.command()
@click.option('--dataset', required=True, type=click.Choice(['FB13', 'WN11', 'WN18RR', 'YAGO3-10'], case_sensitive=False), help='Dataset name')
@click.option('--num-examples', default=10, type=int, help='Number of examples to evaluate')
@click.option('--random', is_flag=True, help='Whether to randomly select examples')
@click.option('--save-path', required=True, type=click.Path(), help='Where to save the results')
@click.option('--context-type', required=True, type=click.Choice(['LLMonly', 'RefKG', 'WebSearch'], case_sensitive=False), help='Model name')
def main(dataset, num_examples, random, save_path, context_type):
    """Evaluate a model on a dataset.

    usage:
            python assess_validator.py --dataset FB13 \
                --num-examples 10 \
                --random \
                --save-path results.json \
                --context-type WebSearch
    
    """

    positive_triples, negative_triples = utils.read_dataset(dataset)

    evaluators = {
        # 'LLMonly': validators.NoContextValidator,
        # 'RefKG': validators.RefKGValidator,
        'WebSearch': validators.WebKGValidator,
    }
    v = evaluators[context_type]

    positive_samples = sample_triples(positive_triples, num_examples//2, random)
    negative_samples = sample_triples(negative_triples, num_examples//2, random)

    pos_results = []
    pos_results.append(v(**{'triples': positive_samples}))
    neg_results = []
    neg_results.append(v(**{'triples': negative_samples}))

    metrics = compute_metrics(pos_results, neg_results)
    logger.info(f"-------------\nMETRICS:\n\n {metrics}\n-------------")

    results_json = [r.model_dump() for r in neg_results] + [r.model_dump() for r in pos_results]
    utils.save_jsonl(results_json, save_path)

if __name__ == "__main__":
    main()
