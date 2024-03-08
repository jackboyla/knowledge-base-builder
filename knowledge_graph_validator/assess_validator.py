import click
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Union, Any, Optional, Literal
import sys
sys.path.insert(0, '../../knowledge_graph_validator')
import utils
import os

logger = utils.create_logger(__name__)


def compute_metrics(pos_results, neg_results):
    tp = 0
    fp = 0
    tn = 0
    fn = 0 
    for val in pos_results['validated_triples']:
        if val['triple_is_valid']:    # property is correctly marked as valid
            tp += 1
        else:                           # property is incorrectly marked as invalid
            fn += 1
    for val in neg_results['validated_triples']:
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
@click.option('--dataset', required=True, type=click.Choice(
    ['FB13', 'UMLS', 'WN11', 'WN18RR', 'Wiki27K', 'YAGO3-10', 'FB15K-237-N', 'CoDeX-S'], 
    case_sensitive=False
    ), help='Dataset name; one of [FB13, UMLS, WN11, WN18RR, Wiki27K, YAGO3-10, FB15K-237-N, CoDeX-S]')
@click.option('--num-examples', default=10, type=int, help='Number of examples to evaluate')
@click.option('--random-seed', required=False, default=None, type=int, help='Random seed to select examples')
@click.option('--model', required=True, type=click.Choice(['gpt-4-1106-preview', 'gpt-3.5-turbo-0125'], case_sensitive=False), help='The model to use as validator.')
@click.option('--context-type', required=True, type=click.Choice(
    ['WorldKnowledgeKGValidator', 'ReferenceKGValidator', 'TextContextKGValidator', 'WikidataKGValidator', 'WebKGValidator', 'WikidataWebKGValidator', 'WikipediaWikidataKGValidator'], 
    case_sensitive=False
    ), help='Model name')
def main(dataset, num_examples, random_seed, model, context_type):
    """Evaluate a model on a dataset.

    usage:
                python assess_validator.py \
                    --dataset CoDeX-S \
                    --num-examples 4 \
                    --random-seed 23 \
                    --model gpt-3.5-turbo-0125 \
                    --context-type WebKGValidator \
        
    """
    os.environ['VALIDATION_MODEL'] = model
    import validators

    positive_triples, negative_triples = utils.read_dataset(dataset)

    evaluators = {
        'WorldKnowledgeKGValidator': validators.WorldKnowledgeKGValidator,
        'WikidataWebKGValidator': validators.WikidataWebKGValidator,
        'ReferenceKGValidator': validators.ReferenceKGValidator,
        'TextContextKGValidator': validators.TextContextKGValidator,
        'WikipediaWikidataKGValidator': validators.WikipediaWikidataKGValidator,
        'WikidataKGValidator': validators.WikidataKGValidator,
        'WebKGValidator': validators.WebKGValidator,
    }
    v = evaluators[context_type]

    positive_samples = sample_triples(positive_triples, num_examples//2, random_seed)
    negative_samples = sample_triples(negative_triples, num_examples//2, random_seed)
    positive_samples = {'triples': positive_samples}
    negative_samples = {'triples': negative_samples}


    if context_type in ['ReferenceKGValidator']:

        '''assuming a path to the reference KG for a given dataset exists'''
        reference_kg = utils.read_reference_kg(dataset)
        negative_samples['reference_knowledge_graph'] = reference_kg
        positive_samples['reference_knowledge_graph'] = reference_kg

    if context_type in ['TextContextKGValidator']:
        pass

    logger.info(f"Validation class --> {context_type}")

    logger.info("Evaluating on positive samples")
    pos_results = {'triples': [], 'validated_triples': []}
    for p in tqdm(positive_samples['triples']):
        try:
            result = (v(**{'triples': [p]}))
            pos_results['triples'].append(result.model_dump()['triples'][0])
            pos_results['validated_triples'].append(result.model_dump()['validated_triples'][0])
        except Exception as e:
            logger.info(f"Error validating {p} due to {e}")

    logger.info("Evaluating on negative samples")
    neg_results = {'triples': [], 'validated_triples': []}
    for n in tqdm(negative_samples['triples']):
        try:
            result = (v(**{'triples': [n]}))
            neg_results['triples'].append(result.model_dump()['triples'][0])
            neg_results['validated_triples'].append(result.model_dump()['validated_triples'][0])
        except Exception as e:
            logger.info(f"Error validating {n} due to {e}")

    # saving
    results_json = [{'negative_results': neg_results, 'positive_results': pos_results}]
    save_dir = Path('../data/results') / f"{dataset}" / f"{context_type}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / f"{num_examples}_{context_type}_{dataset}_seed{random_seed}_{model}.jsonl"
    utils.save_jsonl(results_json, str(save_path))
    
    metrics = compute_metrics(pos_results, neg_results)
    logger.info(f"-------------\nMETRICS:\n\n {metrics}\n-------------")

if __name__ == "__main__":
    main()
