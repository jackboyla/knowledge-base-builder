import json
from typing import List, Dict, Union, Any, Optional, Literal
import os
import sys
import logging
import re
import random
import copy


def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = create_logger(__name__)


def read_jsonl(file_path: str):
    """read a JSONL file into a list of JSON objects"""

    json_lines_list = []

    # Open the .jsonl file and read it line by line
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            # Parse the JSON object from each line
            json_obj = json.loads(line.strip())

            json_lines_list.append(json_obj)

    return json_lines_list


def save_jsonl(jsonl_data, file_path):
    # Writing to a .jsonl file
    with open(file_path, "w", encoding="utf-8") as file:
        for document in jsonl_data:
            # Convert the JSON document to a string
            json_str = json.dumps(document) + "\n"
            # Write the JSON string to the file
            file.write(json_str)

    print(f"Saved to f'{file_path}")


def calc_metrics(tp, fp, tn, fn):
    precision = (tp / (tp + fp)) if tp + fp > 0 else 0
    recall = (tp / (tp + fn)) if tp + fn > 0 else 0
    f1_score = (
        ((2 * (precision * recall)) / (precision + recall))
        if precision + recall > 0
        else 0
    )
    accuracy = ((tp + tn) / (tp + tn + fp + fn)) if tp + tn + fp + fn > 0 else 0
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"Accuracy: {accuracy}")
    print("----------")
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }


###########################

# DATASET LOADING UTILS


def read_nell_sports(file_path) -> List[Dict]:
    with open(file_path) as f:
        triples = []
        for line in f:
            # Strip the trailing period and split the line
            parts = line.rstrip(".").split('"')
            # Extract the subject, relation (assumed), and object
            subject = parts[1]
            relation = parts[2]
            object_ = parts[3]
            triples.append(
                {"subject": subject, "relation": relation, "object": object_}
            )
    return triples


def load_mapping(file_path, dataset_name=None):
    """Load entity or relation to text mapping from a file."""
    mapping = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split("\t")
            # if dataset_name == "WN18RR":
            #     """e.g entity for WN18RR --> stool, solid excretory product evacuated from the bowels"""
            #     value = " ".join(value.split(",")[0])
            mapping[key] = value
    return mapping


def translate_triples(triples, entity_mapping=None, relation_mapping=None):
    """Translate entity and relation IDs in triples to their text representation."""
    translated_triples = []
    for triple in triples:
        translated_triples.append(
            {
                "subject": entity_mapping.get(triple["subject"]) if entity_mapping else triple["subject"],
                "relation": relation_mapping.get(
                    triple["relation"]) if relation_mapping else triple["relation"],
                "object": entity_mapping.get(triple["object"]) if entity_mapping else triple["subject"],
            }
        )
    return translated_triples

def preprocess_complex_relations(input_str: str) -> str:
    # Define a mapping of complex relations to placeholders
    complex_relations = {
        "languages spoken, written, or signed": "languages spoken written or signed",
    }
    # Replace complex relations with placeholders
    for cr, placeholder in complex_relations.items():
        if cr in input_str:
            input_str = input_str.replace(cr, placeholder)
            logger.info(f"Replacing {cr} with {placeholder}")
    return input_str


def parse_triple(input_str: str, dataset_name:str) -> Dict[str, str]:
    """
    USED for FB15K-237N, and CODEX-S, where the triple is given as a string
            e.g "\nThe input triple: \n( Artie Lange, influence influence node influenced by, Jackie Gleason )\n"
    Parses the input string to extract the triple components, handling cases where
    entities contain commas.
    """
    # Preprocessing the relation to handle cases where it is a complex relation
    input_str = preprocess_complex_relations(input_str)

    # Pattern to identify the relation - assumes it will be in between two commas and have NO COMMAS within it.
    relation_pattern = r", ([a-z_ ]+), "

    # Finding the relation using the regex pattern
    match = re.search(relation_pattern, input_str)
    if not match:
        raise ValueError(f"Could not find a relation in the input: {input_str}")

    relation = match.group(1)

    # Splitting the input based on the identified relation, taking into account the extra comma
    head, _, tail = input_str.partition(f", {relation}, ")

    # Cleaning up the head and tail
    head = head.strip(" (\n")
    tail = tail.strip(" )\n")
    # if dataset_name == 'FB15K-237N':
    #     relation = relation.split(" ")
    if type(relation) != list:
        relation = [relation]


    return {"subject": head, "relation": relation, "object": tail}

def negative_sampling(triples):
    """
    Performs negative sampling by swapping the object of one triple with another.

    Parameters:
    triples (list of dicts): A list of dictionaries, each containing 'subject', 'relation', and 'object'.

    Returns:
    list of dicts: The list of dictionaries after performing negative sampling.
    """
    # Ensure there are at least two triples to perform swapping
    if len(triples) < 2:
        raise ValueError("There must be at least two triples to perform negative sampling.")
    
    # Create a new list to hold the negatively sampled triples
    sampled_triples = triples[:]
    
    for i in range(len(triples)):
        # Select a random index different from i
        swap_index = i
        while swap_index == i:
            swap_index = random.randint(0, len(triples) - 1)
        
        # Swap the 'object' of the current triple with the 'object' of the randomly selected triple
        sampled_triples[i]['object'], sampled_triples[swap_index]['object'] = sampled_triples[swap_index]['object'], sampled_triples[i]['object']
    
    return sampled_triples

def read_dataset(
    dataset_name: Literal[
        "FB13", 
        "WN11", 
        "WN18RR", 
        "Wiki27K",
        "YAGO3-10", 
        "FB15K-237-N", 
        "CoDeX-S",
        "UMLS"]) -> List[Dict]:
    positive_triples = []
    negative_triples = []
    logger.info(f"Reading dataset {dataset_name}...")

    if dataset_name in ["FB13", "WN11"]:

        if os.path.exists(f"../data/{dataset_name}/entity2text_capital.txt"):
            entity_mapping_path = f"../data/{dataset_name}/entity2text_capital.txt"
        else:
            entity_mapping_path = f"../data/{dataset_name}/entity2text.txt"
        ent_mapping = load_mapping(entity_mapping_path, dataset_name)
        rel_mapping = load_mapping(f"../data/{dataset_name}/relation2text.txt")

        with open(f"../data/{dataset_name}/test.tsv", "r") as file:
            for line in file:
                parts = line.strip().split("\t")  # Splitting each line by tab
                assert len(parts) == 4  # Ensuring there are exactly four parts
                subject, relation, object_, sentiment = parts
                triple = {
                    "subject": subject,
                    "relation": relation,
                    "object": object_,
                }

                if sentiment == "1":
                    positive_triples.append(triple)
                elif sentiment == "-1":
                    negative_triples.append(triple)
                        
        positive_triples = translate_triples(positive_triples, ent_mapping, rel_mapping)
        negative_triples = translate_triples(negative_triples, ent_mapping, rel_mapping)

    elif dataset_name in ["CoDeX-S"]:  # "FB15K-237N", 
        data_file_path = f"../data/{dataset_name}/{dataset_name}-test.json"

        with open(data_file_path, "r") as file:
            data = json.load(file)

            for item in data:
                input_triple = item["input"].split('\n')[2]
                output = item["output"]

                try:
                    triple = parse_triple(input_triple, dataset_name)

                    if output == "True":
                        positive_triples.append(triple)
                    elif output == "False":
                        negative_triples.append(triple)
                    else:
                        logger.info(f"Output {output} is not recognized. Skipping.")
                except ValueError as e:
                    print(f"Error parsing triple: {e}")
                    continue

                assert (
                    len(triple) == 3
                ), f"Triple parts should have length 3, but have {len(triple)} instead: {triple}"

    elif dataset_name in ["WN18RR", "UMLS", "YAGO3-10"]:

        if os.path.exists(f"../data/{dataset_name}/entity2text_capital.txt"):
            entity_mapping_path = f"../data/{dataset_name}/entity2text_capital.txt"
        else:
            entity_mapping_path = f"../data/{dataset_name}/entity2text.txt"
        ent_mapping = load_mapping(entity_mapping_path, dataset_name)
        rel_mapping = load_mapping(f"../data/{dataset_name}/relation2text.txt")

        data_file_path = f"../data/{dataset_name}/test.tsv"

        triples = []
        with open(data_file_path, "r") as file:
            for line in file:
                parts = line.strip().split("\t")  # Splitting each line by tab
                assert len(parts) == 3
                subject, relation, object_ = parts
                triple = {
                    "subject": subject,
                    "relation": relation,
                    "object": object_,
                }
                triples.append(triple)

        positive_triples = copy.deepcopy(triples)
        negative_triples = negative_sampling(triples)
                        
        positive_triples = translate_triples(positive_triples, ent_mapping, rel_mapping)
        negative_triples = translate_triples(negative_triples, ent_mapping, rel_mapping)


    elif dataset_name in ["FB15K-237-N"]:

        def process_fb15k_file(file) -> List[Dict]:
            triples = []
            for line in file:
                parts = line.strip().split("\t")  # Splitting each line by tab
                assert len(parts) == 3
                subject, relation, object_ = parts
                relation = relation.replace('.', '')
                relation = relation.replace('_', '')
                relation = relation.split('/')[1:]   #  [1:] because relation starts with `/` e.g  /people/person/nationality
                triple = {
                    "subject": subject,
                    "relation": relation,
                    "object": object_,
                }
                triples.append(triple)
            return triples


        entity_mapping_path = f"../data/{dataset_name}/entity2label.txt"
        ent_mapping = load_mapping(entity_mapping_path, dataset_name)

        with open(f"../data/{dataset_name}/o_test_pos.txt", "r") as file:

            positive_triples = process_fb15k_file(file)

        with open(f"../data/{dataset_name}/o_test_neg.txt", "r") as file:

            negative_triples = process_fb15k_file(file)

                        
        positive_triples = translate_triples(positive_triples, ent_mapping)
        negative_triples = translate_triples(negative_triples, ent_mapping)


    elif dataset_name in ["Wiki27K"]:

        def process_wiki27k_file(file) -> List[Dict]:
            triples = []
            for line in file:
                parts = line.strip().split("\t")  # Splitting each line by tab
                assert len(parts) == 3
                subject, relation, object_ = parts
                triple = {
                    "subject": subject,
                    "relation": relation,
                    "object": object_,
                }
                triples.append(triple)
            return triples


        entity_mapping_path = f"../data/{dataset_name}/entity2label.txt"
        ent_mapping = load_mapping(entity_mapping_path, dataset_name)
        with open(f"../data/{dataset_name}/relation2label.json", 'r') as f:
            rel_mapping = json.load(f)

        with open(f"../data/{dataset_name}/o_test_pos.txt", "r") as file:

            positive_triples = process_wiki27k_file(file)

        with open(f"../data/{dataset_name}/o_test_neg.txt", "r") as file:

            negative_triples = process_wiki27k_file(file)

                        
        positive_triples = translate_triples(positive_triples, ent_mapping, rel_mapping)
        negative_triples = translate_triples(negative_triples, ent_mapping, rel_mapping)
    


    save_jsonl(positive_triples, f"pos_tmp.jsonl")
    save_jsonl(negative_triples, f"neg_tmp.jsonl")

    return positive_triples, negative_triples
