import json
from typing import List, Dict, Union, Any, Optional, Literal

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
    f1_score = ((2 * (precision * recall)) / (precision + recall)) if precision + recall > 0 else 0
    accuracy = ((tp + tn) / (tp + tn + fp + fn)) if tp + tn + fp + fn > 0 else 0
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"Accuracy: {accuracy}")
    print("----------")
    return {"precision": precision, "recall": recall, "f1_score": f1_score, "accuracy": accuracy}



###########################
    
# DATASET LOADING UTILS

def read_nell_sports(file_path) -> List[Dict]:
    with open(file_path) as f:
        triples = []
        for line in f:
            # Strip the trailing period and split the line
            parts = line.rstrip('.').split('"')
            # Extract the subject, relation (assumed), and object
            subject = parts[1]
            relation = parts[2]
            object_ = parts[3]
            triples.append({'subject': subject, 'relation': relation, 'object': object_})
    return triples


def read_fb13(file_path) -> List[Dict]:
    positive_triples = []
    negative_triples = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')  # Splitting each line by tab
            if len(parts) == 4:  # Ensuring there are exactly four parts
                subject, relation, object_, sentiment = parts
                triple = {'subject': subject, 'relation': relation, 'object': object_}
                
                if sentiment == '1':
                    positive_triples.append(triple)
                elif sentiment == '-1':
                    negative_triples.append(triple)
    
    return positive_triples, negative_triples
