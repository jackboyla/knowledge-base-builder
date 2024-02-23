import json

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
