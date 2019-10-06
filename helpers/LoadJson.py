import json

def load_json(filepath):
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name
