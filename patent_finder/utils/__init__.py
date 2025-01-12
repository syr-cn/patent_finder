import json

def save_json(data, filename):
    print(f'Writing {len(data)} data to {filename}...')
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f'Reading {len(data)} data from {filename}...')
    return data