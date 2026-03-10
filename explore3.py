import json
import glob

files = glob.glob('responses*.json')
with open(files[0], 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

print(f"Type of raw_data: {type(raw_data)}")
if isinstance(raw_data, dict):
    print(f"Number of keys: {len(raw_data.keys())}")
    for i, (k, v) in enumerate(raw_data.items()):
        if i >= 3: break
        print(f"Key {i}: {k}")
        print(f"Value type: {type(v)}")
        if isinstance(v, dict):
            print(f"  Keys in value: {list(v.keys())}")
            if 'data' in v:
                print(f"  Type of data: {type(v['data'])}")
                if isinstance(v['data'], list) and len(v['data']) > 0:
                    found_types = set()
                    found_oftypes = set()
                    for t in v['data']:
                        if isinstance(t, dict):
                            found_types.add(t.get('trial_type'))
                            found_oftypes.add(t.get('typeoftrial'))
                    print(f"  trial_types in data: {found_types}")
                    print(f"  typeoftrials in data: {found_oftypes}")
        elif isinstance(v, list):
            print(f"  Length of list: {len(v)}")
            if len(v) > 0:
                print(f"  First element keys: {v[0].keys() if isinstance(v[0], dict) else type(v[0])}")

elif isinstance(raw_data, list):
    print(f"Length of list: {len(raw_data)}")
    if len(raw_data) > 0:
        first = raw_data[0]
        print(f"First element type: {first}")

import os
metrics_file = 'all_metrics.json'
if os.path.exists(metrics_file):
    try:
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    except:
        all_metrics = {}
else:
    all_metrics = {}

out_metrics = {
    "raw_data_type": str(type(raw_data))
}
if isinstance(raw_data, dict):
    out_metrics["num_keys"] = len(raw_data.keys())
    if len(raw_data.keys()) > 0:
        out_metrics["first_key_type"] = str(type(raw_data[list(raw_data.keys())[0]]))
elif isinstance(raw_data, list):
    out_metrics["list_length"] = len(raw_data)

all_metrics["explore3.py"] = out_metrics

with open(metrics_file, 'w') as f:
    json.dump(all_metrics, f, indent=4)
