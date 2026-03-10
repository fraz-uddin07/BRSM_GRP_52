import json
import glob
from collections import Counter

files = glob.glob('responses*.json')
with open(files[0], 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

subjects = list(raw_data['fluency-spam'].values())

t_types = Counter()
t_oftypes = Counter()
fluency_examples = []

for s in subjects:
    if 'data' not in s:
        continue
    for t in s['data']:
        t_types[str(t.get('trial_type'))] += 1
        t_oftypes[str(t.get('typeoftrial'))] += 1
        
        # In psychological tasks, categorical fluency might have a specific typeoftrial like 'fluency', 'category_fluency'
        # Or trial_type might be 'survey-text' or 'survey-html-form'
        if 'fluency' in str(t.get('typeoftrial', '')).lower() or 'fluency' in str(t.get('task', '')).lower():
            fluency_examples.append(t)

print("Type of trials:")
for k, v in t_oftypes.items():
    print(f"  {k}: {v}")
    
print("\ntrial_types:")
for k, v in t_types.items():
    print(f"  {k}: {v}")

print(f"\nFound {len(fluency_examples)} fluency trials.")
if fluency_examples:
    print("Keys in a fluency trial:", fluency_examples[0].keys())
    print("Example response:", fluency_examples[0].get('response'))
    print("Full example:", {k: v for k, v in fluency_examples[0].items() if k != 'stimulus'})
else:
    # If not found by name, let's dump a few responses from 'survey-text' or 'survey-html-form'
    print("\nLooking for potential fluency text inputs...")
    for s in subjects:
        if 'data' not in s: continue
        for t in s['data']:
            if t.get('trial_type') == 'survey-html-form' and t.get('typeoftrial') not in ['demographics', 'insights', 'consent']:
                print(f"typeoftrial: {t.get('typeoftrial')}")
                print(f"response: {t.get('response')}")

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

all_metrics["analyze_data.py"] = {
    "t_oftypes": dict(t_oftypes),
    "trial_types": dict(t_types),
    "fluency_trials_found": len(fluency_examples)
}

with open(metrics_file, 'w') as f:
    json.dump(all_metrics, f, indent=4)
