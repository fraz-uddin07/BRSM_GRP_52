import json
import glob
from collections import Counter

files = glob.glob('responses*.json')
with open(files[0], 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# The data might be directly a list of subjects, or a dict. 
# Usually Firebase/Firestore exports look like a dict where keys are document IDs and values are data bags.
if isinstance(raw_data, dict):
    subjects = list(raw_data.values())
else:
    subjects = raw_data

print(f"Total subjects: {len(subjects)}")

trial_types = Counter()
type_of_trials = Counter()

fluency_responses = []

for idx, subj in enumerate(subjects):
    # Some datasets have a 'data' array inside the subject dict.
    if 'data' in subj:
        trials = subj['data']
    else:
        # Check if the list itself is flat or what
        trials = subj if isinstance(subj, list) else []
    
    for t in trials:
        t_type = t.get('trial_type')
        t_oftype = t.get('typeoftrial')
        
        if t_type:
            trial_types[t_type] += 1
        if t_oftype:
            type_of_trials[t_oftype] += 1
            
        if t_oftype == 'fluency' or 'fluency' in str(t_oftype).lower() or 'fluency' in str(t_type).lower():
            fluency_responses.append((idx, t))

print("\n--- Trial Types ---")
for k, v in trial_types.items():
    print(f"{k}: {v}")

print("\n--- Type of Trials ---")
for k, v in type_of_trials.items():
    print(f"{k}: {v}")

print(f"\nFound {len(fluency_responses)} fluency trial entries.")
if fluency_responses:
    print("Example fluency trial:")
    print(json.dumps(fluency_responses[0][1], indent=2, ensure_ascii=False))

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

all_metrics["explore2.py"] = {
    "total_subjects": len(subjects),
    "trial_types": dict(trial_types),
    "type_of_trials": dict(type_of_trials),
    "fluency_trial_entries": len(fluency_responses)
}

with open(metrics_file, 'w') as f:
    json.dump(all_metrics, f, indent=4)
