import json
import glob

files = glob.glob('responses*.json')
raw_data = json.load(open(files[0], encoding='utf-8'))
subjects = list(raw_data['fluency-spam'].values())

vft_trials = []
for s in subjects:
    if 'data' in s:
        for t in s['data']:
            if t.get('task') == 'VFT':
                vft_trials.append(t)

print(f"Found {len(vft_trials)} VFT trials total.")
if vft_trials:
    print("Keys of a VFT trial:", list(vft_trials[0].keys()))
    print("Full content of first VFT trial:", json.dumps(vft_trials[0], indent=2, ensure_ascii=False))

    # Also, is there another trial type right around VFT? Let's check trial just after a VFT.
    for i, t in enumerate(subjects[0]['data']):
        if t.get('task') == 'VFT':
            print(f"\\nTrial {i} (VFT) content:")
            print(t)
            print(f"\\nTrial {i+1} (Next) content:")
            print(subjects[0]['data'][i+1] if i+1 < len(subjects[0]['data']) else 'None')
            break

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

all_metrics["explore_vft.py"] = {
    "vft_trials_found": len(vft_trials)
}

with open(metrics_file, 'w') as f:
    json.dump(all_metrics, f, indent=4)
