import json
import glob
import pandas as pd

files = glob.glob('responses*.json')
raw_data = json.load(open(files[0], encoding='utf-8'))
subjects_data = list(raw_data['fluency-spam'].values())

records = []
for s in subjects_data:
    if 'data' not in s: continue
    subj_id = None
    hi_write = None
    vft_results = {}
    
    for t in s['data']:
        if subj_id is None and t.get('subject'): subj_id = t['subject']
        if t.get('typeoftrial') == 'insights':
            resp = t.get('response', {})
            if isinstance(resp, dict):
                hi_write = resp.get('Hi_Write')
        if t.get('task') == 'VFT':
            domain = t.get('domain')
            if 'practice' in str(domain).lower(): continue
            tagged = t.get('tagged_responses')
            num_words = 0
            if tagged:
                try:
                    num_words = len(json.loads(tagged))
                except: pass
            if domain: vft_results[domain] = num_words

    for domain, nw in vft_results.items():
        records.append({'subject_id': subj_id, 'hi_write': int(hi_write) if hi_write and hi_write.isdigit() else None, 'domain': domain, 'num_words': nw})

df = pd.DataFrame(records)
print("MEAN:")
print(df.groupby('domain')['num_words'].mean())
print("\nSTD:")
print(df.groupby('domain')['num_words'].std())

print("\nCorr:")
subj_df = df.groupby('subject_id').agg({'num_words':'mean', 'hi_write':'first'}).dropna()
print(subj_df.corr())

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

# Convert pandas series/dataframes to dicts for JSON serialization
mean_stats = df.groupby('domain')['num_words'].mean().to_dict()
std_stats = df.groupby('domain')['num_words'].std().to_dict()
corr_matrix = subj_df.corr().to_dict()

all_metrics["get_exact_stats.py"] = {
    "mean_words_per_domain": mean_stats,
    "std_words_per_domain": std_stats,
    "correlation_matrix": corr_matrix
}

with open(metrics_file, 'w') as f:
    json.dump(all_metrics, f, indent=4)
