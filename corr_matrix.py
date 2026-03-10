import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
matplotlib.use('Agg')

output_dir = '.'

files = glob.glob('responses*.json')
raw_data = json.load(open(files[0], encoding='utf-8'))
subjects_data = list(raw_data['fluency-spam'].values())

# Aggregate subject-level metrics for correlation
records = []

for s in subjects_data:
    if 'data' not in s: continue
    
    subj_id = None
    hi_read = np.nan
    hi_write = np.nan
    en_read = np.nan
    en_write = np.nan
    
    subj_words_total = 0
    subj_irts = []
    
    vft_count = 0
    
    for t in s['data']:
        if not subj_id and t.get('subject'): subj_id = t['subject']
        if t.get('typeoftrial') == 'insights':
            resp = t.get('response', {})
            if isinstance(resp, dict):
                try: hi_read = float(resp.get('Hi_Read'))
                except: pass
                try: hi_write = float(resp.get('Hi_Write'))
                except: pass
                try: en_read = float(resp.get('En_Read'))
                except: pass
                try: en_write = float(resp.get('En_Write'))
                except: pass
                
        if t.get('task') == 'VFT' and 'practice' not in str(t.get('domain', '')).lower():
            vft_count += 1
            tr = t.get('tagged_responses')
            rt = t.get('response_times')
            if tr and rt:
                try:
                    words = json.loads(tr)
                    times = json.loads(rt)
                    subj_words_total += len(words)
                    subj_irts.extend(times)
                except: pass
                
    if vft_count > 0:
        records.append({
            'Subject': subj_id,
            'Hindi_Read': hi_read,
            'Hindi_Write': hi_write,
            'Eng_Read': en_read,
            'Eng_Write': en_write,
            'Total_Words': subj_words_total,
            'Mean_IRT': np.mean(subj_irts) if subj_irts else np.nan
        })

df = pd.DataFrame(records)
# drop subjected with no irts
df = df.dropna(subset=['Total_Words', 'Mean_IRT'])

# Compute correlation matrix
corr_cols = ['Hindi_Read', 'Hindi_Write', 'Eng_Read', 'Eng_Write', 'Total_Words', 'Mean_IRT']
corr = df[corr_cols].corr()

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True)
plt.title('Correlation Matrix: Proficiency Scores vs Retrieval Metrics')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig4_correlation_matrix.png'), dpi=300)
plt.close()

print("Correlation matrix saved successfully.")

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

all_metrics["corr_matrix.py"] = {
    "correlation_matrix": corr.to_dict()
}

with open(metrics_file, 'w') as f:
    json.dump(all_metrics, f, indent=4)
