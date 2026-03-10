import json
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up matplotlib to not require a GUI
import matplotlib
matplotlib.use('Agg')

# Locate dataset
files = glob.glob('responses*.json')
with open(files[0], 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

subjects_data = list(raw_data['fluency-spam'].values())

# We want to extract for each subject:
# - Subject ID
# - Demographics: age, gender, Hi_Read, Hi_Write, En_Read, En_Write
# - VFT tasks: domain, num_words, mean_rt

records = []

for s in subjects_data:
    if 'data' not in s: continue
    
    # Subject-level info
    subj_id = None
    age = None
    gender = None
    hi_read = None
    hi_write = None
    
    vft_results = {}
    
    for t in s['data']:
        if subj_id is None and t.get('subject'):
            subj_id = t['subject']
            
        if t.get('typeoftrial') == 'demographics':
            resp = t.get('response', {})
            if isinstance(resp, dict):
                if 'age' in resp: age = resp['age']
                if 'gender' in resp: gender = resp['gender']
        if t.get('typeoftrial') == 'insights':
            resp = t.get('response', {})
            if isinstance(resp, dict):
                hi_read = resp.get('Hi_Read')
                hi_write = resp.get('Hi_Write')
                
        if t.get('task') == 'VFT':
            domain = t.get('domain')
            # ignore practice
            if 'practice' in str(domain).lower():
                continue
                
            tagged = t.get('tagged_responses')
            rt = t.get('response_times')
            
            num_words = 0
            mean_rt = 0
            
            if tagged:
                try:
                    tagged_list = json.loads(tagged)
                    num_words = len(tagged_list)
                except:
                    pass
            if rt:
                try:
                    rt_list = json.loads(rt)
                    if len(rt_list) > 0:
                        mean_rt = sum(rt_list) / len(rt_list)
                except:
                    pass
            
            if domain:
                vft_results[domain] = {'num_words': num_words, 'mean_rt': mean_rt}

    # Create a record for each domain
    for domain, stats in vft_results.items():
        records.append({
            'subject_id': subj_id,
            'age': age,
            'gender': gender,
            'hi_read': int(hi_read) if hi_read and hi_read.isdigit() else None,
            'hi_write': int(hi_write) if hi_write and hi_write.isdigit() else None,
            'domain': domain,
            'num_words': stats['num_words'],
            'mean_rt': stats['mean_rt']
        })

import pandas as pd
df = pd.DataFrame(records)

print(f"Total VFT records parsed: {len(df)}")
print("Summary Statistics (Num Words):")
print(df.groupby('domain')['num_words'].describe())

output_dir = '.'

# Plot 1: Words produced per domain
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='domain', y='num_words', palette='Set2')
plt.title('Distribution of Words Produced During Hindi Fluency Task by Domain')
plt.xlabel('Category Domain')
plt.ylabel('Number of Unique Words Produced')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, 'fig1_words_by_domain.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Relationship between Hindi Writing skill and Num Words
# Avg num words across all domains per subject
subj_df = df.groupby('subject_id').agg({
    'num_words': 'mean',
    'hi_write': 'first'
}).reset_index()

subj_df = subj_df.dropna(subset=['hi_write'])

print("\nSummary Statistics Subject-level:")
print(subj_df.describe())

plt.figure(figsize=(8, 6))
sns.regplot(data=subj_df, x='hi_write', y='num_words', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Effect of Self-Reviewed Hindi Writing Proficiency on Verbal Fluency')
plt.xlabel('Hindi Writing Proficiency (1-5 Scale)')
plt.ylabel('Average Words Produced (Across Domains)')
plt.xticks([1, 2, 3, 4, 5])
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(output_dir, 'fig2_fluency_vs_proficiency.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Plots saved successfully.")

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

all_metrics["analyze.py"] = {
    "total_vft_records_parsed": len(df),
    "domain_num_words_summary": df.groupby('domain')['num_words'].describe().to_dict(),
    "subject_level_summary": subj_df.describe().to_dict()
}

with open(metrics_file, 'w') as f:
    json.dump(all_metrics, f, indent=4)
