import json
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
import os
import matplotlib
matplotlib.use('Agg')

output_dir = '.'

files = glob.glob('responses*.json')
raw_data = json.load(open(files[0], encoding='utf-8'))
subjects_data = list(raw_data['fluency-spam'].values())

# --- Data Structures for Questions ---
# RQ1: Within vs Between Cluster IRTs
rq1_within_irts = []
rq1_between_irts = []

# RQ2: Consecutive SpAM distance vs IRT
rq2_consecutive_dist = []
rq2_consecutive_irt = []

# Word-level neighborhood vs IRT
rq2_word_neighborhood_dist = []
rq2_word_irt = []

# RQ3: Fluency vs total words, mean IRT, cluster size
rq3_fluency_scores = []
rq3_total_words = []
rq3_mean_irt = []
rq3_mean_cluster_size = []

for s in subjects_data:
    if 'data' not in s: continue
    
    subj_id = None
    hi_read = None
    hi_write = None
    
    vft_records = {}
    spam_records = {}
    
    for t in s['data']:
        if not subj_id and t.get('subject'): subj_id = t['subject']
        if t.get('typeoftrial') == 'insights':
            resp = t.get('response', {})
            if isinstance(resp, dict):
                hi_read = resp.get('Hi_Read')
                hi_write = resp.get('Hi_Write')
                
        if t.get('task') == 'VFT' and 'practice' not in str(t.get('domain', '')).lower():
            dom = t.get('domain')
            tr = t.get('tagged_responses')
            rt = t.get('response_times')
            if dom and tr and rt:
                try:
                    words = [w['response'] for w in json.loads(tr)]
                    times = json.loads(rt)
                    vft_records[dom] = {'words': words, 'rt': times}
                except: pass
                
        if t.get('task') == 'SpAM' and 'practice' not in str(t.get('domain', '')).lower():
            dom = t.get('domain')
            dw = t.get('droppedwords')
            if dom and dw:
                if isinstance(dw, str):
                    try: dw = json.loads(dw)
                    except: pass
                coords = {}
                for item in dw:
                    if isinstance(item, dict) and 'word' in item and 'x_norm' in item:
                        coords[item['word']] = (item['x_norm'], item['y_norm'])
                spam_records[dom] = coords

    # Calculate Subject Fluency
    try:
        fluency = int(hi_read) + int(hi_write)
    except:
        fluency = None
        
    subj_words_total = 0
    subj_irts = []
    subj_cluster_sizes = []
    
    for dom in vft_records.keys():
        if dom in spam_records:
            words = vft_records[dom]['words']
            rts = vft_records[dom]['rt']
            coords = spam_records[dom]
            
            # Keep only words present in both VFT and SpAM
            valid_idx = [i for i, w in enumerate(words) if w in coords]
            if len(valid_idx) < 3: continue  # Need enough for clustering
            
            w_valid = [words[i] for i in valid_idx]
            rt_valid = [rts[i] for i in valid_idx]
            c_valid = [coords[w] for w in w_valid]
            
            subj_words_total += len(w_valid)
            subj_irts.extend(rt_valid)
            
            # Perform Clustering (distance threshold 0.3 for normalized space 0-1)
            X = np.array(c_valid)
            clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3)
            labels = clusterer.fit_predict(X)
            
            clusters_count = len(set(labels))
            avg_clust_size = len(w_valid) / clusters_count
            subj_cluster_sizes.append(avg_clust_size)
            
            # IRTs within vs between
            # We look at consecutive retrievals
            for i in range(1, len(w_valid)):
                w1, w2 = w_valid[i-1], w_valid[i]
                l1, l2 = labels[i-1], labels[i]
                irt = rt_valid[i]
                
                # RQ1
                if l1 == l2:  # Same cluster
                    rq1_within_irts.append(irt)
                else:         # Different cluster
                    rq1_between_irts.append(irt)
                    
                # RQ2 consecutive
                d = np.linalg.norm(X[i-1] - X[i])
                rq2_consecutive_dist.append(d)
                rq2_consecutive_irt.append(irt)
                
            # RQ2 word neighborhood
            # Mean distance to all other words in domain
            for i in range(len(w_valid)):
                # distance to all others
                dists = [np.linalg.norm(X[i] - X[j]) for j in range(len(w_valid)) if i != j]
                mean_d = np.mean(dists) if dists else 0
                rq2_word_neighborhood_dist.append(mean_d)
                rq2_word_irt.append(rt_valid[i])

    if fluency is not None and subj_words_total > 0:
        rq3_fluency_scores.append(fluency)
        rq3_total_words.append(subj_words_total)
        rq3_mean_irt.append(np.mean(subj_irts))
        rq3_mean_cluster_size.append(np.mean(subj_cluster_sizes))


# ================ STATISTICAL TESTING & FIGURES ================

print("=== RQ1: Within vs Between Cluster IRTs ===")
mean_within = np.mean(rq1_within_irts)
mean_between = np.mean(rq1_between_irts)
t_stat, p_val = stats.ttest_ind(rq1_within_irts, rq1_between_irts, equal_var=False)
print(f"Mean Within-Cluster IRT: {mean_within:.2f} ms")
print(f"Mean Between-Cluster IRT: {mean_between:.2f} ms")
print(f"Welch's t-test: t = {t_stat:.3f}, p-value = {p_val:.3e}")

# Fig 1: Boxplot of Within vs Between
plt.figure()
data = pd.DataFrame({'IRT Type': ['Within Cluster'] * len(rq1_within_irts) + ['Between Cluster'] * len(rq1_between_irts),
                     'IRT (ms)': rq1_within_irts + rq1_between_irts})
# Remove extreme outliers for visualization
q3 = data['IRT (ms)'].quantile(0.95)
data_clean = data[data['IRT (ms)'] < q3]
sns.boxplot(x='IRT Type', y='IRT (ms)', data=data_clean, palette='pastel')
plt.title('Inter-Response Times (IRTs): Within vs Between Semantic Clusters')
plt.savefig(os.path.join(output_dir, 'fig1_rq1_irt_clusters.png'), dpi=300)
plt.close()

print("\\n=== RQ2: Tighter Semantic Neighborhoods vs IRT ===")
# Correlation between consecutive distance and IRT
r_cons, p_cons = stats.pearsonr(rq2_consecutive_dist, rq2_consecutive_irt)
print(f"Consecutive Distance vs IRT: r = {r_cons:.3f}, p = {p_cons:.3e}")

# Word Neighborhood vs IRT
r_neigh, p_neigh = stats.pearsonr(rq2_word_neighborhood_dist, rq2_word_irt)
print(f"Mean Neighborhood Distance vs Word IRT: r = {r_neigh:.3f}, p = {p_neigh:.3e}")

plt.figure()
sns.regplot(x=rq2_word_neighborhood_dist, y=rq2_word_irt, scatter_kws={'alpha':0.3})
plt.xlabel('Mean Spatial Distance to Semantic Neighborhood (SpAM Coordinates)')
plt.ylabel('Word Retrieval Time (IRT in ms)')
plt.title('Relationship Between Semantic Tighness and Retrieval Efficiency')
plt.ylim(0, np.percentile(rq2_word_irt, 95)) # clip y axis for readability
plt.savefig(os.path.join(output_dir, 'fig2_rq2_neighborhood_irt.png'), dpi=300)
plt.close()


print("\\n=== RQ3: Fluency vs Lexical Retrieval Efficiency ===")
r_words, p_words = stats.pearsonr(rq3_fluency_scores, rq3_total_words)
r_irt, p_irt = stats.pearsonr(rq3_fluency_scores, rq3_mean_irt)
r_clust, p_clust = stats.pearsonr(rq3_fluency_scores, rq3_mean_cluster_size)

print(f"Fluency vs Total Words: r = {r_words:.3f}, p = {p_words:.3e}")
print(f"Fluency vs Mean Subject IRT: r = {r_irt:.3f}, p = {p_irt:.3e}")
print(f"Fluency vs Mean Cluster Size: r = {r_clust:.3f}, p = {p_clust:.3e}")

# Bonferroni correction for multi-comparisons on RQ3
alphas = [0.05/3]*3
p_vals = [p_words, p_irt, p_clust]
sig = [p < a for p, a in zip(p_vals, alphas)]
print(f"Significant after Bonferroni correction (alpha=0.0167)? {sig}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.regplot(x=rq3_fluency_scores, y=rq3_total_words)
plt.title('Fluency vs Total Words')

plt.subplot(1, 3, 2)
sns.regplot(x=rq3_fluency_scores, y=rq3_mean_irt)
plt.title('Fluency vs Mean IRT')

plt.subplot(1, 3, 3)
sns.regplot(x=rq3_fluency_scores, y=rq3_mean_cluster_size)
plt.title('Fluency vs Cluster Size')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig3_rq3_fluency_metrics.png'), dpi=300)
plt.close()

print("\\nAnalysis complete. Figures saved.")

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

all_metrics["report2_analysis.py"] = {
    "RQ1_clusters": {
        "mean_within_irt": float(mean_within),
        "mean_between_irt": float(mean_between),
        "t_stat": float(t_stat),
        "p_val": float(p_val)
    },
    "RQ2_neighborhood": {
        "consecutive_dist_irt_r": float(r_cons),
        "consecutive_dist_irt_p": float(p_cons),
        "word_neigh_dist_irt_r": float(r_neigh),
        "word_neigh_dist_irt_p": float(p_neigh)
    },
    "RQ3_fluency": {
        "fluency_total_words_r": float(r_words),
        "fluency_total_words_p": float(p_words),
        "fluency_mean_irt_r": float(r_irt),
        "fluency_mean_irt_p": float(p_irt),
        "fluency_cluster_size_r": float(r_clust),
        "fluency_cluster_size_p": float(p_clust),
        "significant_alpha_0.0167": [bool(s) for s in sig]
    }
}

with open(metrics_file, 'w') as f:
    json.dump(all_metrics, f, indent=4)
