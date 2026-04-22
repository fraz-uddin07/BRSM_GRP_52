"""
BRSM Report 2: Advanced Statistical Analysis of Hindi Fluency Task
Analyses: ANOVA, Linear Regression, Bayesian Statistics, GLM
"""

import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import euclidean
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
import pingouin as pg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})
sns.set_style("whitegrid")

OUTPUT_DIR = r'd:\BRSM\project\phase2\figures'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. DATA EXTRACTION AND PREPROCESSING
# ============================================================

print("=" * 70)
print("PHASE 1: DATA EXTRACTION AND PREPROCESSING")
print("=" * 70)

with open(r'd:\BRSM\project\phase2\responses.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

participants = raw_data['fluency-spam']

# Extract all data into structured DataFrames
all_vft_rows = []
all_spam_rows = []
all_demo_rows = []

for pid, pdata in participants.items():
    subj = pdata.get('subject_id', pid)
    
    # Demographics extraction
    demo = {'subject_id': subj}
    for trial in pdata.get('data', []):
        tt = trial.get('trial_type', '')
        if tt in ('survey-html-form', 'survey-multi-choice', 'survey-text'):
            resp = trial.get('response', {})
            if isinstance(resp, dict):
                demo.update(resp)
    
    all_demo_rows.append(demo)
    
    # VFT and SpAM extraction
    for trial in pdata.get('data', []):
        task = trial.get('task', '')
        domain = trial.get('domain', '')
        
        if 'practice' in domain:
            continue
        
        if task == 'VFT':
            tagged = trial.get('tagged_responses', '[]')
            times  = trial.get('response_times', '[]')
            if isinstance(tagged, str):
                tagged = json.loads(tagged)
            if isinstance(times, str):
                times = json.loads(times)
            
            for i, (entry, t) in enumerate(zip(tagged, times)):
                all_vft_rows.append({
                    'subject_id': subj,
                    'domain': domain,
                    'word': entry['response'],
                    'word_order': entry['tag'],
                    'irt': t,
                    'is_first_word': i == 0,
                })
        
        elif task == 'SpAM':
            dropped = trial.get('droppedwords', [])
            # Get final positions (last placement for each word id)
            final_pos = {}
            for w in dropped:
                final_pos[w['id']] = w
            
            for wid, w in final_pos.items():
                all_spam_rows.append({
                    'subject_id': subj,
                    'domain': domain,
                    'word': w['word'],
                    'word_id': w['id'],
                    'x_norm': w['x_norm'],
                    'y_norm': w['y_norm'],
                })

df_vft = pd.DataFrame(all_vft_rows)
df_spam = pd.DataFrame(all_spam_rows)
df_demo = pd.DataFrame(all_demo_rows)

# Clean demographics
for col in ['Hi_Read', 'Hi_Write', 'En_Read', 'En_Write', 'age', 'education']:
    if col in df_demo.columns:
        df_demo[col] = pd.to_numeric(df_demo[col], errors='coerce')

# Create Hindi proficiency composite
if 'Hi_Read' in df_demo.columns and 'Hi_Write' in df_demo.columns:
    df_demo['hindi_proficiency'] = (df_demo['Hi_Read'] + df_demo['Hi_Write']) / 2.0
    df_demo['english_proficiency'] = (df_demo['En_Read'] + df_demo['En_Write']) / 2.0

print(f"VFT data: {len(df_vft)} word entries from {df_vft['subject_id'].nunique()} participants")
print(f"SpAM data: {len(df_spam)} placements from {df_spam['subject_id'].nunique()} participants")
print(f"Demographics: {len(df_demo)} participants")
print(f"Domains: {df_vft['domain'].unique()}")

# ============================================================
# 2. DERIVED METRICS
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2: COMPUTING DERIVED METRICS")
print("=" * 70)

# 2a. Per-subject per-domain summary
subj_domain = df_vft.groupby(['subject_id', 'domain']).agg(
    total_words=('word', 'count'),
    mean_irt=('irt', 'mean'),
    median_irt=('irt', 'median'),
    sd_irt=('irt', 'std'),
    first_word_irt=('irt', 'first'),
).reset_index()

# 2b. Compute semantic distances from SpAM
def compute_pairwise_distances(group):
    """Compute mean pairwise distance for a subject's domain SpAM placements."""
    coords = group[['x_norm', 'y_norm']].values
    n = len(coords)
    if n < 2:
        return pd.Series({'mean_pairwise_dist': np.nan, 'n_words_spam': n})
    
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(euclidean(coords[i], coords[j]))
    
    return pd.Series({
        'mean_pairwise_dist': np.mean(dists),
        'median_pairwise_dist': np.median(dists),
        'sd_pairwise_dist': np.std(dists),
        'n_words_spam': n,
    })

spam_summary = df_spam.groupby(['subject_id', 'domain']).apply(compute_pairwise_distances).reset_index()

# 2c. Compute consecutive word distances (spatial)
def compute_consecutive_distances(subj, domain):
    """Get spatial distance between consecutively typed words."""
    vft_words = df_vft[(df_vft['subject_id'] == subj) & (df_vft['domain'] == domain)].sort_values('word_order')
    spam_words = df_spam[(df_spam['subject_id'] == subj) & (df_spam['domain'] == domain)]
    
    if len(spam_words) == 0:
        return []
    
    # Create word -> position mapping
    word_pos = {}
    for _, row in spam_words.iterrows():
        word_pos[row['word'].lower().strip()] = (row['x_norm'], row['y_norm'])
    
    results = []
    prev_word = None
    for _, row in vft_words.iterrows():
        word = row['word'].lower().strip()
        if prev_word is not None and prev_word in word_pos and word in word_pos:
            dist = euclidean(word_pos[prev_word], word_pos[word])
            results.append({
                'subject_id': subj,
                'domain': domain,
                'word_from': prev_word,
                'word_to': word,
                'spatial_distance': dist,
                'irt': row['irt'],
                'word_order': row['word_order'],
            })
        prev_word = word
    return results

consec_rows = []
for subj in df_vft['subject_id'].unique():
    for domain in df_vft[df_vft['subject_id'] == subj]['domain'].unique():
        consec_rows.extend(compute_consecutive_distances(subj, domain))

df_consec = pd.DataFrame(consec_rows)
print(f"Consecutive word-pair data: {len(df_consec)} transitions")

# 2d. Merge everything into a master per-subject-domain DataFrame
master = subj_domain.merge(spam_summary, on=['subject_id', 'domain'], how='left')
master = master.merge(df_demo[['subject_id', 'hindi_proficiency', 'english_proficiency', 
                                'age', 'education', 'gender', 'first_language', 'state_ut']], 
                       on='subject_id', how='left')

# Compute log-transformed IRT for regression
master['log_mean_irt'] = np.log(master['mean_irt'].clip(lower=1))
df_consec['log_irt'] = np.log(df_consec['irt'].clip(lower=1))

print(f"Master DataFrame: {len(master)} rows, {master.columns.tolist()}")
print(f"\nDomain counts in master:")
print(master['domain'].value_counts())

# ============================================================
# 3. ANALYSIS 1: ONE-WAY ANOVA - Word Production Across Domains
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 1: ONE-WAY ANOVA - Word Production Across Domains")
print("=" * 70)

# Only use domains that all participants have (animals, foods)
# But also include colours and body-parts for the analysis
anova_data = master[master['domain'].isin(['animals', 'foods', 'colours', 'body-parts'])].copy()

# Levene's test for homogeneity of variances
groups = [g['total_words'].dropna().values for _, g in anova_data.groupby('domain')]
levene_stat, levene_p = stats.levene(*groups)
print(f"\nLevene's Test: F = {levene_stat:.4f}, p = {levene_p:.4f}")

# One-way ANOVA
anova_result = pg.anova(data=anova_data, dv='total_words', between='domain', detailed=True)
print(f"\nOne-Way ANOVA (DV: Total Words, IV: Domain):")
print(anova_result.to_string())

# Effect size (eta-squared)
eta_sq = anova_result['np2'].values[0]
print(f"\nPartial Eta-Squared: {eta_sq:.4f}")

# Welch's ANOVA (more robust to unequal variances)
welch_result = pg.welch_anova(data=anova_data, dv='total_words', between='domain')
print(f"\nWelch's ANOVA:")
print(welch_result.to_string())

# Post-hoc: Tukey HSD
tukey = pairwise_tukeyhsd(anova_data['total_words'], anova_data['domain'], alpha=0.05)
print(f"\nTukey HSD Post-Hoc:")
print(tukey.summary())

# Games-Howell (non-parametric post-hoc)
gh = pg.pairwise_gameshowell(data=anova_data, dv='total_words', between='domain')
print(f"\nGames-Howell Post-Hoc:")
print(gh.to_string())

# --- ANOVA Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot
order = ['colours', 'animals', 'foods', 'body-parts']
available_order = [d for d in order if d in anova_data['domain'].unique()]
sns.boxplot(data=anova_data, x='domain', y='total_words', order=available_order, 
            palette='Set2', ax=axes[0])
axes[0].set_title('Word Production by Domain')
axes[0].set_xlabel('Domain')
axes[0].set_ylabel('Total Words Produced')

# Violin plot with individual points
sns.violinplot(data=anova_data, x='domain', y='total_words', order=available_order,
               palette='Set2', inner=None, alpha=0.3, ax=axes[1])
sns.stripplot(data=anova_data, x='domain', y='total_words', order=available_order,
              color='black', alpha=0.5, size=4, ax=axes[1])
axes[1].set_title('Word Production Distribution by Domain')
axes[1].set_xlabel('Domain')
axes[1].set_ylabel('Total Words Produced')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'anova_word_production.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[Saved: anova_word_production.png]")


# ============================================================
# 4. ANALYSIS 2: TWO-WAY ANOVA - IRT by Domain and Proficiency
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 2: TWO-WAY ANOVA - Mean IRT by Domain x Hindi Proficiency")
print("=" * 70)

# Create proficiency groups (median split)
median_prof = master['hindi_proficiency'].median()
master['prof_group'] = master['hindi_proficiency'].apply(
    lambda x: 'High' if x >= median_prof else 'Low'
)

two_way_data = master[master['domain'].isin(['animals', 'foods'])].copy()

# Two-way ANOVA using pingouin
aov2 = pg.anova(data=two_way_data, dv='mean_irt', between=['domain', 'prof_group'], detailed=True)
print(f"\nTwo-Way ANOVA (DV: Mean IRT, IVs: Domain x Proficiency):")
print(aov2.to_string())

# Interaction plot  
fig, ax = plt.subplots(figsize=(10, 6))
interaction_data = two_way_data.groupby(['domain', 'prof_group'])['mean_irt'].agg(['mean', 'sem']).reset_index()
for prof in ['Low', 'High']:
    subset = interaction_data[interaction_data['prof_group'] == prof]
    ax.errorbar(subset['domain'], subset['mean'], yerr=subset['sem'], 
                marker='o', capsize=5, label=f'{prof} Proficiency', linewidth=2, markersize=8)
ax.set_title('Interaction: Domain x Hindi Proficiency on Mean IRT')
ax.set_xlabel('Domain')
ax.set_ylabel('Mean IRT (ms)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'twoway_anova_interaction.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Saved: twoway_anova_interaction.png]")


# ============================================================
# 5. ANALYSIS 3: REPEATED-MEASURES CONSIDERATIONS (IRT across domains)
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 3: Kruskal-Wallis (Non-parametric alternative)")
print("=" * 70)

# Since IRT data is likely non-normal, also run non-parametric
kw_groups = [g['mean_irt'].dropna().values for name, g in anova_data.groupby('domain')]
kw_stat, kw_p = stats.kruskal(*kw_groups)
print(f"Kruskal-Wallis H = {kw_stat:.4f}, p = {kw_p:.4f}")

# Normality check
print("\nShapiro-Wilk Normality Tests (mean IRT by domain):")
for domain in anova_data['domain'].unique():
    vals = anova_data[anova_data['domain'] == domain]['mean_irt'].dropna()
    if len(vals) >= 3:
        sw_stat, sw_p = stats.shapiro(vals)
        print(f"  {domain}: W = {sw_stat:.4f}, p = {sw_p:.4f} {'*' if sw_p < 0.05 else ''}")


# ============================================================
# 6. ANALYSIS 4: LINEAR REGRESSION - Predicting IRT from Spatial Distance
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 4: LINEAR REGRESSION - Spatial Distance -> IRT")
print("=" * 70)

if len(df_consec) > 0:
    # Simple linear regression
    X = sm.add_constant(df_consec['spatial_distance'])
    y = df_consec['log_irt']
    model1 = sm.OLS(y, X).fit()
    print(f"\nSimple Linear Regression: log(IRT) ~ Spatial Distance")
    print(model1.summary2().tables[1].to_string())
    print(f"R² = {model1.rsquared:.4f}, Adj R² = {model1.rsquared_adj:.4f}")
    print(f"F({model1.df_model:.0f},{model1.df_resid:.0f}) = {model1.fvalue:.4f}, p = {model1.f_pvalue:.6f}")
    
    # Multiple regression with domain as covariate
    df_consec_model = df_consec.copy()
    df_consec_model['domain'] = df_consec_model['domain'].astype('category')
    model2 = smf.ols('log_irt ~ spatial_distance + C(domain)', data=df_consec_model).fit()
    print(f"\nMultiple Regression: log(IRT) ~ Spatial Distance + Domain")
    print(model2.summary2().tables[1].to_string())
    print(f"R² = {model2.rsquared:.4f}, Adj R² = {model2.rsquared_adj:.4f}")
    
    # Model with word_order as covariate (serial position effect)
    model3 = smf.ols('log_irt ~ spatial_distance + word_order + C(domain)', data=df_consec_model).fit()
    print(f"\nMultiple Regression: log(IRT) ~ Spatial Distance + Word Order + Domain")
    print(model3.summary2().tables[1].to_string())
    print(f"R² = {model3.rsquared:.4f}, Adj R² = {model3.rsquared_adj:.4f}")
    
    # ANOVA table for model comparison
    print(f"\nANOVA: Model 1 vs Model 3:")
    anova_comp = anova_lm(model1, model3)
    print(anova_comp.to_string())
    
    # Regression diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter with regression line
    axes[0, 0].scatter(df_consec['spatial_distance'], df_consec['log_irt'], alpha=0.3, s=10)
    x_line = np.linspace(df_consec['spatial_distance'].min(), df_consec['spatial_distance'].max(), 100)
    axes[0, 0].plot(x_line, model1.params.iloc[0] + model1.params.iloc[1] * x_line, 'r-', linewidth=2)
    axes[0, 0].set_xlabel('Spatial Distance (SpAM)')
    axes[0, 0].set_ylabel('log(IRT)')
    axes[0, 0].set_title('Regression: Spatial Distance -> log(IRT)')
    
    # Residuals vs fitted
    fitted = model3.fittedvalues
    residuals = model3.resid
    axes[0, 1].scatter(fitted, residuals, alpha=0.3, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Fitted')
    
    # QQ plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Residuals')
    
    # Residuals distribution
    axes[1, 1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    xmin, xmax = axes[1, 1].get_xlim()
    x_hist = np.linspace(xmin, xmax, 100)
    axes[1, 1].plot(x_hist, stats.norm.pdf(x_hist, residuals.mean(), residuals.std()), 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Distribution of Residuals')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'regression_diagnostics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[Saved: regression_diagnostics.png]")


# ============================================================
# 7. ANALYSIS 5: MULTIPLE REGRESSION - Predicting Total Words
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 5: MULTIPLE REGRESSION - Predicting Total Word Production")
print("=" * 70)

reg_data = master.dropna(subset=['hindi_proficiency', 'total_words', 'domain']).copy()

model_words = smf.ols('total_words ~ hindi_proficiency + english_proficiency + C(domain)', 
                       data=reg_data).fit()
print(f"\nMultiple Regression: Total Words ~ Hindi Prof + English Prof + Domain")
print(model_words.summary2().tables[1].to_string())
print(f"R² = {model_words.rsquared:.4f}, Adj R² = {model_words.rsquared_adj:.4f}")
print(f"F = {model_words.fvalue:.4f}, p = {model_words.f_pvalue:.6f}")

# Type III ANOVA
anova_type3 = sm.stats.anova_lm(model_words, typ=2)
print(f"\nType II ANOVA Table:")
print(anova_type3.to_string())


# ============================================================
# 8. ANALYSIS 6: GLM - Poisson Regression for Word Counts
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 6: GENERALIZED LINEAR MODEL - Poisson Regression")
print("=" * 70)

glm_data = master.dropna(subset=['hindi_proficiency', 'total_words', 'domain']).copy()

# Poisson regression (word count is count data)
glm_poisson = smf.glm('total_words ~ hindi_proficiency + english_proficiency + C(domain)', 
                        data=glm_data,
                        family=sm.families.Poisson()).fit()
print(f"\nPoisson GLM: Total Words ~ Hindi Prof + English Prof + Domain")
print(glm_poisson.summary2().tables[1].to_string())
print(f"Deviance = {glm_poisson.deviance:.4f}")
print(f"Pearson chi2 = {glm_poisson.pearson_chi2:.4f}")
print(f"AIC = {glm_poisson.aic:.4f}")

# Check for overdispersion
dispersion = glm_poisson.pearson_chi2 / glm_poisson.df_resid
print(f"Dispersion parameter: {dispersion:.4f} (>1 indicates overdispersion)")

# Negative Binomial if overdispersed
if dispersion > 1.5:
    print("\nOverdispersion detected. Fitting Negative Binomial GLM...")
    try:
        glm_nb = smf.glm('total_words ~ hindi_proficiency + english_proficiency + C(domain)',
                          data=glm_data,
                          family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        print(f"Negative Binomial GLM:")
        print(glm_nb.summary2().tables[1].to_string())
        print(f"AIC = {glm_nb.aic:.4f}")
    except Exception as e:
        print(f"NB model failed: {e}")

# GLM for IRT (Gamma family - continuous positive) 
print(f"\n--- Gamma GLM for Mean IRT ---")
glm_gamma_data = master.dropna(subset=['mean_irt', 'hindi_proficiency', 'domain']).copy()
glm_gamma_data = glm_gamma_data[glm_gamma_data['mean_irt'] > 0]

try:
    glm_gamma = smf.glm('mean_irt ~ hindi_proficiency + total_words + C(domain)', 
                          data=glm_gamma_data,
                          family=sm.families.Gamma(sm.families.links.Log())).fit()
    print(f"Gamma GLM: Mean IRT ~ Hindi Prof + Total Words + Domain")
    print(glm_gamma.summary2().tables[1].to_string())
    print(f"Deviance = {glm_gamma.deviance:.4f}")
    print(f"AIC = {glm_gamma.aic:.4f}")
except Exception as e:
    print(f"Gamma GLM failed: {e}")

# GLM Plot - predicted vs actual
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Poisson predictions
glm_data['predicted_words'] = glm_poisson.fittedvalues
axes[0].scatter(glm_data['total_words'], glm_data['predicted_words'], alpha=0.5)
max_val = max(glm_data['total_words'].max(), glm_data['predicted_words'].max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
axes[0].set_xlabel('Actual Total Words')
axes[0].set_ylabel('Predicted Total Words')
axes[0].set_title('Poisson GLM: Predicted vs Actual Word Counts')

# Domain-wise comparison
domain_means = glm_data.groupby('domain').agg(
    actual=('total_words', 'mean'),
    predicted=('predicted_words', 'mean')
).reset_index()
x_pos = range(len(domain_means))
width = 0.35
axes[1].bar([p - width/2 for p in x_pos], domain_means['actual'], width, label='Actual', alpha=0.7)
axes[1].bar([p + width/2 for p in x_pos], domain_means['predicted'], width, label='Predicted', alpha=0.7)
axes[1].set_xticks(list(x_pos))
axes[1].set_xticklabels(domain_means['domain'])
axes[1].set_ylabel('Mean Total Words')
axes[1].set_title('GLM: Actual vs Predicted by Domain')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'glm_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[Saved: glm_results.png]")


# ============================================================
# 9. ANALYSIS 7: BAYESIAN ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 7: BAYESIAN STATISTICAL ANALYSIS")
print("=" * 70)

# Bayesian t-test comparing IRT between proficiency groups
print("\n--- Bayesian Independent t-test: IRT by Proficiency Group ---")
high_irt = master[master['prof_group'] == 'High']['mean_irt'].dropna()
low_irt = master[master['prof_group'] == 'Low']['mean_irt'].dropna()

bayes_t = pg.ttest(high_irt, low_irt, paired=False)
print(bayes_t.to_string())

# Bayesian correlation: spatial distance vs IRT
if len(df_consec) > 0:
    print("\n--- Bayesian Correlation: Spatial Distance vs IRT ---")
    bayes_corr = pg.corr(df_consec['spatial_distance'], df_consec['irt'], method='pearson')
    print(bayes_corr.to_string())
    
    # Compute Bayes Factor manually using BIC approximation
    # BF10 approximation from t-statistic
    n = len(df_consec)
    t_val = bayes_corr['r'].values[0] * np.sqrt(n - 2) / np.sqrt(1 - bayes_corr['r'].values[0]**2)
    # JZS Bayes Factor approximation
    bf10_approx = np.sqrt(n) * np.exp(-0.5 * t_val**2 / n)
    # Using pingouin's built-in BF
    print(f"  Bayes Factor (BF10): {bayes_corr['BF10'].values[0]}")

# Bayesian ANOVA equivalent
print("\n--- Bayesian ANOVA: Total Words by Domain ---")
bayes_anova_data = master[master['domain'].isin(['animals', 'foods'])].copy()
if len(bayes_anova_data) > 0:
    # Compute BF for ANOVA using pingouin
    bayes_aov = pg.anova(data=master[master['domain'].isin(['animals', 'foods', 'colours', 'body-parts'])], 
                          dv='total_words', between='domain', detailed=True)
    print(bayes_aov.to_string())

# Bayesian Regression using PyMC
print("\n--- Bayesian Linear Regression (PyMC) ---")
try:
    import pymc as pm
    import arviz as az
    
    # Prepare data for Bayesian regression
    bayes_reg_data = master.dropna(subset=['hindi_proficiency', 'total_words']).copy()
    bayes_reg_data = bayes_reg_data[bayes_reg_data['domain'].isin(['animals', 'foods'])].copy()
    
    # Standardize predictor
    bayes_reg_data['hindi_prof_z'] = (bayes_reg_data['hindi_proficiency'] - bayes_reg_data['hindi_proficiency'].mean()) / bayes_reg_data['hindi_proficiency'].std()
    
    with pm.Model() as bayes_model:
        # Priors
        alpha = pm.Normal('alpha', mu=10, sigma=5)
        beta_prof = pm.Normal('beta_hindi_prof', mu=0, sigma=3)
        sigma = pm.HalfNormal('sigma', sigma=5)
        
        # Likelihood
        mu = alpha + beta_prof * bayes_reg_data['hindi_prof_z'].values
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=bayes_reg_data['total_words'].values)
        
        # Sample
        trace = pm.sample(2000, tune=1000, cores=1, random_seed=42, 
                          progressbar=True, return_inferencedata=True)
    
    # Summary
    summary = az.summary(trace, var_names=['alpha', 'beta_hindi_prof', 'sigma'])
    print("\nBayesian Regression Summary:")
    print(summary.to_string())
    
    # HDI (Highest Density Interval)
    print(f"\n95% HDI for beta_hindi_prof:")
    hdi = az.hdi(trace, var_names=['beta_hindi_prof'], hdi_prob=0.95)
    print(f"  {hdi}")
    
    # Plot posterior
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    az.plot_posterior(trace, var_names=['alpha'], ax=axes[0])
    az.plot_posterior(trace, var_names=['beta_hindi_prof'], ax=axes[1])
    az.plot_posterior(trace, var_names=['sigma'], ax=axes[2])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bayesian_posteriors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Saved: bayesian_posteriors.png]")
    
    # Trace plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    az.plot_trace(trace, var_names=['alpha', 'beta_hindi_prof', 'sigma'], axes=axes)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bayesian_traceplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Saved: bayesian_traceplot.png]")
    
except Exception as e:
    print(f"PyMC Bayesian analysis error: {e}")
    print("Falling back to Bayesian Information Criterion comparison...")
    
    # BIC-based Bayes Factor approximation
    reg_data_bic = master.dropna(subset=['hindi_proficiency', 'total_words', 'domain']).copy()
    
    # Null model
    null_model = smf.ols('total_words ~ 1', data=reg_data_bic).fit()
    # Full model
    full_model = smf.ols('total_words ~ hindi_proficiency + C(domain)', data=reg_data_bic).fit()
    
    bic_null = null_model.bic
    bic_full = full_model.bic
    delta_bic = bic_null - bic_full
    bf10_bic = np.exp(delta_bic / 2)
    
    print(f"\nBIC Comparison:")
    print(f"  Null Model BIC: {bic_null:.2f}")
    print(f"  Full Model BIC: {bic_full:.2f}")
    print(f"  Delta BIC: {delta_bic:.2f}")
    print(f"  Approximate BF10: {bf10_bic:.4f}")
    
    if bf10_bic > 100:
        evidence = "Extreme evidence for H1"
    elif bf10_bic > 30:
        evidence = "Very strong evidence for H1"
    elif bf10_bic > 10:
        evidence = "Strong evidence for H1"
    elif bf10_bic > 3:
        evidence = "Moderate evidence for H1"
    elif bf10_bic > 1:
        evidence = "Anecdotal evidence for H1"
    elif bf10_bic > 1/3:
        evidence = "Anecdotal evidence for H0"
    elif bf10_bic > 1/10:
        evidence = "Moderate evidence for H0"
    else:
        evidence = "Strong evidence for H0"
    print(f"  Interpretation: {evidence}")


# ============================================================
# 10. ANALYSIS 8: ANCOVA - IRT with Proficiency as Covariate
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 8: ANCOVA - Mean IRT across Domains (Covariate: Hindi Prof)")
print("=" * 70)

ancova_data = master.dropna(subset=['mean_irt', 'hindi_proficiency', 'domain']).copy()
ancova_result = pg.ancova(data=ancova_data, dv='mean_irt', between='domain', 
                           covar='hindi_proficiency')
print(ancova_result.to_string())


# ============================================================
# 11. ANALYSIS 9: Serial Position Effect (Word Order -> IRT)
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 9: Serial Position Effect on IRT")
print("=" * 70)

# Regression: word_order predicting IRT
serial_data = df_vft[~df_vft['is_first_word']].copy()
serial_data['log_irt'] = np.log(serial_data['irt'].clip(lower=1))

model_serial = smf.ols('log_irt ~ word_order + C(domain)', data=serial_data).fit()
print(f"\nLinear Regression: log(IRT) ~ Word Order + Domain")
print(model_serial.summary2().tables[1].to_string())
print(f"R² = {model_serial.rsquared:.4f}")

# Plot serial position effect
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mean IRT by word position
serial_summary = df_vft.groupby('word_order')['irt'].agg(['mean', 'sem']).reset_index()
serial_summary = serial_summary[serial_summary['word_order'] <= 20]
axes[0].errorbar(serial_summary['word_order'], serial_summary['mean'], 
                  yerr=serial_summary['sem'], marker='o', capsize=3, linewidth=2)
axes[0].set_xlabel('Word Position (Serial Order)')
axes[0].set_ylabel('Mean IRT (ms)')
axes[0].set_title('Serial Position Effect on IRT')

# By domain
for domain in ['animals', 'foods']:
    dom_serial = df_vft[df_vft['domain'] == domain].groupby('word_order')['irt'].agg(['mean', 'sem']).reset_index()
    dom_serial = dom_serial[dom_serial['word_order'] <= 20]
    axes[1].errorbar(dom_serial['word_order'], dom_serial['mean'], 
                      yerr=dom_serial['sem'], marker='o', capsize=3, linewidth=2, label=domain)
axes[1].set_xlabel('Word Position (Serial Order)')
axes[1].set_ylabel('Mean IRT (ms)')
axes[1].set_title('Serial Position Effect by Domain')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'serial_position_effect.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Saved: serial_position_effect.png]")


# ============================================================
# 12. ANALYSIS 10: GLM for IRT at Word Level
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 10: GLM (Gamma) - Word-Level IRT Prediction")
print("=" * 70)

word_level = df_vft[~df_vft['is_first_word']].copy()
word_level = word_level[word_level['irt'] > 0]

try:
    glm_word = smf.glm('irt ~ word_order + C(domain)', 
                         data=word_level,
                         family=sm.families.Gamma(sm.families.links.Log())).fit()
    print(f"Gamma GLM: IRT ~ Word Order + Domain")
    print(glm_word.summary2().tables[1].to_string())
    print(f"Deviance = {glm_word.deviance:.4f}")
    print(f"AIC = {glm_word.aic:.4f}")
except Exception as e:
    print(f"Gamma GLM error: {e}")


# ============================================================
# 13. ADDITIONAL VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING ADDITIONAL VISUALIZATIONS")
print("=" * 70)

# Proficiency vs Word Production by Domain
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, domain in enumerate(['animals', 'foods']):
    dom_data = master[master['domain'] == domain]
    axes[i].scatter(dom_data['hindi_proficiency'], dom_data['total_words'], alpha=0.6, s=50)
    # Add regression line
    mask = dom_data[['hindi_proficiency', 'total_words']].dropna()
    if len(mask) > 2:
        z = np.polyfit(mask['hindi_proficiency'], mask['total_words'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(mask['hindi_proficiency'].min(), mask['hindi_proficiency'].max(), 100)
        axes[i].plot(x_line, p(x_line), 'r--', linewidth=2)
    axes[i].set_xlabel('Hindi Proficiency')
    axes[i].set_ylabel('Total Words')
    axes[i].set_title(f'{domain.capitalize()}: Proficiency vs Word Production')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'proficiency_vs_words.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Saved: proficiency_vs_words.png]")

# IRT Distribution by Domain
fig, ax = plt.subplots(figsize=(10, 6))
for domain in ['animals', 'foods', 'colours', 'body-parts']:
    dom_irt = df_vft[df_vft['domain'] == domain]['irt']
    if len(dom_irt) > 0:
        ax.hist(dom_irt, bins=50, alpha=0.4, label=domain, density=True)
ax.set_xlabel('IRT (ms)')
ax.set_ylabel('Density')
ax.set_title('IRT Distribution by Domain')
ax.legend()
ax.set_xlim(0, 30000)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'irt_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Saved: irt_distribution.png]")

# Heatmap: Correlations among all key variables
corr_vars = master[['total_words', 'mean_irt', 'median_irt', 'sd_irt',
                      'mean_pairwise_dist', 'hindi_proficiency', 'english_proficiency']].dropna()
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = corr_vars.corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
            square=True, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation Matrix: Key Variables')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Saved: correlation_heatmap.png]")

# SpAM spread by domain
fig, ax = plt.subplots(figsize=(10, 6))
spam_spread = master.dropna(subset=['mean_pairwise_dist'])
if len(spam_spread) > 0:
    available_domains = [d for d in ['colours', 'animals', 'foods', 'body-parts'] if d in spam_spread['domain'].unique()]
    sns.boxplot(data=spam_spread, x='domain', y='mean_pairwise_dist', 
                order=available_domains, palette='viridis', ax=ax)
    ax.set_title('Semantic Spread (SpAM Pairwise Distance) by Domain')
    ax.set_xlabel('Domain')
    ax.set_ylabel('Mean Pairwise Distance')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'spam_spread_by_domain.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Saved: spam_spread_by_domain.png]")


print("\n" + "=" * 70)
print("ALL ANALYSES COMPLETE")
print("=" * 70)
print(f"\nFigures saved to: {OUTPUT_DIR}")
print(f"Total figures: {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])}")
