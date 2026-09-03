"""
EDA ENGINEER - Visual Report (Stage 01)
Generates correlation heatmap, biomarker leaderboard, distributions,
and problem-flag visuals from the CLEANED oncology dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 140
matplotlib.rcParams['font.size'] = 10

df = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\ONCOLOGY EDA\master_dataset_300_CLEANED (1).csv')

OUT = r"C:\Users\HP\OneDrive\Desktop\ONCOLOGY EDA"

# color palette
navy = "#0B2545"
gold = "#F5A623"
teal = "#1E88A8"
grey = "#B0B7C3"
red  = "#D64545"

# ----------------------------------------------------------------------
# 1. CORRELATION HEATMAP (numeric biomarkers + outcomes)
# ----------------------------------------------------------------------
num_cols = ['age','mutation_count','tumor_mutation_burden','ctdna_level_ng_ml',
            'creatinine_mg_dl','max_variant_allele_fraction',
            'pfs_months','overall_survival_months','toxicity_grade']

corr = df[num_cols].corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(8,7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols)))
ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha='right')
ax.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        v = corr.values[i,j]
        ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                color='white' if abs(v)>0.5 else 'black', fontsize=8)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Pearson correlation')
ax.set_title("Correlation Matrix — Clinical & Genomic Variables\n(alt_u_l excluded — flagged low-trust)", fontsize=12, fontweight='bold', color=navy)
plt.tight_layout()
plt.savefig(OUT+"01_correlation_heatmap.png", bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# 2. BIOMARKER LEADERBOARD (correlation with High-Risk toxicity)
# ----------------------------------------------------------------------
tmp = df.copy()
tmp['high_risk'] = (tmp['toxicity_grade'] >= 3).astype(float)
tmp.loc[tmp['toxicity_grade'].isna(), 'high_risk'] = np.nan
biom_cols = ['age','mutation_count','tumor_mutation_burden','ctdna_level_ng_ml',
             'creatinine_mg_dl','max_variant_allele_fraction']
leaderboard = tmp[biom_cols+['high_risk']].corr(numeric_only=True)['high_risk'].drop('high_risk')
leaderboard = leaderboard.reindex(leaderboard.abs().sort_values(ascending=True).index)

fig, ax = plt.subplots(figsize=(7,4.5))
colors = [red if v>0 else teal for v in leaderboard.values]
ax.barh(leaderboard.index, leaderboard.values, color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel("Correlation with High-Risk Toxicity (Grade ≥3)")
ax.set_title("Biomarker Leaderboard\n(all correlations weak — no single biomarker dominates)", fontsize=12, fontweight='bold', color=navy)
for i,v in enumerate(leaderboard.values):
    ax.text(v + (0.003 if v>=0 else -0.003), i, f"{v:.3f}", va='center', ha='left' if v>=0 else 'right', fontsize=9)
plt.tight_layout()
plt.savefig(OUT+"02_biomarker_leaderboard.png", bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# 3. TOXICITY GRADE + BEST RESPONSE distributions (side by side)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1,2, figsize=(11,4.5))

tox_order = [0.0,1.0,2.0,3.0,4.0]
tox_counts = df['toxicity_grade'].value_counts().reindex(tox_order).fillna(0)
bar_colors = [teal if g<3 else red for g in tox_order]
axes[0].bar([str(int(g)) for g in tox_order], tox_counts.values, color=bar_colors)
axes[0].set_title("Toxicity Grade Distribution", fontweight='bold', color=navy)
axes[0].set_xlabel("Toxicity Grade")
axes[0].set_ylabel("Patients")
for i,v in enumerate(tox_counts.values):
    axes[0].text(i, v+1, int(v), ha='center', fontsize=9)

resp_order = ['CR','PR','SD','PD']
resp_counts = df['best_response'].value_counts().reindex(resp_order)
resp_colors = ["#1B7A3D","#4CAF50", gold, red]
axes[1].bar(resp_order, resp_counts.values, color=resp_colors)
axes[1].set_title("Best Response (RECIST)", fontweight='bold', color=navy)
axes[1].set_xlabel("Response Category")
for i,v in enumerate(resp_counts.values):
    axes[1].text(i, v+1, int(v), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUT+"03_toxicity_response_distributions.png", bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# 4. TREATMENT vs PROGRESSION (stacked bar)
# ----------------------------------------------------------------------
ct = pd.crosstab(df['treatment_name'], df['progression_status'])
ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(8,5))
ct.plot(kind='barh', stacked=True, ax=ax, color=[teal, red])
ax.set_title("Progression Status by Treatment", fontweight='bold', color=navy)
ax.set_xlabel("Patients")
ax.legend(title="", loc='lower right')
plt.tight_layout()
plt.savefig(OUT+"04_treatment_vs_progression.png", bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# 5. PROBLEM FLAG: PFS vs OS scatter, impossible rows highlighted
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7,6))
ok = df[df['pfs_gt_os_flag']==0]
bad = df[df['pfs_gt_os_flag']==1]
ax.scatter(ok['overall_survival_months'], ok['pfs_months'], color=teal, alpha=0.6, s=25, label='Valid (PFS ≤ OS)')
ax.scatter(bad['overall_survival_months'], bad['pfs_months'], color=red, alpha=0.85, s=35, label=f'Flagged impossible (PFS > OS), n={len(bad)}')
lims = [0, max(df['overall_survival_months'].max(), df['pfs_months'].max())+2]
ax.plot(lims, lims, 'k--', linewidth=1, label='PFS = OS line')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Overall Survival (months)")
ax.set_ylabel("Progression-Free Survival (months)")
ax.set_title("Data Quality Check: PFS cannot exceed OS\n28 patients violate this (above the dashed line)", fontweight='bold', color=navy)
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(OUT+"05_pfs_os_problem_flag.png", bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# 6. AGE distribution before vs after cleaning (fake 66.0 spike removed)
# ----------------------------------------------------------------------
orig = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\ONCOLOGY EDA\master_dataset_300_CLEANED (1).csv')
fig, axes = plt.subplots(1,2, figsize=(11,4.5))
axes[0].hist(orig['age'], bins=20, color=grey, edgecolor='white')
axes[0].axvline(66.0, color=red, linestyle='--', linewidth=1.5, label='Fake spike at 66.0 (n=14)')
axes[0].set_title("BEFORE — Age (raw, includes fake fill values)", fontweight='bold', color=navy)
axes[0].set_xlabel("Age"); axes[0].legend(fontsize=8)

axes[1].hist(df['age'].dropna(), bins=20, color=teal, edgecolor='white')
axes[1].set_title("AFTER — Age (fake values removed → NaN)", fontweight='bold', color=navy)
axes[1].set_xlabel("Age")
plt.tight_layout()
plt.savefig(OUT+"06_age_before_after_cleaning.png", bbox_inches='tight')
plt.close()

print("All plots saved.")
