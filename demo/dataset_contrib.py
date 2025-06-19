# load dataset in dataset_influence_scores.csv
import pandas as pd

df = pd.read_csv('../dataset_influence_scores_backup06121234.csv')

# some lines in the df
# dataset_name,score,step,epoch,count
# solar_power,0.10994826323968532,1,0,1
# largest_2021,0.030228186922926836,1,0,2
# largest_2018,0.028586113436268157,1,0,1
# largest_2019,0.028563616374733296,1,0,3
# largest_2020,0.026521975763966532,1,0,2
# largest_2017,0.019620712713701582,1,0,1
# CMIP6_dataset,0.01195891705930291,1,0,4
# ERA5_dataset,0.011084774521802208,1,0,9
# azure_vm_traces_2017,0.005618669942121637,1,0,1
# CloudOpsTSF_dataset,0.003836815524386812,1,0,1

# calculate the contribution of each dataset within the same step
# the contribution is evaluated by influence score
# drop the negative ones and calculate the ratio of each dataset within the same step

# print the first 5 rows
print("First 5 rows:")
print(df.head())

print(f"\nDataFrame shape before cleaning: {df.shape}")

# Remove duplicate header rows that contain 'score' in the score column
df_clean = df[df['score'] != 'score'].copy()

print(f"DataFrame shape after removing header rows: {df_clean.shape}")
print(f"Removed {df.shape[0] - df_clean.shape[0]} duplicate header rows")

# change score, step, epoch and count to float
# so we need to convert the score column to float
df_clean['score'] = df_clean['score'].astype(float)
df_clean['step'] = df_clean['step'].astype(int)
df_clean['epoch'] = df_clean['epoch'].astype(int)
df_clean['count'] = df_clean['count'].astype(int)

# drop the negative ones
df_clean = df_clean[df_clean['score'] > 0]

print(f"DataFrame shape after removing negative scores: {df_clean.shape}")

# group by "step" column and calculate the ratio of each dataset within the same step
df_grouped = df_clean.groupby('step')

# calculate the ratio of each dataset within the same step
def calculate_ratios(group):
    group = group.copy()
    total_score = group['score'].sum()
    group['ratio'] = group['score'] / total_score
    return group

df_with_ratios = df_grouped.apply(calculate_ratios).reset_index(drop=True)

print("Sample of data with ratios:")
print(df_with_ratios.head(10))

# Create a pivot table for stackplot - datasets as columns, steps as index
pivot_data = df_with_ratios.pivot_table(
    index='step', 
    columns='dataset_name', 
    values='ratio', 
    fill_value=0
)

print(f"\nPivot table shape: {pivot_data.shape}")
print(f"Number of unique steps: {len(pivot_data.index)}")
print(f"Number of unique datasets: {len(pivot_data.columns)}")

# Show which datasets contribute most
dataset_total_contribution = pivot_data.sum().sort_values(ascending=False)
print(f"\nTop 10 datasets by total contribution across all steps:")
print(dataset_total_contribution.head(10))

# Create stackplot
import matplotlib.pyplot as plt
import numpy as np

# Order datasets by their contribution in the LAST step instead of total contribution
last_step = pivot_data.index.max()
last_step_contribution = pivot_data.loc[last_step].sort_values(ascending=False)

print(f"\nTop 10 datasets by contribution in the last step (step {last_step}):")
print(last_step_contribution.head(10))

# For better visualization, let's focus on top contributing datasets from the last step
top_datasets = last_step_contribution.head(10).index.tolist()
pivot_top = pivot_data[top_datasets]

# Add an "Others" category for remaining datasets
others_data = pivot_data.drop(columns=top_datasets).sum(axis=1)
pivot_top['Others'] = others_data

print(f"\nCreating stackplot for top {len(top_datasets)} datasets plus 'Others' (ordered by last step contribution)")

# Create the plot with distinguishable colors
plt.figure(figsize=(15, 8))
steps = pivot_top.index
datasets = pivot_top.columns

# Use a colormap that provides distinguishable colors
colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
# Alternative color options for better distinction:
# colors = plt.cm.tab20(np.linspace(0, 1, len(datasets)))
# colors = plt.cm.Paired(np.linspace(0, 1, len(datasets)))

# Create stackplot with custom colors
plt.stackplot(steps, *[pivot_top[dataset] for dataset in datasets], 
              labels=datasets, alpha=0.8, colors=colors)

plt.xlabel('Step')
plt.ylabel('Contribution Ratio')
plt.title('Dataset Contribution Ratios Over Training Steps\n(Ordered by Last Step Contribution)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('dataset_contribution_stackplot_last_step_ordered.png', dpi=300, bbox_inches='tight')
print("Stackplot saved as 'dataset_contribution_stackplot_last_step_ordered.png'")

# Also save the pivot table for further analysis
pivot_data.to_csv('dataset_contribution_ratios.csv')
print("Full contribution ratios saved as 'dataset_contribution_ratios.csv'")

plt.show()

