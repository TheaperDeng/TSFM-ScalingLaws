# load dataset in dataset_influence_scores.csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
# include negative scores but use positive total as denominator

# print the first 5 rows
print("First 5 rows:")
print(df.head())

print(f"\nDataFrame shape before cleaning: {df.shape}")

# Remove duplicate header rows that contain 'score' in the score column
df_clean = df[df['score'] != 'score'].copy()

print(f"DataFrame shape after removing header rows: {df_clean.shape}")
print(f"Removed {df.shape[0] - df_clean.shape[0]} duplicate header rows")

# change score, step, epoch and count to numeric types
df_clean['score'] = df_clean['score'].astype(float)
df_clean['step'] = df_clean['step'].astype(int)
df_clean['epoch'] = df_clean['epoch'].astype(int)
df_clean['count'] = df_clean['count'].astype(int)

# Keep ALL scores (including negative ones)
print(f"DataFrame shape (keeping all scores): {df_clean.shape}")
print(f"Number of positive scores: {(df_clean['score'] > 0).sum()}")
print(f"Number of negative scores: {(df_clean['score'] < 0).sum()}")
print(f"Number of zero scores: {(df_clean['score'] == 0).sum()}")

# group by "step" column and calculate the ratio of each dataset within the same step
df_grouped = df_clean.groupby('step')

# calculate the ratio of each dataset within the same step
# use sum of positive scores as denominator for all items
def calculate_ratios_with_negative(group):
    group = group.copy()
    # Calculate total of POSITIVE scores only
    positive_total = group[group['score'] > 0]['score'].sum()
    
    if positive_total > 0:
        # Apply this denominator to ALL scores (positive and negative)
        group['ratio'] = group['score'] / positive_total
    else:
        # Handle edge case where no positive scores exist
        group['ratio'] = 0
    
    return group

df_with_ratios = df_grouped.apply(calculate_ratios_with_negative).reset_index(drop=True)

# Debug: Check if ratios are calculated correctly for each step
print("Debug - checking ratio calculations for first few steps:")
for step in sorted(df_with_ratios['step'].unique())[:3]:
    step_data = df_with_ratios[df_with_ratios['step'] == step]
    positive_sum = step_data[step_data['ratio'] > 0]['ratio'].sum()
    negative_sum = step_data[step_data['ratio'] < 0]['ratio'].sum()
    print(f"Step {step}: positive_sum={positive_sum:.6f}, negative_sum={negative_sum:.6f}, total={positive_sum + negative_sum:.6f}")
    print(f"  Number of datasets: {len(step_data)}, positive: {(step_data['ratio'] > 0).sum()}, negative: {(step_data['ratio'] < 0).sum()}")
    if abs(positive_sum - 1.0) > 0.001:
        print(f"  WARNING: Positive ratios don't sum to 1!")
        print(f"  Positive scores: {step_data[step_data['score'] > 0]['score'].tolist()}")
        print(f"  Positive ratios: {step_data[step_data['ratio'] > 0]['ratio'].tolist()}")
    print()

print("Sample of data with ratios (including negative):")
print(df_with_ratios.head(10))

# Show some statistics about ratios
print(f"\nRatio statistics:")
print(f"Positive ratios: {(df_with_ratios['ratio'] > 0).sum()}")
print(f"Negative ratios: {(df_with_ratios['ratio'] < 0).sum()}")
print(f"Zero ratios: {(df_with_ratios['ratio'] == 0).sum()}")

# Create a pivot table for plotting - datasets as columns, steps as index
pivot_data = df_with_ratios.pivot_table(
    index='step', 
    columns='dataset_name', 
    values='ratio', 
    fill_value=0
)

print(f"\nPivot table shape: {pivot_data.shape}")
print(f"Number of unique steps: {len(pivot_data.index)}")
print(f"Number of unique datasets: {len(pivot_data.columns)}")

# Show which datasets contribute most (by absolute value)
dataset_total_contribution = pivot_data.sum().sort_values(ascending=False, key=abs)
print(f"\nTop 10 datasets by total absolute contribution across all steps:")
print(dataset_total_contribution.head(10))

# Order datasets by their contribution in the LAST step (by absolute value)
last_step = pivot_data.index.max()
last_step_contribution = pivot_data.loc[last_step].sort_values(ascending=False, key=abs)

print(f"\nTop 10 datasets by absolute contribution in the last step (step {last_step}):")
print(last_step_contribution.head(10))

# For better visualization, let's focus on top contributing datasets from the last step
top_datasets = last_step_contribution.head(10).index.tolist()
pivot_top = pivot_data[top_datasets].copy()

# Add an "Others" category for remaining datasets - but handle positive and negative separately
others_datasets = pivot_data.drop(columns=top_datasets)

# Create separate positive and negative "Others" contributions
others_positive = others_datasets.copy()
others_negative = others_datasets.copy()
others_positive[others_positive < 0] = 0
others_negative[others_negative > 0] = 0

others_positive_sum = others_positive.sum(axis=1)
others_negative_sum = others_negative.sum(axis=1)

# Add both positive and negative "Others" to the data
pivot_top['Others_positive'] = others_positive_sum
pivot_top['Others_negative'] = others_negative_sum

print(f"\nCreating plot for top {len(top_datasets)} datasets plus 'Others' (ordered by last step absolute contribution)")

# Separate positive and negative contributions for plotting
pivot_positive = pivot_top.copy()
pivot_negative = pivot_top.copy()

# Set negative values to 0 in positive data, positive values to 0 in negative data
pivot_positive[pivot_positive < 0] = 0
pivot_negative[pivot_negative > 0] = 0

# For the "Others_negative" column, we want its values in the negative plot
pivot_positive['Others_negative'] = 0  # Remove from positive
pivot_negative['Others_positive'] = 0  # Remove from negative

# Rename columns for clarity in the legend
# datasets = list(top_datasets) + ['Others']
datasets = ['Others'] + top_datasets[::-1]
pivot_positive = pivot_positive[top_datasets + ['Others_positive']].rename(columns={'Others_positive': 'Others'})
pivot_negative = pivot_negative[top_datasets + ['Others_negative']].rename(columns={'Others_negative': 'Others'})

# Create the plot with same coloring as create_improved_stackplot.py
plt.figure(figsize=(16, 10))
steps = pivot_positive.index

# Create stackplot for positive contributions (above 0) - using default matplotlib colors
positive_artists = plt.stackplot(steps, *[pivot_positive[dataset] for dataset in datasets], 
                                labels=[f"{dataset} (positive)" for dataset in datasets], 
                                alpha=0.85)

# Create stackplot for negative contributions (below 0) - using different colors (darker/complementary)
# Use a different colormap for negative contributions to distinguish them
negative_colors = plt.cm.Dark2(np.linspace(0, 1, len(datasets)))
negative_artists = plt.stackplot(steps, *[pivot_negative[dataset] for dataset in datasets], 
                                labels=[f"{dataset} (negative)" for dataset in datasets], 
                                alpha=0.85, colors=negative_colors)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.xlabel('Training Step', fontsize=14)
plt.ylabel('Contribution Ratio', fontsize=14)
plt.title('Dataset Contribution Ratios Over Training Steps\n(Including Negative Contributions, Ordered by Last Step Absolute Contribution)', fontsize=16)

# Create custom legend showing both positive and negative colors for datasets
handles = []
labels = []
for i, dataset in enumerate(datasets):
    # Add positive color
    pos_color = positive_artists[i].get_facecolor()[0]
    handles.append(plt.Rectangle((0,0),1,1, color=pos_color, alpha=0.85))
    labels.append(f"{dataset} (+)")
    
    # Add negative color
    neg_color = negative_colors[i]
    handles.append(plt.Rectangle((0,0),1,1, color=neg_color, alpha=0.85))
    labels.append(f"{dataset} (-)")

plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, 
           title='Datasets', title_fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('dataset_contribution_with_negative.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'dataset_contribution_with_negative.png'")

# Also save the pivot table for further analysis
pivot_data.to_csv('dataset_contribution_ratios_with_negative.csv')
print("Full contribution ratios (with negative) saved as 'dataset_contribution_ratios_with_negative.csv'")

# Verify that positive contributions in the PLOTTED data sum to 1 for each step
print(f"\nVerification - Positive contributions in plotted data should sum to 1.0 for each step:")
plotted_positive_sums = pivot_positive.sum(axis=1)
print(f"Sample of plotted positive sums for first 5 steps: {plotted_positive_sums.head().tolist()}")
print(f"All plotted positive sums equal to 1.0: {np.allclose(plotted_positive_sums, 1.0)}")
if not np.allclose(plotted_positive_sums, 1.0):
    print(f"WARNING: Plotted positive sums range from {plotted_positive_sums.min():.6f} to {plotted_positive_sums.max():.6f}")

# Verify that positive contributions sum to 1 for each step
print(f"\nVerification - All positive contributions should sum to 1.0 for each step:")
# Create a copy where negative values are set to 0, then sum across columns
pivot_positive_only = pivot_data.copy()
pivot_positive_only[pivot_positive_only < 0] = 0
positive_sums = pivot_positive_only.sum(axis=1)
print(f"Sample of positive sums for first 5 steps: {positive_sums.head().tolist()}")
print(f"All positive sums equal to 1.0: {np.allclose(positive_sums, 1.0)}")
if not np.allclose(positive_sums, 1.0):
    print(f"WARNING: Positive sums range from {positive_sums.min():.6f} to {positive_sums.max():.6f}")
    
    # Additional debugging
    print(f"Steps with positive sum != 1.0:")
    for step in pivot_data.index:
        step_positive_sum = pivot_positive_only.loc[step].sum()
        if abs(step_positive_sum - 1.0) > 0.001:
            print(f"  Step {step}: {step_positive_sum:.6f}")
            # Show the original data for this step
            step_orig = df_with_ratios[df_with_ratios['step'] == step]
            pos_scores = step_orig[step_orig['score'] > 0]['score']
            pos_ratios = step_orig[step_orig['score'] > 0]['ratio']
            print(f"    Positive scores sum: {pos_scores.sum():.6f}")
            print(f"    Positive ratios sum: {pos_ratios.sum():.6f}")
            break  # Just show one example

# Print some summary statistics
print(f"\nSummary Statistics:")
print(f"Steps range: {pivot_data.index.min()} to {pivot_data.index.max()}")
print(f"Total positive contribution in last step (plotted): {pivot_positive.loc[last_step].sum():.4f}")
print(f"Total negative contribution in last step (plotted): {pivot_negative.loc[last_step].sum():.4f}")
print(f"Total positive contribution in last step (all data): {pivot_data.loc[last_step][pivot_data.loc[last_step] > 0].sum():.4f}")
print(f"Total negative contribution in last step (all data): {pivot_data.loc[last_step][pivot_data.loc[last_step] < 0].sum():.4f}")
print(f"Net contribution in last step: {pivot_data.loc[last_step].sum():.4f}")

plt.show() 