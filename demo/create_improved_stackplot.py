import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the processed data
pivot_data = pd.read_csv('dataset_contribution_ratios.csv', index_col=0)

# Order datasets by their contribution in the LAST step
last_step = pivot_data.index.max()
last_step_contribution = pivot_data.loc[last_step].sort_values(ascending=False)

print(f"Top 10 datasets by contribution in the last step (step {last_step}):")
print(last_step_contribution.head(10))

# Get top datasets ordered by last step contribution
top_datasets = last_step_contribution.head(10).index.tolist()
pivot_top = pivot_data[top_datasets].copy()

# Add "Others" category
others_data = pivot_data.drop(columns=top_datasets).sum(axis=1)
pivot_top['Others'] = others_data

# Reorder columns to put "Others" at the top of stackplot
# For stackplot, the first dataset in the list appears at the bottom
# So we want: highest contributors first (bottom), then "Others" last (top)
datasets_ordered = ['Others'] + top_datasets[::-1]
pivot_final = pivot_top[datasets_ordered]

steps = pivot_final.index

print(f"\nDataset order in stackplot (from bottom to top):")
for i, dataset in enumerate(datasets_ordered, 1):
    if dataset == 'Others':
        contribution = others_data.iloc[-1]
    else:
        contribution = last_step_contribution[dataset]
    print(f"{i:2d}. {dataset:<20} : {contribution:.6f}")

# Create the final stackplot with tab20 colors
plt.figure(figsize=(16, 10))

# Use tab20 colormap for maximum distinction
# colors_tab20 = plt.cm.tab20(np.linspace(0, 1, len(datasets_ordered)))

# Create stackplot (first dataset appears at bottom)
plt.stackplot(steps, *[pivot_final[dataset] for dataset in datasets_ordered], 
              labels=datasets_ordered, alpha=0.85)#, colors=colors_tab20)

plt.xlabel('Training Step', fontsize=14)
plt.ylabel('Contribution Ratio', fontsize=14)
plt.title('Dataset Contribution Ratios Over Training Steps\n(Ordered by Last Step Contribution, tab20 Colors)', fontsize=16)

# Add legend with better positioning
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, 
           title='Datasets', title_fontsize=14)

plt.grid(True, alpha=0.3)

# Set axis limits
plt.xlim(steps.min(), steps.max())
plt.ylim(0, 1)

# Improve layout
plt.tight_layout()

# Save the plot
plt.savefig('final_stackplot_tab20_others_top.png', dpi=300, bbox_inches='tight')
print(f"\nFinal stackplot saved as 'final_stackplot_tab20_others_top.png'")
print("- Uses tab20 color scheme")
print("- 'Others' category is at the top")
print("- Legend shows order from bottom to top")

plt.show() 