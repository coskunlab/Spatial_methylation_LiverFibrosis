#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm

def process_file(excel_path):
    """
    Processes a single file to get colocalization data.
    """
    try:
        channels_df = pd.read_excel(excel_path)
        channels_df.dropna(subset=['StitchPath'], inplace=True)
        basePath = Path(channels_df['StitchPath'].iloc[-1])
        pkl_path = basePath / '05 PKL single cell' / f"{excel_path.stem}_pixel_dataframe.pkl"
        
        if not pkl_path.exists():
            return None
            
        df = pd.read_pickle(pkl_path)
        # Downsample for performance
        return df.sample(n=10000, random_state=42) if len(df) > 10000 else df
    except Exception as e:
        print(f"Error processing {excel_path.name}: {e}")
        return None

def main():
    """
    Main processing loop.
    """
    master_excel_path = Path("../Data/5Nov2025_all_slides.xlsx")
    output_dir = Path('../figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        master_df = pd.read_excel(master_excel_path)
    except Exception as e:
        sys.exit(f"Error reading master Excel file: {e}")

    all_data = []
    for _, row in tqdm(master_df.iterrows(), total=master_df.shape[0], desc="Loading Data"):
        df_sample = process_file(Path(row['ExcelPath']))
        if df_sample is not None:
            df_sample['Diet'] = row['Diet']
            df_sample['Fibrosis'] = row['Fibrosis']
            df_sample['group'] = f"{row['Diet']}-{row['Fibrosis']}"
            all_data.append(df_sample)

    if not all_data:
        sys.exit("No data loaded. Exiting.")

    full_df = pd.concat(all_data, ignore_index=True)
    marker_cols = [col for col in full_df.columns if col not in ['X', 'Y', 'cell_id', 'mask_type', 'Diet', 'Fibrosis', 'group']]

    print("\nGenerating comparative colocalization plots...")

    # Generate correlation heatmaps for each group (replaces tiny scatter plot grids)
    try:
        groups = full_df['group'].unique()
        clean_marker_names = [col.replace('Intensity_', '') for col in marker_cols]

        for group in groups:
            group_df = full_df[full_df['group'] == group]

            # Calculate correlation matrix
            corr_matrix = group_df[marker_cols].corr()

            # Create heatmap
            plt.figure(figsize=(6, 5))
            mask = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)

            sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0,
                       vmin=-1, vmax=1, mask=mask,
                       xticklabels=clean_marker_names, yticklabels=clean_marker_names,
                       cbar_kws={'label': 'Correlation'})

            plt.title(f'Marker Colocalization Heatmap - {group}', fontweight="bold", fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            safe_group = group.replace(' ', '_').replace('-', '_')
            plt.savefig(output_dir / f'27_colocalization_heatmap_{safe_group}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: 27_colocalization_heatmap_{safe_group}.png")

    except Exception as e:
        print(f"Could not generate correlation heatmaps: {e}")

    # Calculate gradient standard deviation heatmap (like Jiaxun's analysis)
    print("\nCalculating colocalization gradient standard deviations across groups...")
    try:
        # Get unique groups
        groups = full_df['group'].unique()
        n_markers = len(marker_cols)

        # Store gradients for each group
        gradient_matrices = {}

        for group in groups:
            group_df = full_df[full_df['group'] == group]
            # Normalize the data
            df_normalized = (group_df[marker_cols] - group_df[marker_cols].min()) / (group_df[marker_cols].max() - group_df[marker_cols].min())

            # Calculate gradient matrix for this group
            m = np.zeros((n_markers, n_markers))
            for i in range(n_markers):
                for j in range(n_markers):
                    if i != j:
                        x = df_normalized[marker_cols[i]].values
                        y = df_normalized[marker_cols[j]].values
                        # Calculate linear fit gradient
                        m[i, j], _ = np.polyfit(x, y, 1)

            gradient_matrices[group] = m

        # Calculate standard deviation across groups
        std_matrix = np.zeros((n_markers, n_markers))
        for i in range(n_markers):
            for j in range(n_markers):
                gradients = [abs(gradient_matrices[group][i, j]) for group in groups]
                std_matrix[i, j] = np.std(gradients)

        # Create heatmap
        plt.figure(figsize=(6, 5))
        mask = np.tril(np.ones_like(std_matrix, dtype=bool))

        # Clean marker names for display
        clean_marker_names = [col.replace('Intensity_', '') for col in marker_cols]

        g = sns.heatmap(std_matrix, annot=False, cmap="Blues", mask=mask,
                       xticklabels=clean_marker_names, yticklabels=clean_marker_names)
        g.tick_params(axis='x', reset=True, bottom=False, labeltop=True, labelbottom=False)
        g.set_xticklabels(clean_marker_names, fontsize=10, rotation=30, ha='left')
        g.set_yticklabels(clean_marker_names, fontsize=10, rotation=0)

        plt.title('Colocalization Gradient Standard Deviation Across Groups', pad=20, fontweight="bold", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / '27_colocalization_gradient_std.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: 27_colocalization_gradient_std.png")

    except Exception as e:
        print(f"Could not generate gradient std heatmap: {e}")

    # Generate individual scatterplots for all pairwise marker combinations
    print("\nGenerating pairwise marker scatterplots for each group...")
    try:
        groups = full_df['group'].unique()
        clean_marker_names = [col.replace('Intensity_', '') for col in marker_cols]

        # For each group
        for group in groups:
            group_df = full_df[full_df['group'] == group]

            # Normalize the data
            df_normalized = (group_df[marker_cols] - group_df[marker_cols].min()) / (group_df[marker_cols].max() - group_df[marker_cols].min())

            safe_group = group.replace(' ', '_').replace('-', '_')

            # Create scatterplot for all pairwise combinations
            for i, marker_col_i in enumerate(marker_cols):
                for j, marker_col_j in enumerate(marker_cols):
                    marker_name_i = clean_marker_names[i]
                    marker_name_j = clean_marker_names[j]

                    # Get data
                    x = df_normalized[marker_col_i].values
                    y = df_normalized[marker_col_j].values

                    # Calculate linear fit
                    m, c = np.polyfit(x, y, 1)
                    y_predicted = m * x + c

                    # Create figure
                    plt.figure(figsize=(4, 3))
                    plt.scatter(x, y, s=1, alpha=0.5, c='purple')

                    # Add trendline for all plots
                    plt.plot(x, y_predicted, color='red', linewidth=2)

                    plt.xlabel(marker_name_i, fontsize=14)
                    plt.ylabel(marker_name_j, fontsize=14)
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.title(f'{marker_name_j} vs. {marker_name_i} - {group}', fontsize=16, fontweight='bold')
                    plt.tight_layout()

                    # Save figure
                    safe_marker_i = marker_name_i.replace(' ', '_').replace('-', '').replace('/', '')
                    safe_marker_j = marker_name_j.replace(' ', '_').replace('-', '').replace('/', '')
                    plt.savefig(output_dir / f'27_scatter_{safe_marker_j}_vs_{safe_marker_i}_{safe_group}.png', dpi=300, bbox_inches='tight')
                    plt.close()

            print(f"Saved: All pairwise scatterplots for {group}")

    except Exception as e:
        print(f"Could not generate pairwise scatterplots: {e}")

if __name__ == '__main__':
    main()
