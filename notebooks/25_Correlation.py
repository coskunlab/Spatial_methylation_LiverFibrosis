#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm

def process_file(excel_path, diet, fibrosis, output_dir, return_data=False):
    """
    Processes a single file for correlation analysis.
    """
    print(f"\n--- Processing {excel_path.name} (Diet: {diet}, Fibrosis: {fibrosis}) ---")

    try:
        channels_df = pd.read_excel(excel_path)
        channels_df.dropna(subset=['StitchPath'], inplace=True)
        basePath = Path(channels_df['StitchPath'].iloc[-1])
        pkl_path = basePath / '05 PKL single cell' / f"{excel_path.stem}_pixel_dataframe.pkl"

        if not pkl_path.exists():
            print(f"Warning: PKL file not found for {excel_path.name} at {pkl_path}")
            return None

        df = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"Error processing {excel_path.name}: {e}")
        return None

    # Aggregate to cell-level data
    df_numeric = df.drop(columns=['X', 'Y', 'mask_type'])
    cell_df = df_numeric.groupby('cell_id').mean()
    marker_cols = [col for col in cell_df.columns if col not in ['X', 'Y']]

    # Calculate correlation matrix
    corr_matrix = cell_df[marker_cols].corr()

    # Plotting
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
        plt.title(f'Marker Correlation - {excel_path.stem}\n(Diet: {diet}, Fibrosis: {fibrosis})')
        plt.savefig(output_dir / f'25_{excel_path.stem}_{diet}_{fibrosis}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: 25_{excel_path.stem}_{diet}_{fibrosis}_correlation_heatmap.png")
    except Exception as e:
        print(f"Could not generate correlation heatmap for {excel_path.name}: {e}")

    if return_data:
        return cell_df[marker_cols]
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

    print("\n" + "="*80)
    print("Processing individual files...")
    print("="*80)

    for _, row in tqdm(master_df.iterrows(), total=master_df.shape[0], desc="Processing Experiments"):
        process_file(Path(row['ExcelPath']), row['Diet'], row['Fibrosis'], output_dir)

    # --- Aggregate 2x2 Subplot ---
    print("\n" + "="*80)
    print("Creating aggregate correlation plots...")
    print("="*80)

    # Collect data for each group (new data only, no Jiaxun)
    group_data = {
        'Regular-Control': [],
        'High Fat-Control': [],
        'Regular-Fibrotic': [],
        'High Fat-Fibrotic': []
    }

    # Re-process to collect data (this time with return_data=True)
    for _, row in tqdm(master_df.iterrows(), total=master_df.shape[0], desc="Collecting Data"):
        excel_path = Path(row['ExcelPath'])
        diet = row['Diet']
        fibrosis = row['Fibrosis']
        group_key = f"{diet}-{fibrosis}"

        data = process_file(excel_path, diet, fibrosis, output_dir, return_data=True)
        if data is not None:
            group_data[group_key].append(data)

    # Create 2x2 subplot
    sample_names = ['Regular-Control', 'High Fat-Control', 'Regular-Fibrotic', 'High Fat-Fibrotic']
    fig = plt.figure(figsize=(15, 10))

    for i, group_name in enumerate(sample_names, 1):
        if group_data[group_name]:
            # Concatenate all data for this group
            combined_data = pd.concat(group_data[group_name], ignore_index=True)
            corr_matrix = combined_data.corr()

            plt.subplot(2, 2, i)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar_kws={'shrink': 0.8})
            plt.title(group_name, pad=20, fontweight="bold", fontsize=18)
            plt.xticks(rotation=30, fontsize=12)
            plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / '25_correlation_all_samples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 25_correlation_all_samples.png")

if __name__ == '__main__':
    main()
