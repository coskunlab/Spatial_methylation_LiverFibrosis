#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tqdm import tqdm
import tifffile as tf
from cellpose import models, core
import torch
from pathlib import Path
from statannotations.Annotator import Annotator

# Set font scale to original size
sns.set(font_scale=1)

# Check GPU access
assert torch.cuda.is_available()

# Cellpose use GPU
use_GPU = core.use_gpu()
print('>>> GPU activated? %d' % use_GPU)

# Call logger_setup to have output of cellpose written
from cellpose.io import logger_setup
logger_setup()

def get_sample_classification(sample_name):
    """
    Classify sample by diet and fibrosis status based on sample name.

    Pattern:
    - 1L-6L: Regular diet, Control (no fibrosis)
    - 7L-10L: High Fat diet, Control (no fibrosis)
    - 11L-16L: Regular diet, Fibrosis
    - 17L-20L: High Fat diet, Fibrosis
    """
    import re
    match = re.search(r'(\d+)L', str(sample_name))
    if not match:
        return {'diet': 'Unknown', 'fibrosis': 'Unknown'}

    sample_num = int(match.group(1))

    if 1 <= sample_num <= 6:
        return {'diet': 'Regular', 'fibrosis': 'Control'}
    elif 7 <= sample_num <= 10:
        return {'diet': 'High Fat', 'fibrosis': 'Control'}
    elif 11 <= sample_num <= 16:
        return {'diet': 'Regular', 'fibrosis': 'Fibrosis'}
    elif 17 <= sample_num <= 20:
        return {'diet': 'High Fat', 'fibrosis': 'Fibrosis'}
    else:
        return {'diet': 'Unknown', 'fibrosis': 'Unknown'}

def segment_nuclei(img, model, diameter, flow_threshold=0.8, cellprob_threshold=0):
    """
    Segment nuclei in an image using Cellpose.
    Returns the number of cells detected.
    """
    segChannels = [0, 0]  # cyto, nuc of RGB
    result = model.eval(
        img,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=segChannels
    )

    # Handle different return formats between Cellpose versions
    if len(result) == 4:
        masks, flows, styles, diams = result
    elif len(result) == 3:
        masks, flows, styles = result
    else:
        masks = result[0]

    # Count number of unique cells (excluding background label 0)
    num_cells = len(np.unique(masks)) - 1
    return num_cells, masks

def main():
    """
    Main processing loop to quantify cell loss across cycles.
    """
    # Read master Excel file
    master_excel_path = Path("../Data/5Nov2025_all_slides.xlsx")
    try:
        master_df = pd.read_excel(master_excel_path)
    except Exception as e:
        sys.exit(f"Error reading master Excel file: {e}")

    # Output directory
    output_dir = Path('../figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Cellpose parameters
    imgRes = 0.177  # resolution of images, um/pixel
    model_choice = 'nuclei'
    model = models.CellposeModel(gpu=use_GPU, model_type=model_choice)
    diameter = 10 / imgRes  # 10 um diameter
    flow_threshold = 0.8
    cellprob_threshold = 0
    nucChan = 'Hoechst'

    # Store results
    all_results = []

    print("\n" + "="*80)
    print("Processing samples to quantify cell loss across cycles...")
    print("="*80)

    for _, row in tqdm(master_df.iterrows(), total=master_df.shape[0], desc="Processing Samples"):
        excel_path = Path(row['ExcelPath'])
        print(f"\n[DEBUG] Processing sample: {excel_path.stem}")
        sys.stdout.flush()

        # Get classification
        classification = get_sample_classification(excel_path.stem)
        diet = classification['diet']
        fibrosis = classification['fibrosis']

        if not excel_path.is_file():
            print(f"Warning: Skipping invalid file path: {excel_path}")
            continue

        try:
            print(f"[DEBUG] Reading Excel file...")
            channels_df = pd.read_excel(excel_path)
            channels_df.dropna(subset=['StitchPath'], inplace=True)

            if channels_df.empty:
                print(f"[DEBUG] Empty channels_df, skipping")
                continue

            # Get base path from latest cycle
            basePath = Path(channels_df['StitchPath'].iloc[-1])
            regSavePath = basePath / '02 TIF stitched registered'
            print(f"[DEBUG] regSavePath: {regSavePath}")

            if not regSavePath.exists():
                print(f"Warning: Registered images folder not found for {excel_path.name}")
                continue

            # Get all unique cycles
            cycles = sorted(channels_df['Cycle'].unique())
            print(f"[DEBUG] Found {len(cycles)} cycles: {cycles}")

            # Track cell counts per cycle
            cycle_cell_counts = {}

            for cycle in cycles:
                cycle_str = f'Cycle{str(cycle).zfill(2)}'
                print(f"[DEBUG] Processing {cycle_str}...")

                # Find nucleus image for this cycle
                fileName = list(regSavePath.glob(f'*{cycle_str}*{nucChan}*.tif'))

                if len(fileName) != 1:
                    print(f"Warning: Could not find unique nucleus image for {cycle_str} in {excel_path.name}")
                    continue

                print(f"[DEBUG] Reading TIFF: {fileName[0].name}")
                # Read image
                imgNuc = tf.imread(fileName[0])
                print(f"[DEBUG] Image shape: {imgNuc.shape}, dtype: {imgNuc.dtype}")

                print(f"[DEBUG] Starting segmentation...")
                # Segment and count cells
                num_cells, masks = segment_nuclei(imgNuc, model, diameter, flow_threshold, cellprob_threshold)
                print(f"[DEBUG] Segmentation complete. Found {num_cells} cells")
                cycle_cell_counts[cycle] = num_cells

            # Calculate percentage of cells remaining relative to first cycle
            if cycle_cell_counts and min(cycle_cell_counts.keys()) in cycle_cell_counts:
                first_cycle = min(cycle_cell_counts.keys())
                first_cycle_count = cycle_cell_counts[first_cycle]

                if first_cycle_count > 0:
                    for cycle, count in cycle_cell_counts.items():
                        pct_remaining = (count / first_cycle_count) * 100
                        pct_loss = 100 - pct_remaining

                        all_results.append({
                            'sample': excel_path.stem,
                            'diet': diet,
                            'fibrosis': fibrosis,
                            'group': f"{diet} - {fibrosis}",
                            'cycle': cycle,
                            'cell_count': count,
                            'pct_remaining': pct_remaining,
                            'pct_loss': pct_loss
                        })

        except Exception as e:
            print(f"Error processing {excel_path.name}: {e}")
            continue

    if not all_results:
        sys.exit("No data was processed. Exiting.")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    print("\n" + "="*80)
    print("Creating plots...")
    print("="*80)

    # --- Plot 1: Cell loss across cycles (line plot) ---
    fig, ax = plt.subplots(figsize=(6, 3))

    # Create color palette for diet/fibrosis combinations
    palette = {
        'Regular - Control': 'blue',
        'High Fat - Control': 'orange',
        'Regular - Fibrosis': 'green',
        'High Fat - Fibrosis': 'red'
    }

    # Plot each sample as a line
    for sample in results_df['sample'].unique():
        sample_data = results_df[results_df['sample'] == sample]
        group = sample_data['group'].iloc[0]
        color = palette.get(group, 'gray')
        ax.plot(sample_data['cycle'], sample_data['pct_loss'],
                marker='o', alpha=0.3, color=color, linewidth=1)

    # Plot mean with error bars for each group
    for group, color in palette.items():
        group_data = results_df[results_df['group'] == group]
        if not group_data.empty:
            mean_data = group_data.groupby('cycle')['pct_loss'].agg(['mean', 'sem']).reset_index()
            ax.errorbar(mean_data['cycle'], mean_data['mean'], yerr=mean_data['sem'],
                       marker='o', linewidth=3, markersize=8, label=group, color=color)

    ax.set_xlabel('Cycle', fontsize=14)
    ax.set_ylabel('Cell Loss (%)', fontsize=14)
    ax.set_title('Cell Loss Across Cycles by Diet and Fibrosis Status', fontsize=16)
    ax.legend(fontsize=10, bbox_to_anchor=(1, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '36_cell_loss_across_cycles_line.png', dpi=300)
    plt.close()
    print("Saved: 36_cell_loss_across_cycles_line.png")

    # --- Plot 2: Cell loss at final cycle (boxplot) ---
    # Get final cycle for each sample
    final_cycle_data = results_df.loc[results_df.groupby('sample')['cycle'].idxmax()]

    if not final_cycle_data.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(data=final_cycle_data, x='group', y='pct_loss', hue='diet', ax=ax, legend=False)
        sns.stripplot(data=final_cycle_data, x='group', y='pct_loss', color='.25', ax=ax)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

        # Add statistical annotations
        pairs = []
        unique_fibrosis = final_cycle_data['fibrosis'].unique()
        for fib in unique_fibrosis:
            group1 = f"Regular - {fib}"
            group2 = f"High Fat - {fib}"
            if group1 in final_cycle_data['group'].values and group2 in final_cycle_data['group'].values:
                pairs.append((group1, group2))

        if pairs:
            annotator = Annotator(ax, pairs, data=final_cycle_data, x='group', y='pct_loss')
            annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
            annotator.apply_and_annotate()

        plt.title('Cell Loss at Final Cycle', fontsize=16)
        plt.ylabel('Cell Loss (%)', fontsize=12)
        plt.xlabel('Group', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / '36_cell_loss_final_cycle_boxplot.png', dpi=300)
        plt.close()
        print("Saved: 36_cell_loss_final_cycle_boxplot.png")

    # --- Plot 3: Cell count across cycles (line plot) ---
    fig, ax = plt.subplots(figsize=(6, 3))

    # Plot each sample as a line
    for sample in results_df['sample'].unique():
        sample_data = results_df[results_df['sample'] == sample]
        group = sample_data['group'].iloc[0]
        color = palette.get(group, 'gray')
        ax.plot(sample_data['cycle'], sample_data['cell_count'],
                marker='o', alpha=0.3, color=color, linewidth=1)

    # Plot mean with error bars for each group
    for group, color in palette.items():
        group_data = results_df[results_df['group'] == group]
        if not group_data.empty:
            mean_data = group_data.groupby('cycle')['cell_count'].agg(['mean', 'sem']).reset_index()
            ax.errorbar(mean_data['cycle'], mean_data['mean'], yerr=mean_data['sem'],
                       marker='o', linewidth=3, markersize=8, label=group, color=color)

    ax.set_xlabel('Cycle', fontsize=14)
    ax.set_ylabel('Cell Count', fontsize=14)
    ax.set_title('Cell Count Across Cycles by Diet and Fibrosis Status', fontsize=16)
    ax.legend(fontsize=10, bbox_to_anchor=(1, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '36_cell_count_across_cycles_line.png', dpi=300)
    plt.close()
    print("Saved: 36_cell_count_across_cycles_line.png")

    # --- Plot 4: Cell loss per cycle (boxplot for each cycle) ---
    cycles_list = sorted(results_df['cycle'].unique())
    if len(cycles_list) > 1:
        fig, axes = plt.subplots(1, len(cycles_list), figsize=(2.5*len(cycles_list), 3), sharey=True)
        if len(cycles_list) == 1:
            axes = [axes]

        for idx, cycle in enumerate(cycles_list):
            cycle_data = results_df[results_df['cycle'] == cycle]
            sns.boxplot(data=cycle_data, x='group', y='pct_loss', hue='diet', ax=axes[idx], legend=False)
            sns.stripplot(data=cycle_data, x='group', y='pct_loss', color='.25', ax=axes[idx])
            axes[idx].set_title(f'Cycle {cycle}', fontsize=14)
            axes[idx].set_xlabel('')
            axes[idx].set_ylabel('Cell Loss (%)' if idx == 0 else '', fontsize=12)
            axes[idx].tick_params(axis='x', rotation=45)
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
            if idx == len(cycles_list) - 1:
                axes[idx].legend(bbox_to_anchor=(1, 1), loc='upper left')

        plt.suptitle('Cell Loss by Cycle', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / '36_cell_loss_per_cycle_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: 36_cell_loss_per_cycle_boxplots.png")

    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)

if __name__ == '__main__':
    main()
