#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.spatial import KDTree
import sys
from pathlib import Path
from tqdm import tqdm
import tifffile as tf
from skimage.measure import regionprops_table
from statannotations.Annotator import Annotator
from itertools import combinations

# Set font scale to double all font sizes
sns.set(font_scale=2)


def generate_single_cell_dataframe(channels_df, reg_path, mask_image):
    """
    Generates a single-cell dataframe by measuring marker intensities within masked regions.
    """
    if mask_image.max() == 0:
        return None

    props = regionprops_table(mask_image, properties=('label', 'centroid', 'area'))
    sc_df = pd.DataFrame(props).set_index('label')
    sc_df.rename(columns={'centroid-0': 'Location_Center_Y', 'centroid-1': 'Location_Center_X'}, inplace=True)

    unique_channels = channels_df.dropna(subset=['Marker', 'Cycle', 'StitchPath'])

    for _, row in unique_channels.iterrows():
        marker_name = row['Marker']
        cycle_num = row['Cycle']
        cycle_str = f'Cycle{str(cycle_num).zfill(2)}'
        file_pattern = f"*{cycle_str}*{marker_name}*.tif"
        
        try:
            img_file = next(reg_path.glob(file_pattern))
            img = tf.imread(img_file)
            intensity_props = regionprops_table(mask_image, intensity_image=img, properties=('label', 'mean_intensity'))
            intensity_df = pd.DataFrame(intensity_props).set_index('label')
            temp_df = sc_df.join(intensity_df)
            sc_df[f'Intensity_IntegratedIntensity_{marker_name}'] = temp_df['mean_intensity'] * temp_df['area']
        except Exception as e:
            print(f"Could not process {marker_name} in {reg_path.parent.name}: {e}")
            
    return sc_df

def prepare_spatial_data(sc_df, intensity_quantile_threshold=0.75):
    """
    Converts the single-cell dataframe to a long-format spatial dataframe.
    """
    all_marker_data = []
    marker_columns = {col: col.replace('Intensity_IntegratedIntensity_', '') for col in sc_df.columns if col.startswith('Intensity_IntegratedIntensity_')}

    for col_name, marker_name in marker_columns.items():
        if sc_df[col_name].nunique() <= 1:
            threshold = sc_df[col_name].iloc[0]
        else:
            threshold = sc_df[col_name].quantile(intensity_quantile_threshold)
        positive_cells = sc_df[sc_df[col_name] >= threshold]

        if not positive_cells.empty:
            all_marker_data.append(pd.DataFrame({
                'X': positive_cells['Location_Center_X'],
                'Y': positive_cells['Location_Center_Y'],
                'marker': marker_name
            }))

    return pd.concat(all_marker_data, ignore_index=True).dropna() if all_marker_data else None

def calculate_nn_matrix(spatial_df, radius=60):
    """
    Calculates the mean number of neighbors within a given radius for each pair of marker types.
    """
    markers = spatial_df['marker'].unique()
    heatmap = pd.DataFrame(index=markers, columns=markers, dtype=float)

    for marker1 in markers:
        for marker2 in markers:
            coords1 = spatial_df[spatial_df['marker'] == marker1][['X', 'Y']]
            coords2 = spatial_df[spatial_df['marker'] == marker2][['X', 'Y']]

            if coords1.empty or coords2.empty:
                heatmap.loc[marker1, marker2] = 0
                continue

            kd_tree1 = KDTree(coords1)
            kd_tree2 = KDTree(coords2)
            indexes = kd_tree1.query_ball_tree(kd_tree2, r=radius)
            heatmap.loc[marker1, marker2] = np.mean([len(neighbors) for neighbors in indexes])
            
    return heatmap

def main():
    """
    Main processing loop.
    """
    master_excel_path = Path("../Data/5Nov2025_all_slides.xlsx")
    try:
        master_df = pd.read_excel(master_excel_path)
    except Exception as e:
        sys.exit(f"Error reading master Excel file: {e}")

    all_results = []
    
    for _, row in tqdm(master_df.iterrows(), total=master_df.shape[0], desc="Processing Experiments"):
        excel_path = Path(row['ExcelPath'])
        diet = row['Diet']
        fibrosis = row['Fibrosis']

        if not excel_path.is_file():
            print(f"Warning: Skipping invalid file path: {excel_path}")
            continue
        
        try:
            channels_df = pd.read_excel(excel_path)
        except Exception as e:
            print(f"Error reading channel excel file {excel_path.name}: {e}")
            continue

        channels_df.dropna(subset=['StitchPath'], inplace=True)
        if channels_df.empty:
            continue
            
        basePath = Path(channels_df['StitchPath'].iloc[-1])
        regSavePath = basePath / '02 TIF stitched registered'
        maskPath = basePath / '04 TIF masks'
        nucleus_mask_file = maskPath / f"{excel_path.stem}_nucleus_mask.tif"

        if not nucleus_mask_file.exists():
            continue
        
        nucleus_mask = tf.imread(nucleus_mask_file)
        sc_df = generate_single_cell_dataframe(channels_df, regSavePath, nucleus_mask)
        if sc_df is None: continue

        spatial_df = prepare_spatial_data(sc_df)
        if spatial_df is None: continue

        heatmap = calculate_nn_matrix(spatial_df)
        all_results.append({'heatmap': heatmap, 'diet': diet, 'fibrosis': fibrosis, 'name': excel_path.stem})

    if not all_results:
        sys.exit("No data was processed. Exiting.")

    print(f"\n{'='*80}\nAggregating results and plotting...\n{'='*80}")
    
    output_dir = Path('../figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- Comparative Plotting ---
    # Example: Compare 5-mC vs aSMA nearest neighbors across different diet/fibrosis groups
    plot_data = []
    for res in all_results:
        try:
            nn_value = res['heatmap'].loc['5-mC', 'aSMA']
            plot_data.append({
                'value': nn_value,
                'Diet': res['diet'],
                'Fibrosis': res['fibrosis'],
                'group': f"{res['diet']} - {res['fibrosis']}"
            })
        except KeyError:
            continue

    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(data=plot_df, x='group', y='value', hue='Diet', ax=ax)
        sns.stripplot(data=plot_df, x='group', y='value', color='.25', ax=ax)

        # Add statistical annotations
        pairs = []
        unique_fibrosis = plot_df['Fibrosis'].unique()
        for fib in unique_fibrosis:
            group1 = f"Regular - {fib}"
            group2 = f"High Fat - {fib}"
            if group1 in plot_df['group'].values and group2 in plot_df['group'].values:
                pairs.append((group1, group2))

        if pairs:
            annotator = Annotator(ax, pairs, data=plot_df, x='group', y='value')
            annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
            annotator.apply_and_annotate()

        plt.title('Nearest Neighbors\n(5-mC to aSMA)', pad=20)
        plt.ylabel('Mean # of Neighbors\n(r=60)')
        plt.xlabel('Group')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / '22_comparative_5mC_vs_aSMA.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: 22_comparative_5mC_vs_aSMA.png")

    # --- Comparative Boxplots for ALL Pairwise Marker Combinations ---
    # Collect all unique markers from the heatmaps
    all_markers = set()
    for res in all_results:
        if res['heatmap'] is not None:
            all_markers.update(res['heatmap'].index.tolist())

    all_markers = sorted(list(all_markers))
    print(f"\nFound {len(all_markers)} unique markers: {all_markers}")

    # Generate all pairwise combinations
    marker_pairs = list(combinations(all_markers, 2))
    print(f"Generating boxplots for {len(marker_pairs)} marker pairs...")

    for marker1, marker2 in tqdm(marker_pairs, desc="Creating boxplots"):
        plot_data = []

        for res in all_results:
            try:
                nn_value = res['heatmap'].loc[marker1, marker2]
                plot_data.append({
                    'value': nn_value,
                    'Diet': res['diet'],
                    'Fibrosis': res['fibrosis'],
                    'group': f"{res['diet']} - {res['fibrosis']}"
                })
            except (KeyError, AttributeError):
                pass

        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=plot_df, x='group', y='value', hue='Diet', ax=ax)
            sns.stripplot(data=plot_df, x='group', y='value', color='.25', ax=ax)

            # Add statistical annotations
            pairs = []
            unique_fibrosis = plot_df['Fibrosis'].unique()
            for fib in unique_fibrosis:
                group1 = f"Regular - {fib}"
                group2 = f"High Fat - {fib}"
                if group1 in plot_df['group'].values and group2 in plot_df['group'].values:
                    pairs.append((group1, group2))

            if pairs:
                try:
                    annotator = Annotator(ax, pairs, data=plot_df, x='group', y='value')
                    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
                    annotator.apply_and_annotate()
                except Exception:
                    pass  # Skip annotation if it fails

            # Clean marker names for filenames
            safe_marker1 = marker1.replace(' ', '_').replace('-', '').replace('/', '')
            safe_marker2 = marker2.replace(' ', '_').replace('-', '').replace('/', '')

            plt.title(f'Nearest Neighbors\n({marker1} to {marker2})', pad=20)
            plt.ylabel('Mean # of Neighbors\n(r=60)')
            plt.xlabel('Group')
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(output_dir / f'22_comparative_{safe_marker1}_vs_{safe_marker2}.png', dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    main()