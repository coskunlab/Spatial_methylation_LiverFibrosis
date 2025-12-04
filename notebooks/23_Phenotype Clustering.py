#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import sys
from pathlib import Path
from tqdm import tqdm

# Set font scale to double all font sizes
sns.set(font_scale=2)

def process_file(excel_path, diet, fibrosis, output_dir):
    """
    Processes a single file for phenotype clustering.
    """
    print(f"\n--- Processing {excel_path.name} (Diet: {diet}, Fibrosis: {fibrosis}) ---")
    
    try:
        channels_df = pd.read_excel(excel_path)
        channels_df.dropna(subset=['StitchPath'], inplace=True)
        basePath = Path(channels_df['StitchPath'].iloc[-1])
        pkl_path = basePath / '05 PKL single cell' / f"{excel_path.stem}_pixel_dataframe.pkl"
        
        if not pkl_path.exists():
            print(f"Warning: PKL file not found for {excel_path.name} at {pkl_path}")
            return
            
        df = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"Error processing {excel_path.name}: {e}")
        return

    # Aggregate to cell-level data
    df_numeric = df.drop(columns=['X', 'Y', 'mask_type'])
    cell_df = df_numeric.groupby('cell_id').mean()
    marker_cols = [col for col in cell_df.columns if col not in ['X', 'Y']]
    X = cell_df[marker_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    # --- Clustering & Visualization ---
    # Heatmap
    try:
        g = sns.clustermap(X_scaled, cmap='coolwarm', standard_scale=1, figsize=(20, 15))
        g.fig.suptitle(f'Hierarchical Clustering - {excel_path.stem}\n(Diet: {diet}, Fibrosis: {fibrosis})', y=0.98)
        plt.savefig(output_dir / f'23_{excel_path.stem}_{diet}_{fibrosis}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: 23_{excel_path.stem}_{diet}_{fibrosis}_heatmap.png")
    except Exception as e:
        print(f"Could not generate heatmap for {excel_path.name}: {e}")

    # Clustering and t-SNE
    try:
        # Perform KMeans clustering
        n_clusters = min(10, len(X_scaled) // 100)  # Adaptive number of clusters
        n_clusters = max(5, n_clusters)  # At least 5 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
        X_tsne = tsne.fit_transform(X_scaled)

        # Plot t-SNE colored by clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, s=5, cmap='tab10')
        plt.colorbar(scatter, label='Cluster')
        plt.title(f't-SNE Plot - {excel_path.stem}\n(Diet: {diet}, Fibrosis: {fibrosis})')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig(output_dir / f'23_{excel_path.stem}_{diet}_{fibrosis}_tsne.png', dpi=300)
        plt.close()
        print(f"Saved: 23_{excel_path.stem}_{diet}_{fibrosis}_tsne.png")

        # Create cluster-marker heatmap
        cell_df_with_clusters = cell_df[marker_cols].copy()
        cell_df_with_clusters['cluster'] = cluster_labels
        cluster_means = cell_df_with_clusters.groupby('cluster').mean()

        # Remove 'Intensity_' prefix for cleaner labels
        cluster_means.columns = [col.replace('Intensity_', '') for col in cluster_means.columns]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cluster_means.T, cmap='coolwarm', center=0, cbar_kws={'label': 'Mean Intensity'})
        plt.title(f'Cluster-Marker Heatmap - {excel_path.stem}\n(Diet: {diet}, Fibrosis: {fibrosis})')
        plt.xlabel('Cluster')
        plt.ylabel('Marker')
        plt.tight_layout()
        plt.savefig(output_dir / f'23_{excel_path.stem}_{diet}_{fibrosis}_cluster_marker_heatmap.png', dpi=300)
        plt.close()
        print(f"Saved: 23_{excel_path.stem}_{diet}_{fibrosis}_cluster_marker_heatmap.png")
    except Exception as e:
        print(f"Could not generate t-SNE/clustering for {excel_path.name}: {e}")

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

    # --- Aggregate Clustering Analysis (with subsampling) ---
    print("\n" + "="*80)
    print("Creating aggregate phenotype clustering plots...")
    print("="*80)

    # Collect all pixel data from new data only (no Jiaxun)
    all_pixel_data = []

    print("Collecting data from master Excel files...")
    for _, row in tqdm(master_df.iterrows(), total=master_df.shape[0], desc="Collecting Data"):
        excel_path = Path(row['ExcelPath'])
        diet = row['Diet']
        fibrosis = row['Fibrosis']
        try:
            channels_df = pd.read_excel(excel_path)
            channels_df.dropna(subset=['StitchPath'], inplace=True)
            basePath = Path(channels_df['StitchPath'].iloc[-1])
            pkl_path = basePath / '05 PKL single cell' / f"{excel_path.stem}_pixel_dataframe.pkl"

            if pkl_path.exists():
                df = pd.read_pickle(pkl_path)
                df_numeric = df.drop(columns=['X', 'Y', 'mask_type'])
                cell_df = df_numeric.groupby('cell_id').mean()
                marker_cols = [col for col in cell_df.columns if col not in ['X', 'Y']]
                # Add group information
                cell_df['Diet'] = diet
                cell_df['Fibrosis'] = fibrosis
                cell_df['group'] = f"{diet}-{fibrosis}"
                all_pixel_data.append(cell_df)
        except Exception:
            pass

    if all_pixel_data:
        # Combine all data
        combined_df = pd.concat(all_pixel_data, ignore_index=True)
        print(f"Combined dataset size: {len(combined_df)} pixels")

        # Store marker columns (before adding group info)
        marker_cols = [col for col in combined_df.columns if col not in ['Diet', 'Fibrosis', 'group']]

        # Subsample if too large (to make PARC tractable)
        MAX_SAMPLES = 50000
        if len(combined_df) > MAX_SAMPLES:
            print(f"Subsampling from {len(combined_df)} to {MAX_SAMPLES} pixels for clustering...")
            combined_df = combined_df.sample(n=MAX_SAMPLES, random_state=42)

        # Standardize data
        X = combined_df[marker_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(np.log(X + 0.1))

        # Run PARC clustering
        print("Running PARC clustering on combined data...")
        try:
            import parc
            parc_model = parc.PARC(X_scaled, jac_weighted_edges=False, random_seed=0)
            parc_model.run_PARC()
            parc_labels = parc_model.labels

            # Create aggregate heatmap
            combined_df['label'] = parc_labels
            df_per_label = combined_df.groupby('label')[marker_cols].mean()

            # Remove 'Intensity_' prefix from column names for cleaner labels
            df_per_label.columns = [col.replace('Intensity_', '') for col in df_per_label.columns]

            g = sns.clustermap(df_per_label, cmap='coolwarm', standard_scale=1, figsize=(15, 10))
            plt.savefig(output_dir / '23_phenotype_clustering_heatmap_aggregate.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved: 23_phenotype_clustering_heatmap_aggregate.png")

            # Run t-SNE
            print("Running t-SNE on combined data...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
            X_tsne = tsne.fit_transform(X_scaled)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=parc_labels, s=2, cmap='tab20')
            plt.colorbar(scatter, label='Cluster')
            plt.title('Aggregate t-SNE Plot\n(New Data Only)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.savefig(output_dir / '23_phenotype_clustering_tsne_aggregate.png', dpi=300)
            plt.close()
            print("Saved: 23_phenotype_clustering_tsne_aggregate.png")

            # Create t-SNE plot colored by diet & fibrosis category
            plt.figure(figsize=(10, 8))

            # Create color mapping for groups
            group_palette = {
                'Regular-Control': 'blue',
                'High Fat-Control': 'orange',
                'Regular-Fibrotic': 'green',
                'High Fat-Fibrotic': 'red'
            }

            # Map groups to colors
            colors = combined_df['group'].map(group_palette)

            # Plot each group separately for legend
            for group, color in group_palette.items():
                mask = combined_df['group'] == group
                if mask.any():
                    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                              c=color, s=2, label=group, alpha=0.6)

            plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
            plt.title('Aggregate t-SNE Plot by Diet & Fibrosis\n(New Data Only)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.tight_layout()
            plt.savefig(output_dir / '23_phenotype_clustering_tsne_aggregate_by_group.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved: 23_phenotype_clustering_tsne_aggregate_by_group.png")

        except ImportError:
            print("Warning: PARC not installed. Skipping PARC clustering. Install with: pip install parc")
        except Exception as e:
            print(f"Could not complete aggregate clustering: {e}")

        # --- Per-Condition Clustering Analysis ---
        print("\n" + "="*80)
        print("Creating per-condition phenotype clustering plots...")
        print("="*80)

        unique_groups = combined_df['group'].unique()
        for group in unique_groups:
            print(f"\nProcessing group: {group}")
            group_df = combined_df[combined_df['group'] == group]

            # Get marker data for this group
            X_group = group_df[marker_cols].values

            # Skip if too few samples
            if len(X_group) < 100:
                print(f"Skipping {group}: too few samples ({len(X_group)})")
                continue

            # Subsample if needed
            MAX_GROUP_SAMPLES = 10000
            if len(X_group) > MAX_GROUP_SAMPLES:
                print(f"Subsampling {group} from {len(X_group)} to {MAX_GROUP_SAMPLES} pixels...")
                sample_idx = np.random.choice(len(X_group), MAX_GROUP_SAMPLES, replace=False)
                X_group = X_group[sample_idx]

            # Standardize
            X_group_scaled = StandardScaler().fit_transform(np.log(X_group + 0.1))

            # Run PARC clustering
            try:
                import parc
                print(f"Running PARC clustering for {group}...")
                parc_model = parc.PARC(X_group_scaled, jac_weighted_edges=False, random_seed=0)
                parc_model.run_PARC()
                group_labels = parc_model.labels

                # Create heatmap
                group_df_subset = pd.DataFrame(X_group, columns=marker_cols)
                group_df_subset['label'] = group_labels
                df_per_label = group_df_subset.groupby('label').mean()

                # Remove 'Intensity_' prefix
                df_per_label.columns = [col.replace('Intensity_', '') for col in df_per_label.columns]

                g = sns.clustermap(df_per_label, cmap='coolwarm', standard_scale=1, figsize=(15, 10))
                safe_group = group.replace(' ', '_').replace('-', '_')
                plt.savefig(output_dir / f'23_phenotype_clustering_heatmap_{safe_group}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved: 23_phenotype_clustering_heatmap_{safe_group}.png")

                # Run t-SNE
                print(f"Running t-SNE for {group}...")
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_group_scaled)-1))
                X_tsne = tsne.fit_transform(X_group_scaled)

                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=group_labels, s=2, cmap='tab20')
                plt.colorbar(scatter, label='Cluster')
                plt.title(f't-SNE Plot - {group}')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.savefig(output_dir / f'23_phenotype_clustering_tsne_{safe_group}.png', dpi=300)
                plt.close()
                print(f"Saved: 23_phenotype_clustering_tsne_{safe_group}.png")

            except ImportError:
                print(f"Warning: PARC not installed. Skipping PARC clustering for {group}.")
            except Exception as e:
                print(f"Could not complete clustering for {group}: {e}")

    else:
        print("No data collected for aggregate analysis.")

if __name__ == '__main__':
    main()
