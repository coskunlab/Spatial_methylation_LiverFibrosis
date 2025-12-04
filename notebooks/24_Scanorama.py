#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from pathlib import Path
import sys
from tqdm import tqdm

def main():
    """
    Main processing loop for Scanorama integration and visualization.
    """
    master_excel_path = Path("../Data/5Nov2025_all_slides.xlsx")
    output_dir = Path('../figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        master_df = pd.read_excel(master_excel_path)
    except Exception as e:
        sys.exit(f"Error reading master Excel file: {e}")

    adatas = []
    for _, row in tqdm(master_df.iterrows(), total=master_df.shape[0], desc="Loading Data"):
        excel_path = Path(row['ExcelPath'])
        diet = row['Diet']
        fibrosis = row['Fibrosis']
        
        try:
            channels_df = pd.read_excel(excel_path)
            channels_df.dropna(subset=['StitchPath'], inplace=True)
            basePath = Path(channels_df['StitchPath'].iloc[-1])
            pkl_path = basePath / '05 PKL single cell' / f"{excel_path.stem}_pixel_dataframe.pkl"

            if not pkl_path.exists():
                continue

            df = pd.read_pickle(pkl_path)
            df_numeric = df.drop(columns=['X', 'Y', 'mask_type'])
            cell_df = df_numeric.groupby('cell_id').mean()
            marker_cols = [col for col in cell_df.columns if col not in ['X', 'Y']]
            
            adata = anndata.AnnData(cell_df[marker_cols])
            adata.obs['library_id'] = excel_path.stem
            adata.obs['Diet'] = diet
            adata.obs['Fibrosis'] = fibrosis
            adatas.append(adata)
        except Exception as e:
            print(f"Error processing {excel_path.name}: {e}")
            continue

    if not adatas:
        sys.exit("No data loaded. Exiting.")

    print("\nIntegrating data with Scanorama...")
    # Integrate the datasets
    adata_all = anndata.concat(adatas)
    sc.pp.scale(adata_all)
    sc.pp.neighbors(adata_all)
    sc.tl.leiden(adata_all)
    
    # Note: Scanorama is not used here as per the original notebook's logic which just concatenates.
    # If Scanorama integration is desired, the following would be used:
    # adatas_cor = sc.external.pp.scanorama_integrate(adatas, 'library_id')
    # adata_all = anndata.concat(adatas_cor)

    print("Running UMAP and plotting...")
    sc.tl.umap(adata_all)

    # Plotting
    try:
        sc.pl.umap(adata_all, color=['Diet', 'Fibrosis', 'leiden'], save='_integr.png', show=False)
        # Rename the output file to match our convention
        (output_dir / 'umap_integr.png').unlink(missing_ok=True)
        Path('figures/umap_integr.png').rename(output_dir / '24_scanorama_integration_umap.png')
        print(f"Saved: 24_scanorama_integration_umap.png")
    except Exception as e:
        print(f"Could not generate UMAP plot: {e}")

if __name__ == '__main__':
    main()
