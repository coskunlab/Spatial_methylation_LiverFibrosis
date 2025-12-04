#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import napari
from pathlib import Path
from scipy.spatial import KDTree
from magicgui import magicgui
import tifffile

def load_sample_data(excel_path):
    """
    Load cell data and background image for a sample.
    """
    try:
        channels_df = pd.read_excel(excel_path)
        channels_df.dropna(subset=['StitchPath'], inplace=True)
        basePath = Path(channels_df['StitchPath'].iloc[-1])
        pkl_path = basePath / '05 PKL single cell' / f"{excel_path.stem}_pixel_dataframe.pkl"

        if not pkl_path.exists():
            print(f"Warning: PKL file not found at {pkl_path}")
            return None, None, None, None

        # Load pixel data and aggregate to cells
        df = pd.read_pickle(pkl_path)
        df_numeric = df.drop(columns=['mask_type'])
        cell_df = df_numeric.groupby('cell_id').mean().reset_index()

        # Get available markers
        marker_cols = [col for col in cell_df.columns if col.startswith('Intensity_')]
        available_markers = [col.replace('Intensity_', '') for col in marker_cols]

        # Load background image from "02 TIF stitched registered"
        # Get parent directory (1 level up from last StitchPath)
        image_dir = basePath.parent / "02 TIF stitched registered"

        # Find first TIFF file in the directory
        background = None
        if image_dir.exists():
            tif_files = list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff"))
            if tif_files:
                image_path = tif_files[0]
                background = tifffile.imread(str(image_path))
                print(f"Loaded background image: {image_path.name}")
            else:
                print(f"Warning: No TIFF files found in {image_dir}")
        else:
            print(f"Warning: Image directory not found at {image_dir}")

        # Load Cycle 5 DAPI/Hoechst image
        # Find Cycle 5 path from channels_df
        cycle5_row = channels_df[(channels_df['Cycle'] == 5) &
                                  ((channels_df['Marker'] == 'DAPI') | (channels_df['Marker'] == 'Hoechst'))]

        dapi_image = None
        if not cycle5_row.empty:
            cycle5_path = Path(cycle5_row['StitchPath'].iloc[0])
            dapi_dir = cycle5_path / "02 TIF stitched registered"
            if dapi_dir.exists():
                dapi_files = list(dapi_dir.glob("*.tif")) + list(dapi_dir.glob("*.tiff"))
                if dapi_files:
                    dapi_image = tifffile.imread(str(dapi_files[0]))
                    print(f"Loaded DAPI/Hoechst image: {dapi_files[0].name}")

        return cell_df, background, dapi_image, excel_path.stem, available_markers
    except Exception as e:
        print(f"Error loading data from {excel_path}: {e}")
        return None, None, None, None, None

def build_bipartite_proximity_graph(cell_df, marker1, marker2, radius=50):
    """
    Build bipartite proximity graph connecting cells of marker1 to nearby cells of marker2.
    Uses radius-based proximity (Jiaxun's approach).
    Returns:
        - marker1_positions: positions of marker1 cells
        - marker2_positions: positions of marker2 cells
        - edges: list of edge coordinates connecting marker1 to marker2
    """
    # Threshold for positive cells (top 50% of expression)
    marker1_col = f'Intensity_{marker1}'
    marker2_col = f'Intensity_{marker2}'

    # Filter for positive cells (above median expression)
    marker1_threshold = cell_df[marker1_col].median()
    marker2_threshold = cell_df[marker2_col].median()

    marker1_cells = cell_df[cell_df[marker1_col] > marker1_threshold]
    marker2_cells = cell_df[cell_df[marker2_col] > marker2_threshold]

    # Get positions (Y, X for Napari)
    marker1_positions = marker1_cells[['Y', 'X']].values
    marker2_positions = marker2_cells[['Y', 'X']].values

    # Build KDTree for radius-based search
    tree1 = KDTree(marker1_positions)
    tree2 = KDTree(marker2_positions)

    # Find all pairs within radius
    indexes = tree1.query_ball_tree(tree2, r=radius)

    # Build edges
    edges = []
    for i, neighbors in enumerate(indexes):
        for j in neighbors:
            edges.append([marker1_positions[i], marker2_positions[j]])

    return marker1_positions, marker2_positions, edges

def visualize_proximity_map(viewer, cell_df, background, dapi_image, sample_name, marker1, marker2, radius=50):
    """
    Visualize bipartite proximity graph in Napari (Jiaxun's approach).
    """
    viewer.layers.clear()

    # Set scale bar properties
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    viewer.scale_bar.font_size = 36

    # Add background image if available
    if background is not None:
        img_layer = viewer.add_image(background, name=f'{sample_name} - Background',
                        colormap='gray', blending='additive', opacity=0.5,
                        scale=(0.325, 0.325))  # Set pixel size in um

    # Add DAPI/Hoechst image if available
    if dapi_image is not None:
        viewer.add_image(dapi_image, name=f'{sample_name} - DAPI/Hoechst',
                        colormap='blue', blending='additive', opacity=1.0,
                        scale=(0.325, 0.325))  # Set pixel size in um

    # Build bipartite proximity graph
    marker1_positions, marker2_positions, edges = build_bipartite_proximity_graph(
        cell_df, marker1, marker2, radius=radius)

    # Add marker1 cells (magenta)
    if len(marker1_positions) > 0:
        viewer.add_points(marker1_positions, name=f'{sample_name} - {marker1}',
                         size=25, face_color='magenta', blending='additive',
                         scale=(0.325, 0.325))  # Set pixel size in um

    # Add marker2 cells (green)
    if len(marker2_positions) > 0:
        viewer.add_points(marker2_positions, name=f'{sample_name} - {marker2}',
                         size=25, face_color='green', blending='additive',
                         scale=(0.325, 0.325))  # Set pixel size in um

    # Add proximity edges (yellow)
    if edges:
        viewer.add_shapes(edges, shape_type='line', name=f'{sample_name} - Proximity',
                         edge_color='yellow', edge_width=15, opacity=1.0, blending='additive',
                         scale=(0.325, 0.325))  # Set pixel size in um

    print(f"Loaded: {sample_name}")
    print(f"  {marker1}: {len(marker1_positions)} cells (magenta)")
    print(f"  {marker2}: {len(marker2_positions)} cells (green)")
    print(f"  Proximity edges: {len(edges)} (radius={radius}px)")

def main():
    """
    Interactive Napari viewer for bipartite proximity maps (Jiaxun's approach).
    """
    master_excel_path = Path("../Data/5Nov2025_all_slides.xlsx")

    try:
        master_df = pd.read_excel(master_excel_path)
    except Exception as e:
        print(f"Error reading master Excel file: {e}")
        return

    # Create sample selection options
    sample_options = {}
    for _, row in master_df.iterrows():
        label = f"{Path(row['ExcelPath']).stem} ({row['Diet']}-{row['Fibrosis']})"
        sample_options[label] = row['ExcelPath']

    # Load first sample to get available markers
    first_excel_path = Path(sample_options[list(sample_options.keys())[0]])
    _, _, _, _, available_markers = load_sample_data(first_excel_path)

    if available_markers is None:
        print("ERROR: Could not load initial sample to get marker list")
        return

    # Create Napari viewer
    viewer = napari.Viewer()

    # Create sample selector widget with marker selection
    @magicgui(
        call_button='Load Proximity Map',
        sample={'choices': list(sample_options.keys())},
        marker1={'choices': available_markers, 'label': 'Marker 1 (Magenta)'},
        marker2={'choices': available_markers, 'label': 'Marker 2 (Green)'},
        radius={'widget_type': 'Slider', 'min': 10, 'max': 200, 'value': 60, 'label': 'Proximity Radius (px)'}
    )
    def load_sample_widget(sample: str, marker1: str = '5-mC', marker2: str = 'Fibronectin', radius: int = 60):
        print(f"\n=== Loading proximity map ===")
        print(f"Sample: {sample}")
        print(f"Marker 1 (magenta): {marker1}")
        print(f"Marker 2 (green): {marker2}")
        print(f"Radius: {radius}px")

        excel_path = Path(sample_options[sample])
        cell_df, background, dapi_image, sample_name, _ = load_sample_data(excel_path)

        if cell_df is not None:
            # Check if markers exist
            marker1_col = f'Intensity_{marker1}'
            marker2_col = f'Intensity_{marker2}'

            if marker1_col not in cell_df.columns:
                print(f"ERROR: {marker1} not found in dataset")
                return
            if marker2_col not in cell_df.columns:
                print(f"ERROR: {marker2} not found in dataset")
                return

            print(f"Visualizing...")
            visualize_proximity_map(viewer, cell_df, background, dapi_image, sample_name, marker1, marker2, radius)
            print(f"=== Complete ===\n")
        else:
            print(f"ERROR: Failed to load data for {sample}")

    viewer.window.add_dock_widget(load_sample_widget, area='right')

    # Load first sample by default with 5-mC and Fibronectin
    if sample_options and available_markers:
        first_sample = list(sample_options.keys())[0]
        excel_path = Path(sample_options[first_sample])
        cell_df, background, dapi_image, sample_name, _ = load_sample_data(excel_path)

        if cell_df is not None:
            # Use default markers if available
            default_m1 = '5-mC' if '5-mC' in available_markers else available_markers[0]
            default_m2 = 'Fibronectin' if 'Fibronectin' in available_markers else (available_markers[1] if len(available_markers) > 1 else available_markers[0])
            visualize_proximity_map(viewer, cell_df, background, dapi_image, sample_name, default_m1, default_m2, radius=60)

    napari.run()

if __name__ == '__main__':
    main()
