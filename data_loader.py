import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path


class SpatialBiomarkerDataset(Dataset):
    """
    Dataset for spatial biomarker data with cell type information
    """

    def __init__(self, data_path: str, config):
        self.config = config
        self.regions = []
        self.all_biomarkers = set()
        self.all_cell_types = set()
        self.cell_type_to_idx = {}

        self.load_data(data_path)
        self.build_cell_type_vocab()

    def load_data(self, data_path: str):
        """
        Load data from the hierarchical folder structure
        Expected structure:
        data_path/
        ├── study1/
        │   ├── region1/
        │   │   ├── cell_types.csv
        │   │   ├── positions.csv
        │   │   └── biomarkers.csv
        │   ├── region2/
        │   │   ├── cell_types.csv
        │   │   ├── positions.csv
        │   │   └── biomarkers.csv
        │   └── ...
        ├── study2/
        │   ├── region1/
        │   │   ├── cell_types.csv
        │   │   ├── positions.csv
        │   │   └── biomarkers.csv
        │   └── ...
        └── ...
        """
        for study_dir in os.listdir(data_path):
            study_path = os.path.join(data_path, study_dir)
            if os.path.isdir(study_path):
                print(f"Processing study: {study_dir}")

                for region_dir in os.listdir(study_path):
                    region_path = os.path.join(study_path, region_dir)
                    if os.path.isdir(region_path):
                        try:
                            region_data = self.load_region_data(region_path, study_dir, region_dir)
                            if region_data and len(region_data['coordinates']) >= self.config.min_cells_per_region:
                                self.regions.append(region_data)
                        except Exception as e:
                            print(f"Error processing region {region_dir} in study {study_dir}: {e}")

        print(f"Loaded {len(self.regions)} regions from {data_path}")
        print(f"Found {len(self.all_biomarkers)} unique biomarkers")
        print(f"Found {len(self.all_cell_types)} unique cell types")

    def load_region_data(self, region_path: str, study_name: str, region_name: str) -> Optional[Dict]:
        """Load and process data for a single region"""

        # Define expected file paths
        celltype_path = os.path.join(region_path, self.config.celltype_filename)
        position_path = os.path.join(region_path, self.config.position_filename)
        biomarker_path = os.path.join(region_path, self.config.biomarker_filename)

        # Check if all required files exist
        required_files = [celltype_path, position_path, biomarker_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                return None

        try:
            # Load CSV files
            celltype_df = pd.read_csv(celltype_path)
            position_df = pd.read_csv(position_path)
            biomarker_df = pd.read_csv(biomarker_path)

            # Process the data
            region_data = self.process_region_dataframes(
                celltype_df, position_df, biomarker_df, study_name, region_name
            )

            return region_data

        except Exception as e:
            print(f"Error loading region data from {region_path}: {e}")
            return None

    def process_region_dataframes(self, celltype_df: pd.DataFrame, position_df: pd.DataFrame,
                                  biomarker_df: pd.DataFrame, study_name: str, region_name: str) -> Dict:
        """
        Process the three dataframes into required format

        Expected formats:
        - celltype_df: columns ['cell_id', 'cell_type']
        - position_df: columns ['cell_id', 'x', 'y']
        - biomarker_df: columns ['cell_id', 'biomarker1', 'biomarker2', ...]
        """

        # Identify column names (handle variations)
        celltype_cols = self.identify_celltype_columns(celltype_df)
        position_cols = self.identify_position_columns(position_df)
        biomarker_cols = self.identify_biomarker_columns(biomarker_df)

        if not celltype_cols or not position_cols:
            raise ValueError("Could not identify required columns")

        cell_id_col = celltype_cols['cell_id']
        celltype_col = celltype_cols['cell_type']
        pos_cell_id_col = position_cols['cell_id']
        x_col = position_cols['x']
        y_col = position_cols['y']
        bio_cell_id_col = biomarker_cols['cell_id']
        biomarker_features = biomarker_cols['biomarkers']

        # Merge dataframes on cell_id
        merged_df = celltype_df.merge(position_df, left_on=cell_id_col, right_on=pos_cell_id_col, how='inner')
        merged_df = merged_df.merge(biomarker_df, left_on=cell_id_col, right_on=bio_cell_id_col, how='inner')

        # Prepare output data
        coordinates = []
        cell_types = []
        biomarkers = []
        intensities = []

        for _, row in merged_df.iterrows():
            # Get coordinates
            x_coord = float(row[x_col])
            y_coord = float(row[y_col])
            coordinates.append((x_coord, y_coord))

            # Get cell type
            cell_type = str(row[celltype_col])
            cell_types.append(cell_type)
            self.all_cell_types.add(cell_type)

            # Get biomarker data
            cell_biomarkers = []
            cell_intensities = []

            for biomarker in biomarker_features:
                if biomarker in row and pd.notna(row[biomarker]):
                    intensity = float(row[biomarker])
                    if intensity > 0:  # Only include positive intensities
                        cell_biomarkers.append(biomarker)
                        cell_intensities.append(intensity)
                        self.all_biomarkers.add(biomarker)

            if len(cell_biomarkers) >= self.config.min_biomarkers_per_cell:
                biomarkers.append(cell_biomarkers)
                intensities.append(cell_intensities)
            else:
                # Remove corresponding entries if cell doesn't meet criteria
                coordinates.pop()
                cell_types.pop()

        return {
            'coordinates': coordinates,
            'cell_types': cell_types,
            'biomarkers': biomarkers,
            'intensities': intensities,
            'study_name': study_name,
            'region_name': region_name,
            'num_cells': len(coordinates)
        }

    def identify_celltype_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Identify cell type and cell ID columns"""
        cell_id_candidates = ['cell_id', 'cellid', 'id', 'cell', 'Cell_ID', 'CellID']
        celltype_candidates = ['cell_type', 'celltype', 'type', 'Cell_Type', 'CellType', 'cell_class', 'class']

        cell_id_col = None
        celltype_col = None

        # Find cell ID column
        for col in df.columns:
            if col in cell_id_candidates or any(
                    candidate.lower() in col.lower() for candidate in ['cell_id', 'cellid']):
                cell_id_col = col
                break

        # Find cell type column
        for col in df.columns:
            if col in celltype_candidates or any(
                    candidate.lower() in col.lower() for candidate in ['cell_type', 'celltype', 'type']):
                celltype_col = col
                break

        if not cell_id_col or not celltype_col:
            return {}

        return {'cell_id': cell_id_col, 'cell_type': celltype_col}

    def identify_position_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Identify position columns"""
        cell_id_candidates = ['cell_id', 'cellid', 'id', 'cell', 'Cell_ID', 'CellID']
        x_candidates = ['x', 'X', 'x_coord', 'X_coord', 'pos_x', 'centroid_x']
        y_candidates = ['y', 'Y', 'y_coord', 'Y_coord', 'pos_y', 'centroid_y']

        cell_id_col = None
        x_col = None
        y_col = None

        # Find columns
        for col in df.columns:
            col_lower = col.lower()
            if col in cell_id_candidates or 'cell' in col_lower and 'id' in col_lower:
                cell_id_col = col
            elif col in x_candidates or ('x' in col_lower and any(kw in col_lower for kw in ['coord', 'pos'])):
                x_col = col
            elif col in y_candidates or ('y' in col_lower and any(kw in col_lower for kw in ['coord', 'pos'])):
                y_col = col

        if not all([cell_id_col, x_col, y_col]):
            return {}

        return {'cell_id': cell_id_col, 'x': x_col, 'y': y_col}

    def identify_biomarker_columns(self, df: pd.DataFrame) -> Dict:
        """Identify biomarker columns"""
        cell_id_candidates = ['cell_id', 'cellid', 'id', 'cell', 'Cell_ID', 'CellID']

        cell_id_col = None
        biomarker_cols = []

        # Find cell ID column
        for col in df.columns:
            col_lower = col.lower()
            if col in cell_id_candidates or ('cell' in col_lower and 'id' in col_lower):
                cell_id_col = col
                break

        # Find biomarker columns (all numeric columns except cell_id)
        for col in df.columns:
            if col != cell_id_col and df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                biomarker_cols.append(col)

        return {'cell_id': cell_id_col, 'biomarkers': biomarker_cols}

    def build_cell_type_vocab(self):
        """Build cell type vocabulary"""
        unique_cell_types = sorted(list(self.all_cell_types))
        self.cell_type_to_idx = {cell_type: idx for idx, cell_type in enumerate(unique_cell_types)}
        self.idx_to_cell_type = {idx: cell_type for cell_type, idx in self.cell_type_to_idx.items()}

    def get_cell_type_index(self, cell_type: str) -> int:
        """Get index for cell type"""
        return self.cell_type_to_idx.get(cell_type, 0)  # Default to 0 if unknown

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        return self.regions[idx]

    def get_all_biomarkers(self) -> List[str]:
        return list(self.all_biomarkers)

    def get_all_cell_types(self) -> List[str]:
        return list(self.all_cell_types)

    def get_num_cell_types(self) -> int:
        return len(self.all_cell_types)

    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the dataset"""
        num_cells_per_region = [region['num_cells'] for region in self.regions]
        study_counts = {}
        cell_type_counts = {}

        for region in self.regions:
            study = region['study_name']
            study_counts[study] = study_counts.get(study, 0) + 1

            for cell_type in region['cell_types']:
                cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1

        return {
            'total_regions': len(self.regions),
            'total_biomarkers': len(self.all_biomarkers),
            'total_cell_types': len(self.all_cell_types),
            'mean_cells_per_region': np.mean(num_cells_per_region),
            'std_cells_per_region': np.std(num_cells_per_region),
            'min_cells_per_region': np.min(num_cells_per_region),
            'max_cells_per_region': np.max(num_cells_per_region),
            'studies': study_counts,
            'cell_type_distribution': cell_type_counts,
            'biomarkers': sorted(list(self.all_biomarkers)),
            'cell_types': sorted(list(self.all_cell_types))
        }


def create_data_loaders(config):
    """Create training and validation data loaders"""

    # Load datasets
    train_dataset = SpatialBiomarkerDataset(
        os.path.join(config.data_path, 'train'), config
    )
    val_dataset = SpatialBiomarkerDataset(
        os.path.join(config.data_path, 'val'), config
    )

    # Ensure consistent vocabularies
    all_biomarkers = list(set(train_dataset.get_all_biomarkers() + val_dataset.get_all_biomarkers()))
    all_cell_types = list(set(train_dataset.get_all_cell_types() + val_dataset.get_all_cell_types()))

    # Update vocabularies in both datasets
    cell_type_to_idx = {cell_type: idx for idx, cell_type in enumerate(sorted(all_cell_types))}
    train_dataset.cell_type_to_idx = cell_type_to_idx
    val_dataset.cell_type_to_idx = cell_type_to_idx
    train_dataset.all_cell_types = set(all_cell_types)
    val_dataset.all_cell_types = set(all_cell_types)

    # Print dataset statistics
    print("Training Dataset Statistics:")
    train_stats = train_dataset.get_dataset_statistics()
    for key, value in train_stats.items():
        if key not in ['biomarkers', 'cell_types']:
            print(f"  {key}: {value}")

    print("\nValidation Dataset Statistics:")
    val_stats = val_dataset.get_dataset_statistics()
    for key, value in val_stats.items():
        if key not in ['biomarkers', 'cell_types']:
            print(f"  {key}: {value}")

    print(f"\nTotal unique biomarkers: {len(all_biomarkers)}")
    print(f"Total unique cell types: {len(all_cell_types)}")

    # Custom collate function
    def collate_fn(batch):
        return batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0




    )

    return train_loader, val_loader, all_biomarkers, len(all_cell_types)