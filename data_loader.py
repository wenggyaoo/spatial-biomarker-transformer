import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path
from scipy.spatial import KDTree


def normalize_biomarker_expression(bm_exp_df):
    """
    Normalize biomarker expression table

    Only support arcsinh-zscore pipeline for now

    Args:
        bm_exp_df (pd.DataFrame): dataframe of raw biomarker expression

    Returns:
        pd.DataFrame: dataframe of normalized biomarker expression
    """
    assert 'CELL_ID' not in bm_exp_df.columns
    bm_exp_df = np.arcsinh(bm_exp_df / (5 * np.quantile(bm_exp_df, 0.2, axis=0) + 1e-5))
    bm_exp_df = bm_exp_df / bm_exp_df.std(0)
    
    # Clip extreme values to 5
    bm_exp_df = np.clip(bm_exp_df, 0, 5)
    
    # 0-1 normalize the expression intensities
    # bm_exp_df = bm_exp_df.div((1e-5 + bm_exp_df.sum(axis=1)), axis=0)
    return bm_exp_df


class SpatialBiomarkerDataset(Dataset):
    """
    Dataset for spatial biomarker data with optional cell type information
    """

    def __init__(self, data_path: str, config, valid_biomarkers: Optional[List[str]] = None):
        self.config = config
        self.regions = []
        self.all_biomarkers = set()
        self.all_cell_types = set()
        self.cell_type_to_idx = {}
        self.valid_biomarkers = load_biomarker_info_from_csv(config)[0] if \
            valid_biomarkers is None else valid_biomarkers
        
        self.biomarker_rename = json.load(open(config.biomarker_name_mapping_file, 'r'))

        # data_path = os.path.join(data_path, 'train')

        self.load_data(data_path)
        if self.config.use_cell_types:
            self.build_cell_type_vocab()
        self.all_biomarkers = list(self.all_biomarkers)

    def load_data(self, data_path: str):
        """
        Load data from the hierarchical folder structure
        Expected structure:
        data_path/
        ├── study1/
        │   ├── region1/
        │   │   ├── cell_data.csv
        │   │   ├── expression.csv
        │   │   └── cell_type.csv (optional, for future use)
        │   ├── region2/
        │   │   ├── cell_data.csv
        │   │   ├── expression.csv
        │   │   └── cell_type.csv (optional, for future use)
        │   └── ...
        ├── study2/
        │   ├── region1/
        │   │   ├── cell_data.csv
        │   │   ├── expression.csv
        │   │   └── cell_type.csv (optional, for future use)
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
                            region_data = self.load_region_data(region_path)
                            if region_data and len(region_data['coordinates']) >= self.config.min_cells_per_region:
                                region_data['study_name'] = study_dir
                                region_data['region_name'] = region_dir
                                self.regions.append(region_data)
                        except Exception as e:
                            print(f"Error processing region {region_dir} in study {study_dir}: {e}")

        print(f"Loaded {len(self.regions)} regions from {data_path}")
        print(f"Found {len(self.all_biomarkers)} unique biomarkers")
        if self.config.use_cell_types:
            print(f"Found {len(self.all_cell_types)} unique cell types")
        else:
            print("Cell type loading is disabled")

    def load_region_data(self, region_path: str) -> Optional[Dict]:
        """Load and process data for a single region"""
        # Define required file paths
        position_path = os.path.join(region_path, self.config.position_filename)
        biomarker_path = os.path.join(region_path, self.config.biomarker_filename)

        # Cell type file is optional for future use
        celltype_path = None
        if self.config.use_cell_types:
            celltype_path = os.path.join(region_path, self.config.celltype_filename)

        # Check if required files exist
        required_files = [position_path, biomarker_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing required file: {file_path}")

        # Check optional cell type file
        celltype_df = None
        if self.config.use_cell_types and celltype_path and os.path.exists(celltype_path):
            try:
                celltype_df = pd.read_csv(celltype_path)
            except Exception as e:
                print(f"Warning: Could not load cell type file {celltype_path}: {e}")
                print("Proceeding without cell type information")

        # Load required CSV files
        position_df = pd.read_csv(position_path)
        biomarker_df = pd.read_csv(biomarker_path)

        # Process the data
        region_data = self.process_region_dataframes(celltype_df, position_df, biomarker_df)
        return region_data

    def process_region_dataframes(self,
                                  celltype_df: Optional[pd.DataFrame],
                                  position_df: pd.DataFrame,
                                  biomarker_df: pd.DataFrame) -> Dict:
        """
        Process the dataframes into required format

        Expected formats:
        - celltype_df: columns ['cell_id', 'cell_type'] (optional, for future use)
        - position_df: columns ['cell_id', 'x', 'y']
        - biomarker_df: columns ['cell_id', 'biomarker1', 'biomarker2', ...]
        """

        # Identify column names (handle variations)
        celltype_cols = None
        if self.config.use_cell_types and celltype_df is not None:
            celltype_cols = self.identify_celltype_columns(celltype_df)

        position_cols = self.identify_position_columns(position_df)
        biomarker_cols = self.identify_biomarker_columns(biomarker_df)
        if not position_cols:
            raise ValueError("Could not identify required position columns")
        if not biomarker_cols:
            raise ValueError("Could not identify required biomarker columns")

        # Extract position & biomarker column names
        pos_cell_id_col = position_cols['cell_id']
        x_col = position_cols['x']
        y_col = position_cols['y']
        bio_cell_id_col = biomarker_cols['cell_id']
        biomarker_features = biomarker_cols['biomarkers']
        assert position_df[pos_cell_id_col].is_unique
        assert position_df[pos_cell_id_col].tolist() == biomarker_df[bio_cell_id_col].tolist()

        # Extract cell type columns (for future use)
        cell_id_col = None
        celltype_col = None
        if celltype_cols:
            cell_id_col = celltype_cols['cell_id']
            celltype_col = celltype_cols['cell_type']
            assert position_df[pos_cell_id_col].tolist() == celltype_df[cell_id_col].tolist()

        # Extract biomarker expression matrix (exclude cell_id column)
        if self.valid_biomarkers:
            excluded = set()
            for bm in biomarker_features:
                if bm not in self.valid_biomarkers:
                    if bm not in self.biomarker_rename:
                        excluded.add(bm)
                    else:
                        # Some are renamed to EMPTY or invalid 
                        if self.biomarker_rename[bm] not in self.valid_biomarkers:
                            excluded.add(bm)

            if excluded:
                print(f"Excluding biomarkers: {excluded}")
            biomarker_features = [b for b in biomarker_features if b not in excluded]
        biomarker_expression_df = biomarker_df[biomarker_features].copy()
        biomarker_expression_df = normalize_biomarker_expression(biomarker_expression_df)

        # Compose region data
        region_data = {
            'coordinates': np.array(position_df[[x_col, y_col]]),
            'intensities': np.array(biomarker_expression_df),
            'cell_ids': position_df[pos_cell_id_col].tolist(),
            'biomarkers': [
                bm if bm in self.valid_biomarkers else self.biomarker_rename[bm]
                for bm in biomarker_features
            ],
            'cell_types': [] if celltype_col is None else celltype_df[celltype_col].tolist(),
            'num_cells': position_df.shape[0],
        }
        assert region_data['coordinates'].shape == (region_data['num_cells'], 2)
        assert region_data['intensities'].shape == (region_data['num_cells'], len(biomarker_features))
        assert len(region_data['cell_ids']) == region_data['num_cells']
        region_data['kdtree'] = KDTree(region_data['coordinates'])
        
        self.all_biomarkers.update(region_data['biomarkers'])
        self.all_cell_types.update(region_data['cell_types'])
        return region_data

    def identify_celltype_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Identify cell type and cell ID columns"""
        cell_id_candidates = ['cell_id', 'cellid', 'id', 'cell', 'Cell_ID', 'CellID', 'CELL_ID']
        celltype_candidates = [
            'cell_type', 'celltype', 'type', 'Cell_Type', 'CellType', 
            'cell_class', 'class', 'ANNOTATION_LABEL', 'annotation_label',
            'label', 'Label', 'LABEL'  # Add more variations
        ]

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
                    candidate.lower() in col.lower() for candidate in [
                        'cell_type', 'celltype', 'type', 'annotation', 'label'
                    ]):
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
        """Build cell type vocabulary with unknown class as index 0"""
        # unique_cell_types = sorted(list(self.all_cell_types))
        unique_cell_types = sorted(list(set(str(ct) for ct in self.all_cell_types)))

        # Always put unknown class at index 0
        self.cell_type_to_idx = {"unknown": 0}

        # Add all other cell types starting from index 1
        for idx, cell_type in enumerate(unique_cell_types, start=1):
            if cell_type != "unknown":  # Avoid duplicate if "unknown" already exists
                self.cell_type_to_idx[cell_type] = idx

        self.idx_to_cell_type = {idx: cell_type for cell_type, idx in self.cell_type_to_idx.items()}

    def get_cell_type_index(self, cell_type: str) -> int:
        """Get index for cell type (for future use)"""
        return self.cell_type_to_idx.get(cell_type, 0)  # Default to 0 if unknown

    def get_cell_type_name(self, idx: int) -> str:
        """Get cell type name from index"""
        return self.idx_to_cell_type.get(idx, "unknown")

    def get_cell_type_vocab_size(self) -> int:
        if not self.config.use_cell_types:
            return 0
        return len(self.cell_type_to_idx)

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        return self.regions[idx]

    def get_all_biomarkers(self) -> List[str]:
        return self.all_biomarkers

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

            if self.config.use_cell_types:
                for cell_type in region['cell_types']:
                    cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1

        stats = {
            'total_regions': len(self.regions),
            'total_biomarkers': len(self.all_biomarkers),
            'mean_cells_per_region': np.mean(num_cells_per_region),
            'std_cells_per_region': np.std(num_cells_per_region),
            'min_cells_per_region': np.min(num_cells_per_region),
            'max_cells_per_region': np.max(num_cells_per_region),
            'studies': study_counts,
            'biomarkers': sorted(list(self.all_biomarkers))
        }

        if self.config.use_cell_types:
            stats.update({
                'total_cell_types': len(self.all_cell_types),
                'cell_type_distribution': cell_type_counts,
                'cell_types': sorted(list(self.all_cell_types))
            })
        else:
            stats.update({
                'total_cell_types': 1,  # Only "unknown" cell type
                'cell_type_distribution': {'unknown': sum(len(region['cell_types']) for region in self.regions)},
                'cell_types': ['unknown']
            })

        return stats


def load_biomarker_info_from_csv(config):
    """
    Load biomarker information from CSV file and return valid biomarkers list.
    """
    biomarker_info = {}
    valid_biomarkers = []

    filepath = config.biomarker_sequence_file

    try:
        df = pd.read_csv(filepath)

        # Skip header row and process data
        for _, row in df.iterrows():
            biomarker_name = str(row.iloc[0]).strip()
            amino_acid_seq = row.iloc[1]

            # Check if biomarker is legitimate
            if pd.isna(amino_acid_seq):
                # Legit biomarker but no sequence
                biomarker_info[biomarker_name] = {
                    'type': 'legit_no_sequence',
                    'sequence': None
                }
                valid_biomarkers.append(biomarker_name)
            elif str(amino_acid_seq).upper() == 'N/A':
                # Not a legitimate biomarker
                biomarker_info[biomarker_name] = {
                    'type': 'not_legit',
                    'sequence': None
                }
                # Don't add to valid_biomarkers
            else:
                # Has amino acid sequence
                biomarker_info[biomarker_name] = {
                    'type': 'has_sequence',
                    'sequence': str(amino_acid_seq).strip()
                }
                valid_biomarkers.append(biomarker_name)

        print(f"Loaded biomarker info from {filepath}")
        print(f"  - Valid biomarkers: {len(valid_biomarkers)}")
        print(f"  - Invalid biomarkers: {len(biomarker_info) - len(valid_biomarkers)}")

    except FileNotFoundError:
        print(f"ERROR: The file '{filepath}' was not found.")
        return [], {}
    except Exception as e:
        print(f"ERROR loading biomarker info: {e}")
        return [], {}

    return valid_biomarkers, biomarker_info