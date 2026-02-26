import torch
import numpy as np
from typing import List, Tuple, Dict
import random
from scipy.spatial import KDTree


class SpatialSampler:
    """
    Handle spatial sampling of cells and their neighbors with cell type information.
    This version is optimized using a KDTree for efficient neighbor searches.

    It implements a conditional neighbor finding strategy:
    1. It first finds all neighbors within a given radius.
    2. If the count of these neighbors exceeds `max_neighbors`, it discards them
       and instead finds the `max_neighbors` closest cells.
    """

    def __init__(self, config):
        self.config = config

    def find_neighbors(self, center_idx: int, kdtree: KDTree) -> List[int]:
        """
        Find neighboring cells using a conditional strategy.

        need parameters `max_neighbors` and `neighbor_distance_threshold` from config.
        
        
        Args:
            center_idx: The index of the center cell in the original coordinate list.
            kdtree: The pre-built KDTree of all cell coordinates.

        Returns:
            A list of indices for the neighboring cells.
        """
        center_coord = kdtree.data[center_idx]

        n_to_query = self.config.max_neighbors + 1
        distances, indices = kdtree.query(center_coord, k=n_to_query)
        filtered_neighbors = [
            idx for dist, idx in zip(distances, indices) 
            if dist <= self.config.neighbor_distance_threshold and idx != center_idx
        ]
        assert len(filtered_neighbors) <= self.config.max_neighbors
        return filtered_neighbors

    def sample_region(self, region_data: Dict) -> Dict:
        """
        Sample a center cell and its neighbors from a region.

        Args:
            region_data: Dictionary containing 'coordinates', 'kdtree', 'cell_types', 'biomarkers', 'intensities'.

        Returns:
            Dictionary with sampled data including cell types.
        """
        coordinates = region_data['coordinates']
        intensities = region_data['intensities']
        cell_ids = region_data['cell_ids']
        cell_types = region_data['cell_types']

        if region_data['num_cells'] == 0:
            return {
                'coordinates': [],
                'intensities': [],
                'cell_ids': [],
                'cell_types': [],
                'biomarkers': [],
                'original_center_idx': None,
            }

        # Build the KDTree from all coordinates in the region.
        kdtree = region_data['kdtree']

        # Randomly sample a center cell
        center_idx = random.randint(0, len(coordinates) - 1)

        # Find neighbors using the new conditional logic
        neighbor_indices = self.find_neighbors(center_idx, kdtree)

        # Create relative coordinates (relative to the center cell)
        center_coord = coordinates[center_idx]

        # Gather all sampled indices: the center cell plus its neighbors
        all_indices = [center_idx] + neighbor_indices
        sampled_coords = coordinates[all_indices]
        assert sampled_coords.shape == (len(all_indices), 2)
        relative_coords = sampled_coords - center_coord.reshape((1, 2))

        return {
            'coordinates': relative_coords,
            'intensities': intensities[all_indices],
            'cell_ids': [cell_ids[i] for i in all_indices],
            'cell_types': [cell_types[i] for i in all_indices] if cell_types else [],
            'biomarkers': region_data['biomarkers'],
            'study_name': region_data['study_name'],
            'region_name': region_data['region_name'],
            'original_center_idx': center_idx,
        }
