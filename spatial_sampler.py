import torch
import numpy as np
from typing import List, Tuple, Dict
import random


class SpatialSampler:
    """
    Handle spatial sampling of cells and their neighbors with cell type information
    """

    def __init__(self, config):
        self.config = config

    def calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two coordinates"""
        return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    def find_neighbors(self, center_idx: int, coordinates: List[Tuple[float, float]]) -> List[int]:
        """
        Find neighboring cells within distance threshold
        """
        center_coord = coordinates[center_idx]
        neighbors = []

        for i, coord in enumerate(coordinates):
            if i != center_idx:
                distance = self.calculate_distance(center_coord, coord)
                if distance <= self.config.neighbor_distance_threshold:
                    neighbors.append(i)

        # Limit number of neighbors
        if len(neighbors) > self.config.max_neighbors:
            neighbors = random.sample(neighbors, self.config.max_neighbors)

        return neighbors

    def sample_region(self, region_data: Dict) -> Dict:
        """
        Sample a center cell and its neighbors from a region

        Args:
            region_data: Dictionary containing 'coordinates', 'cell_types', 'biomarkers', 'intensities'

        Returns:
            Dictionary with sampled data including cell types
        """
        coordinates = region_data['coordinates']
        cell_types = region_data['cell_types']
        biomarkers = region_data['biomarkers']
        intensities = region_data['intensities']

        # Randomly sample center cell
        center_idx = random.randint(0, len(coordinates) - 1)

        # Find neighbors
        neighbor_indices = self.find_neighbors(center_idx, coordinates)

        # Create relative coordinates (relative to center cell)
        center_coord = coordinates[center_idx]
        relative_coords = []

        # Add center cell (at origin)
        relative_coords.append((0.0, 0.0))
        sampled_cell_types = [cell_types[center_idx]]
        sampled_biomarkers = [biomarkers[center_idx]]
        sampled_intensities = [intensities[center_idx]]

        # Add neighbors
        for neighbor_idx in neighbor_indices:
            neighbor_coord = coordinates[neighbor_idx]
            rel_x = neighbor_coord[0] - center_coord[0]
            rel_y = neighbor_coord[1] - center_coord[1]
            relative_coords.append((rel_x, rel_y))
            sampled_cell_types.append(cell_types[neighbor_idx])
            sampled_biomarkers.append(biomarkers[neighbor_idx])
            sampled_intensities.append(intensities[neighbor_idx])

        return {
            'center_idx': 0,  # Center cell is always at index 0
            'coordinates': relative_coords,
            'cell_types': sampled_cell_types,
            'biomarkers': sampled_biomarkers,
            'intensities': sampled_intensities,
            'original_center_idx': center_idx,
            'center_cell_type': cell_types[center_idx]
        }