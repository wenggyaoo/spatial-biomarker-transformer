import torch
import numpy as np
from scipy.stats import rankdata
from typing import List, Tuple, Dict, Optional
from config import Config


class IntensityMasker:
    """
    Handles intensity-level masking for center cells and neighboring cells separately
    
    This class does not perform any kind of normalization on the masked/visible intensities.
    """

    def __init__(self, config: Config):
        self.config = config

    def mask_sample(self, sample_data):
        center_intensities = sample_data['intensities'][0:1]  # (1, num_biomarkers)
        neighbor_intensities = sample_data['intensities'][1:]  # (num_neighbors, num_biomarkers)
        
        center_masked, center_mask_flag = self.apply_center_intensity_masking(center_intensities)  # (1, num_biomarkers)
        neighbor_masked, neighbor_mask_flag = self.apply_neighbor_intensity_masking(neighbor_intensities)  # (num_neighbors, num_biomarkers)
        
        masked_items = {sample_data['biomarkers'][i]: center_intensities[0, i]
                        for i in range(len(sample_data['biomarkers'])) if center_mask_flag[0, i] == 1}
        
        sample_data['masked_intensities'] = np.concatenate([center_masked, neighbor_masked], 0)
        sample_data['mask_flags'] = np.concatenate([center_mask_flag, neighbor_mask_flag], 0)
        sample_data['masked_items'] = masked_items        
        return sample_data

    def apply_center_intensity_masking(self, center_intensities):
        """
        Apply intensity masking to center cell

        Args:
            center_intensities: 2d numpy array of biomarker intensities, (1, num_biomarkers)

        Returns:
            masked_intensities: Intensities with some values masked
            mask_indices: Indices of masked positions
            original_intensities: Original intensity values
        """
        assert center_intensities.ndim == 2 and center_intensities.shape[0] == 1
        if center_intensities.shape[1] == 0:
            raise ValueError("Center cell has no intensities")

        if not self.config.enable_center_intensity_masking:
            return center_intensities, np.zeros_like(center_intensities)

        if np.random.random() > self.config.center_intensity_mask_probability:
            return center_intensities, np.zeros_like(center_intensities)

        # Determine number of intensities to mask
        num_biomarkers = center_intensities.shape[1]
        min_mask = max(1, int(num_biomarkers * self.config.center_intensity_min_mask_ratio))
        max_mask = min(num_biomarkers, int(num_biomarkers * self.config.center_intensity_max_mask_ratio))
        num_to_mask = np.random.randint(min_mask, max_mask + 1, (1,))

        # Preserve top biomarkers if specified
        protected_mask = np.zeros_like(center_intensities, dtype=int)
        if self.config.preserve_top_biomarkers > 0:
            protected_mask = self._select_masks(
                center_intensities, np.array([self.config.preserve_top_biomarkers]),
                'highest', np.zeros_like(center_intensities))

        # Select indices to mask based on strategy
        mask_flag = self._select_masks(
            center_intensities,
            num_to_mask,
            self.config.center_mask_strategy,
            protected_mask)

        # Apply masking
        masked_intensities = center_intensities * (1 - mask_flag)
        return masked_intensities, mask_flag

    def apply_neighbor_intensity_masking(self, neighbor_intensities):
        """
        Apply intensity masking to neighbor cells

        Args:
            neighbor_biomarkers: List of biomarker lists for each neighbor
            neighbor_intensities: List of intensity lists for each neighbor

        Returns:
            masked_intensities: Intensities with some values masked for each neighbor
            mask_indices: Indices of masked positions for each neighbor
            original_intensities: Original intensity values for each neighbor
        """
        if not self.config.enable_neighbor_intensity_masking:
            return neighbor_intensities, np.zeros_like(neighbor_intensities)

        # Determine number of intensities to mask for each neighbor
        n_neighbors, num_biomarkers = neighbor_intensities.shape
        min_mask = max(1, int(num_biomarkers * self.config.neighbor_intensity_min_mask_ratio))
        max_mask = min(num_biomarkers, int(num_biomarkers * self.config.neighbor_intensity_max_mask_ratio))
        num_to_mask = np.random.randint(min_mask, max_mask + 1, (n_neighbors,))
        num_to_mask = np.where(np.random.rand(n_neighbors) < self.config.neighbor_intensity_mask_probability, num_to_mask, 0)            

        # Select indices to mask
        mask_flag = self._select_masks(
            neighbor_intensities,
            num_to_mask,
            self.config.neighbor_mask_strategy,
            np.zeros_like(neighbor_intensities, dtype=bool))

        # Apply masking
        masked_intensities = neighbor_intensities * (1 - mask_flag)
        return masked_intensities, mask_flag

    def _select_masks(self,
                      intensities: np.ndarray,
                      num_to_mask: np.ndarray,
                      strategy: str,
                      protected_mask: np.ndarray) -> np.ndarray:
        """
        Select indices to mask based on strategy
        
        Args:
            intensities: 2d numpy array of biomarker intensities: (num_cells, num_biomarkers)
            num_to_mask: 1d numpy array of number of intensities to mask for each cell: (num_cells,)
            strategy: Masking strategy ('random', 'highest', 'lowest', 'middle')
            protected_mask: 2d numpy array indicating protected positions: (num_cells, num_biomarkers),
                1 for protected, 0 for available
        """
        assert intensities.ndim == 2
        assert intensities.shape == protected_mask.shape
        assert num_to_mask.ndim == 1 and num_to_mask.shape[0] == intensities.shape[0]
        available_mask = 1 - (protected_mask * 1)  # 1 for available, 0 for protected

        if np.sum(available_mask) == 0 or np.sum(num_to_mask) == 0:
            return np.zeros_like(intensities, dtype=int)

        if strategy == 'random':
            # Generate random values for all available positions
            random_values = np.random.rand(*intensities.shape)
            # Set random values for protected positions to +inf (to exclude them)
            random_values = np.where(available_mask, random_values, np.inf)
            # Rank each row and select top `num_to_mask` positions per row
            random_ranks = rankdata(random_values, axis=1)
            selected_mask = (random_ranks <= num_to_mask[:, None]).astype(int)

        elif strategy == 'highest':
            # Set protected positions to -inf (to exclude them)
            _intensities = np.where(intensities == intensities, intensities, -np.inf)  # Replace NaN with -inf
            _intensities = np.where(available_mask, _intensities, -np.inf)
            # Rank in descending order
            ranks = rankdata(-_intensities, axis=1)
            selected_mask = (ranks <= num_to_mask[:, None]).astype(int)

        elif strategy == 'lowest':
            # Set protected positions to +inf (to exclude them)
            _intensities = np.where(intensities == intensities, intensities, np.inf)  # Replace NaN with +inf
            _intensities = np.where(available_mask, _intensities, np.inf)
            # Rank in ascending order
            ranks = rankdata(_intensities, axis=1)
            selected_mask = (ranks <= num_to_mask[:, None]).astype(int)

        elif strategy == 'middle':
            medians = np.nanmedian(intensities, axis=1, keepdims=True)
            # Set protected positions to median
            _intensities = np.where(intensities == intensities, intensities, np.Inf)
            _intensities = np.where(available_mask, _intensities, np.Inf)
            # Compute absolute differences from the median
            deviations = np.abs(_intensities - medians)
            ranks = rankdata(deviations, axis=1)
            selected_mask = (ranks <= num_to_mask[:, None]).astype(int)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Ensure only available positions are masked
        selected_mask = ((selected_mask * available_mask) > 0).astype(int)
        return selected_mask
