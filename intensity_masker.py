import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from config import Config


class IntensityMasker:
    """
    Handles intensity-level masking for center cells and neighboring cells separately
    """

    def __init__(self, config: Config):
        self.config = config

    def apply_center_intensity_masking(self,
                                       center_biomarkers: List[str],
                                       center_intensities: List[float]) -> Tuple[List[float], List[int], List[float]]:
        """
        Apply intensity masking to center cell

        Args:
            center_biomarkers: List of biomarker names for center cell
            center_intensities: List of intensity values for center cell

        Returns:
            masked_intensities: Intensities with some values masked
            mask_indices: Indices of masked positions
            original_intensities: Original intensity values
        """
        if not self.config.enable_center_intensity_masking:
            return center_intensities.copy(), [], center_intensities.copy()

        if len(center_intensities) == 0:
            return [], [], []

        # Decide whether to apply masking based on probability
        if np.random.random() > self.config.center_intensity_mask_probability:
            return center_intensities.copy(), [], center_intensities.copy()

        # Determine number of intensities to mask
        min_mask = max(1, int(len(center_intensities) * self.config.center_intensity_min_mask_ratio))
        max_mask = min(len(center_intensities),
                       int(len(center_intensities) * self.config.center_intensity_max_mask_ratio))
        num_to_mask = np.random.randint(min_mask, max_mask + 1)

        # Preserve top biomarkers if specified
        protected_indices = set()
        if self.config.preserve_top_biomarkers > 0:
            # Get indices of top biomarkers by intensity
            intensity_array = np.array(center_intensities)
            top_indices = np.argsort(intensity_array)[-self.config.preserve_top_biomarkers:]
            protected_indices = set(top_indices.tolist())

        # Select indices to mask based on strategy
        mask_indices = self._select_mask_indices(
            center_intensities,
            num_to_mask,
            self.config.center_mask_strategy,
            protected_indices
        )

        # Apply masking
        masked_intensities = center_intensities.copy()
        for idx in mask_indices:
            masked_intensities[idx] = self.config.center_intensity_mask_value

        return masked_intensities, mask_indices, center_intensities.copy()

    def apply_neighbor_intensity_masking(self,
                                         neighbor_biomarkers: List[List[str]],
                                         neighbor_intensities: List[List[float]]) -> Tuple[
        List[List[float]], List[List[int]], List[List[float]]]:
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
            return [intensities.copy() for intensities in neighbor_intensities], \
                [[] for _ in neighbor_intensities], \
                [intensities.copy() for intensities in neighbor_intensities]

        masked_intensities = []
        all_mask_indices = []
        original_intensities = [intensities.copy() for intensities in neighbor_intensities]

        for neighbor_idx, intensities in enumerate(neighbor_intensities):
            if len(intensities) == 0:
                masked_intensities.append([])
                all_mask_indices.append([])
                continue

            # Decide whether to apply masking based on probability
            if np.random.random() > self.config.neighbor_intensity_mask_probability:
                masked_intensities.append(intensities.copy())
                all_mask_indices.append([])
                continue

            # Determine number of intensities to mask
            min_mask = max(1, int(len(intensities) * self.config.neighbor_intensity_min_mask_ratio))
            max_mask = min(len(intensities), int(len(intensities) * self.config.neighbor_intensity_max_mask_ratio))
            num_to_mask = np.random.randint(min_mask, max_mask + 1)

            # Select indices to mask
            mask_indices = self._select_mask_indices(
                intensities,
                num_to_mask,
                self.config.neighbor_mask_strategy,
                set()  # No protected indices for neighbors
            )

            # Apply masking
            neighbor_masked = intensities.copy()
            for idx in mask_indices:
                neighbor_masked[idx] = self.config.neighbor_intensity_mask_value

            masked_intensities.append(neighbor_masked)
            all_mask_indices.append(mask_indices)

        return masked_intensities, all_mask_indices, original_intensities

    def _select_mask_indices(self,
                             intensities: List[float],
                             num_to_mask: int,
                             strategy: str,
                             protected_indices: set) -> List[int]:
        """
        Select indices to mask based on strategy
        """
        available_indices = [i for i in range(len(intensities)) if i not in protected_indices]

        if len(available_indices) == 0:
            return []

        if num_to_mask >= len(available_indices):
            return available_indices

        intensity_array = np.array([intensities[i] for i in available_indices])

        if strategy == 'random':
            selected_available = np.random.choice(available_indices, num_to_mask, replace=False)
            return selected_available.tolist()

        elif strategy == 'highest':
            # Mask highest intensity values
            sorted_indices = np.argsort(intensity_array)[::-1]  # Descending order
            selected_available_idx = sorted_indices[:num_to_mask]
            return [available_indices[i] for i in selected_available_idx]

        elif strategy == 'lowest':
            # Mask lowest intensity values
            sorted_indices = np.argsort(intensity_array)  # Ascending order
            selected_available_idx = sorted_indices[:num_to_mask]
            return [available_indices[i] for i in selected_available_idx]

        elif strategy == 'middle':
            # Mask middle-range intensity values
            sorted_indices = np.argsort(intensity_array)
            start_idx = max(0, len(sorted_indices) // 4)
            end_idx = min(len(sorted_indices), 3 * len(sorted_indices) // 4)
            middle_indices = sorted_indices[start_idx:end_idx]

            if len(middle_indices) < num_to_mask:
                # Fall back to random if not enough middle values
                selected_available = np.random.choice(available_indices, num_to_mask, replace=False)
                return selected_available.tolist()
            else:
                selected_middle = np.random.choice(middle_indices, num_to_mask, replace=False)
                return [available_indices[i] for i in selected_middle]

        else:
            # Default to random
            selected_available = np.random.choice(available_indices, num_to_mask, replace=False)
            return selected_available.tolist()

    def get_masking_statistics(self, mask_info: Dict) -> Dict:
        """Calculate statistics about the applied masking"""
        stats = {
            'center_cells_masked': 0,
            'center_intensities_masked': 0,
            'center_total_intensities': 0,
            'neighbor_cells_masked': 0,
            'neighbor_intensities_masked': 0,
            'neighbor_total_intensities': 0,
            'center_mask_ratio': 0.0,
            'neighbor_mask_ratio': 0.0
        }

        if 'center_mask_indices' in mask_info:
            center_masks = mask_info['center_mask_indices']
            for mask_indices in center_masks:
                if len(mask_indices) > 0:
                    stats['center_cells_masked'] += 1
                    stats['center_intensities_masked'] += len(mask_indices)

        if 'center_original_intensities' in mask_info:
            for intensities in mask_info['center_original_intensities']:
                stats['center_total_intensities'] += len(intensities)

        if 'neighbor_mask_indices' in mask_info:
            neighbor_masks = mask_info['neighbor_mask_indices']
            for cell_masks in neighbor_masks:
                for mask_indices in cell_masks:
                    if len(mask_indices) > 0:
                        stats['neighbor_cells_masked'] += 1
                        stats['neighbor_intensities_masked'] += len(mask_indices)

        if 'neighbor_original_intensities' in mask_info:
            for cell_intensities in mask_info['neighbor_original_intensities']:
                for intensities in cell_intensities:
                    stats['neighbor_total_intensities'] += len(intensities)

        # Calculate ratios
        if stats['center_total_intensities'] > 0:
            stats['center_mask_ratio'] = stats['center_intensities_masked'] / stats['center_total_intensities']

        if stats['neighbor_total_intensities'] > 0:
            stats['neighbor_mask_ratio'] = stats['neighbor_intensities_masked'] / stats['neighbor_total_intensities']

        return stats