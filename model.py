import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from intensity_masker import IntensityMasker


class SpatialBiomarkerTransformer(nn.Module):
    """
    Main transformer model for spatial biomarker analysis with intensity-level masking
    """

    def __init__(self, config, biomarker_embedder, num_cell_types):
        super().__init__()
        self.config = config
        self.biomarker_embedder = biomarker_embedder
        self.num_cell_types = num_cell_types

        # Initialize intensity masker
        self.intensity_masker = IntensityMasker(config)

        # Get embedding dimension from embedder
        self.biomarker_dim = biomarker_embedder.get_embedding_dim()

        # Projection layer to map embedder output to model dimension
        self.embedder_projection = nn.Linear(self.biomarker_dim, config.d_model)

        # Positional embedding
        from positional_embedding import SpatialPositionalEmbedding
        self.pos_embedding = SpatialPositionalEmbedding(config.d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)

        # Task-specific heads
        self.mask_prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, self.biomarker_dim)
        )

        self.celltype_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, num_cell_types)
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, self.biomarker_dim)
        )

        # Learnable query for center cell reconstruction
        self.center_query = nn.Parameter(torch.randn(1, 1, config.d_model))

    def create_embeddings_with_masking(self, batch_data, cell_type_to_idx):
        """
        Create embeddings for a batch with intensity-level masking applied
        """
        batch_size = len(batch_data)
        max_seq_len = max(len(region['coordinates']) for region in batch_data)

        # Original embeddings (without masking)
        original_embeddings = torch.zeros(batch_size, max_seq_len, self.biomarker_dim,
                                          device=self.config.device)

        # Masked embeddings (with intensity masking applied)
        masked_embeddings = torch.zeros(batch_size, max_seq_len, self.biomarker_dim,
                                        device=self.config.device)

        coordinates = torch.zeros(batch_size, max_seq_len, 2, device=self.config.device)
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool,
                                     device=self.config.device)
        cell_type_labels = torch.zeros(batch_size, dtype=torch.long, device=self.config.device)

        # Store masking information
        mask_info = {
            'center_mask_indices': [],
            'center_original_intensities': [],
            'center_masked_intensities': [],
            'neighbor_mask_indices': [],
            'neighbor_original_intensities': [],
            'neighbor_masked_intensities': []
        }

        for i, region in enumerate(batch_data):
            seq_len = len(region['coordinates'])

            # Process center cell (index 0)
            center_biomarkers = region['biomarkers'][0]
            center_intensities = region['intensities'][0]

            # Apply center cell masking
            center_masked_intensities, center_mask_indices, center_original = self.intensity_masker.apply_center_intensity_masking(
                center_biomarkers, center_intensities
            )

            # Store center masking info
            mask_info['center_mask_indices'].append(center_mask_indices)
            mask_info['center_original_intensities'].append(center_original)
            mask_info['center_masked_intensities'].append(center_masked_intensities)

            # Create embeddings for center cell
            center_original_tensor = torch.tensor(center_original, dtype=torch.float32, device=self.config.device)
            center_masked_tensor = torch.tensor(center_masked_intensities, dtype=torch.float32,
                                                device=self.config.device)

            original_embeddings[i, 0] = self.biomarker_embedder(center_biomarkers, center_original_tensor)
            masked_embeddings[i, 0] = self.biomarker_embedder(center_biomarkers, center_masked_tensor)

            # Process neighbor cells
            neighbor_biomarkers = region['biomarkers'][1:seq_len]
            neighbor_intensities = region['intensities'][1:seq_len]

            # Apply neighbor masking
            neighbor_masked_intensities, neighbor_mask_indices, neighbor_original = self.intensity_masker.apply_neighbor_intensity_masking(
                neighbor_biomarkers, neighbor_intensities
            )

            # Store neighbor masking info
            mask_info['neighbor_mask_indices'].append(neighbor_mask_indices)
            mask_info['neighbor_original_intensities'].append(neighbor_original)
            mask_info['neighbor_masked_intensities'].append(neighbor_masked_intensities)

            # Create embeddings for neighbors
            for j, (biomarkers, original_ints, masked_ints) in enumerate(
                    zip(neighbor_biomarkers, neighbor_original, neighbor_masked_intensities)):
                cell_idx = j + 1  # Offset by 1 since center is at index 0

                original_tensor = torch.tensor(original_ints, dtype=torch.float32, device=self.config.device)
                masked_tensor = torch.tensor(masked_ints, dtype=torch.float32, device=self.config.device)

                original_embeddings[i, cell_idx] = self.biomarker_embedder(biomarkers, original_tensor)
                masked_embeddings[i, cell_idx] = self.biomarker_embedder(biomarkers, masked_tensor)

            # Store coordinates
            coords_tensor = torch.tensor(region['coordinates'], dtype=torch.float32, device=self.config.device)
            coordinates[i, :seq_len] = coords_tensor

            # Create attention mask
            attention_mask[i, :seq_len] = True

            # Store center cell type label
            center_cell_type = region['center_cell_type']
            cell_type_labels[i] = cell_type_to_idx.get(center_cell_type, 0)

        return (original_embeddings, masked_embeddings, coordinates,
                attention_mask, cell_type_labels, mask_info)

    def forward(self, batch_data, cell_type_to_idx, task='all'):
        """
        Forward pass with intensity-level masking
        """
        # Create embeddings with masking applied
        (original_embeddings, masked_embeddings, coordinates,
         attention_mask, cell_type_labels, mask_info) = self.create_embeddings_with_masking(
            batch_data, cell_type_to_idx
        )

        # Project embeddings to model dimension
        original_projected = self.embedder_projection(original_embeddings)
        masked_projected = self.embedder_projection(masked_embeddings)

        # Add positional embeddings
        pos_embed = self.pos_embedding(coordinates)
        original_projected = original_projected + pos_embed
        masked_projected = masked_projected + pos_embed

        # Create padding mask for attention
        src_key_padding_mask = ~attention_mask

        results = {
            'cell_type_labels': cell_type_labels,
            'mask_info': mask_info,
            'masking_stats': self.intensity_masker.get_masking_statistics(mask_info)
        }

        # Masked intensity prediction task
        if task in ['mask_prediction', 'all']:
            # Use masked embeddings for encoding
            encoded = self.encoder(masked_projected, src_key_padding_mask=src_key_padding_mask)

            # Use decoder to predict center cell
            batch_size = masked_embeddings.shape[0]
            center_queries = self.center_query.expand(batch_size, -1, -1)

            decoded_center = self.decoder(
                center_queries,
                encoded,
                memory_key_padding_mask=src_key_padding_mask
            )

            # Predict original center cell embedding
            mask_predictions = self.mask_prediction_head(decoded_center.squeeze(1))

            results['mask_predictions'] = mask_predictions
            results['mask_targets'] = original_embeddings[:, 0]  # Original center cell embeddings

        # Cell type prediction
        if task in ['celltype', 'all']:
            # Use original embeddings for cell type prediction, but mask center cell
            celltype_embeddings = original_projected.clone()
            celltype_embeddings[:, 0] = 0  # Completely mask center cell

            encoded_celltype = self.encoder(celltype_embeddings, src_key_padding_mask=src_key_padding_mask)

            # Average neighbor embeddings (exclude center cell)
            neighbor_embeddings = encoded_celltype[:, 1:, :]
            neighbor_mask = attention_mask[:, 1:]

            if neighbor_mask.any():
                neighbor_embeddings_masked = neighbor_embeddings * neighbor_mask.unsqueeze(-1).float()
                neighbor_sum = neighbor_embeddings_masked.sum(dim=1)
                neighbor_count = neighbor_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
                neighbor_avg = neighbor_sum / neighbor_count
            else:
                neighbor_avg = torch.zeros(batch_size, self.config.d_model, device=self.config.device)

            celltype_logits = self.celltype_classifier(neighbor_avg)
            results['celltype_logits'] = celltype_logits

        # Reconstruction task
        if task in ['reconstruction', 'all']:
            # Use original embeddings for reconstruction
            encoded_recon = self.encoder(original_projected, src_key_padding_mask=src_key_padding_mask)

            batch_size = original_embeddings.shape[0]
            center_queries = self.center_query.expand(batch_size, -1, -1)

            decoded_center = self.decoder(
                center_queries,
                encoded_recon,
                memory_key_padding_mask=src_key_padding_mask
            )

            reconstruction = self.reconstruction_head(decoded_center.squeeze(1))
            results['reconstruction'] = reconstruction
            results['reconstruction_targets'] = original_embeddings[:, 0]

        return results