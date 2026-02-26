import torch
import torch.nn as nn
from intensity_masker import IntensityMasker
from esm_embedder import BiomarkerEmbedder
from positional_embedding import SinusoidalRotationalEmbedding, SinusoidalSpatialEmbedding
import numpy as np


class SpatialBiomarkerTransformer(nn.Module):
    """
    Main transformer model for spatial biomarker analysis with intensity-level masking
    """

    def __init__(self,
                 config,
                 biomarker_embedder=None,
                 intensity_masker=None,
                 is_teacher_model=False):
        super().__init__()
        self.config = config
        self.is_teacher_model = is_teacher_model
        self.masking = not is_teacher_model
        
        # Biomarker embedder
        self.biomarker_embedder = biomarker_embedder
        assert self.biomarker_embedder is not None, "Biomarker embedder must be provided"
        assert self.biomarker_embedder.embedding_dim is not None, "Biomarker embedder is not initialized"
        self.biomarker_dim = self.biomarker_embedder.embedding_dim

        # Initialize intensity masker
        self.intensity_masker = IntensityMasker(config) if intensity_masker is None else intensity_masker

        self.input_norm = nn.LayerNorm(config.d_model)

        # Projection layer to map embedder output to model dimension
        self.embedder_projection = nn.Sequential(
            nn.Linear(self.biomarker_dim, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Positional embedding
        if config.positional_embedder == "rotational":
            self.pos_embedding = SinusoidalRotationalEmbedding(config.d_model)
        else:
            self.pos_embedding = SinusoidalSpatialEmbedding(config.d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(config.d_model + self.biomarker_dim),
            nn.Linear(config.d_model + self.biomarker_dim, 1024),  # Note: d_model + biomarker_dim
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 1)
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Proper weight initialization to prevent gradient explosion"""
        for layer in self.embedder_projection:
            if isinstance(layer, nn.Linear):
                # Xavier/Glorot initialization with smaller scale

                # nn.init.xavier_normal_(layer.weight, gain=0.1)  # Smaller gain
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

        for layer in self.reconstruction_head:
            if isinstance(layer, nn.Linear):
                # Xavier/Glorot initialization with smaller scale

                # nn.init.xavier_normal_(layer.weight, gain=0.1)  # Smaller gain
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

        # Initialize transformer layers more conservatively
        for name, param in self.encoder.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_normal_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def create_embeddings(self, batch_data):
        """
        Create embeddings for a batch with intensity-level masking applied
        """
        device = self.config.device
        batch_size = len(batch_data)
        max_seq_len = max(len(region['coordinates']) for region in batch_data)

        # Original & masked embeddings, coordinates (for positional encoding calculation), and padding masks
        embeddings = torch.zeros(batch_size, max_seq_len, self.biomarker_dim, device=device)
        coordinates = torch.zeros(batch_size, max_seq_len, 2, device=device)
        pad_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)

        masked_biomarker_expression = []

        for i, sample in enumerate(batch_data):
            # Mark padding positions
            seq_len = len(sample['coordinates'])
            pad_mask[i, seq_len:] = True

            # Populate coordinates and original embeddings
            biomarkers = sample['biomarkers']
            coordinates[i, :seq_len] = torch.tensor(sample['coordinates'], dtype=torch.float32, device=device)

            if self.masking:
                masked_sample = self.intensity_masker.mask_sample(sample)
                masked_intensities = masked_sample['masked_intensities']
                if self.config.normalize_masked_expression:
                    masked_intensities = self._renormalize_intensities(masked_intensities)
                embeddings[i, :seq_len] = self.biomarker_embedder(biomarkers, masked_intensities)

                # Extract masked center cell biomarker expression
                masked_biomarker_expression.append(masked_sample['masked_items'])
            else:
                embeddings[i, :seq_len] = self.biomarker_embedder(biomarkers, sample['intensities'])

        return embeddings, coordinates, pad_mask, masked_biomarker_expression

    def _renormalize_intensities(self, masked_intensities):
        """
        Rescale masked intensities to maintain the same sum as original intensities
        """
        raise NotImplementedError("Renormalization not implemented yet")

    def forward(self, batch_data):
        """
        Forward pass with intensity-level masking
        """
        # Create embeddings with masking applied
        embeddings, coordinates, pad_mask, masked_biomarker_expression = \
            self.create_embeddings(batch_data)

        results = {}

        x = self.embedder_projection(embeddings)
        x = x + self.pos_embedding(coordinates)
        x = self.input_norm(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        center_encoded = x[:, 0]
        results['center_encoded'] = center_encoded

        if not self.is_teacher_model:
            # reconstruction task implementation
            if self.config.spec_index_recon:
                results.update(self._perform_spec_batch_reconstruction(
                    center_encoded, batch_data))
            else:
                results.update(self._perform_batch_reconstruction(
                    center_encoded, batch_data, masked_biomarker_expression))

        return results

    def _perform_batch_reconstruction(self, center_encoded, batch_data, masked_biomarker_expression):
        """
        Perform reconstruction for all masked biomarkers across the entire batch

        Args:
            center_encoded: (batch_size, d_model) - encoded center cells
            batch_data: original batch data
            masked_biomarker_expression: list of dicts with masked biomarker info for each sample

        Returns:
            Dictionary with reconstruction targets and predictions
        """

        recon_biomarker_names = []
        recon_sample_indices = []
        recon_targets = []
        target_descs = []

        for sample_idx, (sample_data, masked_items) in enumerate(zip(batch_data, masked_biomarker_expression)):
            for biomarker_name, intensity in masked_items.items():
                recon_biomarker_names.append(biomarker_name)
                recon_sample_indices.append(sample_idx)
                recon_targets.append(intensity)
                target_descs.append((
                    sample_data['study_name'],
                    sample_data['region_name'],
                    sample_data['original_center_idx'],
                    biomarker_name,
                    intensity
                ))

        if len(recon_targets) == 0:
            return {'recon_targets': torch.tensor([], device=self.config.device),
                    'recon_results': torch.tensor([], device=self.config.device),
                    'target_descs': target_descs}

        recon_sample_indices = torch.tensor(recon_sample_indices, device=self.config.device, dtype=torch.long)
        recon_center_encodings = center_encoded[recon_sample_indices]  # (num_recon, d_model)
        recon_biomarker_embs = self.biomarker_embedder.get_batched_embeddings(recon_biomarker_names)  # (num_recon, biomarker_dim)

        recon_targets = torch.tensor(recon_targets, dtype=torch.float32, device=self.config.device)
        recon_inputs = torch.cat([recon_center_encodings, recon_biomarker_embs], dim=-1)  # (num_recon, d_model + biomarker_dim)
        recon_results = self.reconstruction_head(recon_inputs).squeeze(-1)
        return {'recon_target': recon_targets,
                'recon_results': recon_results,
                'target_descs': target_descs}

    # def _perform_spec_batch_reconstruction(self, center_encoded, batch_data):
    #     """
    #     Perform reconstruction for all masked biomarkers across the entire batch

    #     Args:
    #         center_encoded: (batch_size, d_model) - encoded center cells
    #         batch_data: original batch data

    #     Returns:
    #         Dictionary with reconstruction targets and predictions
    #     """
    #     target_biomarker = "PanCK"
    #     all_targets = []
    #     # if target_biomarker not in batch_data[0]:
    #     #     return {
    #     #         'recon_target': [],
    #     #         'recon_results': []
    #     #     }
    #     for sample_data in batch_data:
    #         center_idx = 0
    #         center_target_biomarker_index = sample_data['biomarkers'].index(target_biomarker)
    #         center_target_biomarker_val = sample_data['intensities'][center_idx, center_target_biomarker_index]
    #         all_targets.append(center_target_biomarker_val)
    #     all_targets = torch.tensor(all_targets, dtype=torch.float32, device=self.config.device)

    #     # Use a zero placeholder for biomarker embedding
    #     bm_emb_input = torch.zeros((len(batch_data), self.biomarker_dim), device=self.config.device)
    #     recon_inputs = torch.concat([center_encoded, bm_emb_input], dim=-1)
    #     all_preds = self.reconstruction_head(recon_inputs).squeeze(-1)

    #     if np.random.rand() < 0.02:
    #         print(f"Target stats: mean={all_targets.mean():.4f}, std={all_targets.std():.4f}")
    #         print(f"Prediction stats: mean={all_preds.mean():.4f}, std={all_preds.std():.4f}")
    #         print(f"Sample targets: {all_targets[:5]}")
    #         print(f"Sample predictions: {all_preds[:5]}")

    #     return {
    #         'recon_target': all_targets,
    #         'recon_results': all_preds,
    #     }
    
    def freeze_backbone(self):
        """Freeze all parameters except celltype_head"""
        for name, param in self.named_parameters():
            if 'celltype_head' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True