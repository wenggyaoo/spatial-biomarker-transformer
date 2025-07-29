import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np

from config import Config
from data_loader import create_data_loaders
from esm_embedder import BiomarkerEmbedder
from spatial_sampler import SpatialSampler
from model import SpatialBiomarkerTransformer


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # Create data loaders
        self.train_loader, self.val_loader, all_biomarkers, num_cell_types = create_data_loaders(config)
        self.num_cell_types = num_cell_types

        # Get cell type mapping from train dataset
        self.cell_type_to_idx = self.train_loader.dataset.cell_type_to_idx

        # Initialize biomarker embedder
        self.biomarker_embedder = BiomarkerEmbedder(config)
        self.biomarker_embedder.build_biomarker_vocab(all_biomarkers)
        self.biomarker_embedder.to(self.device)

        # Initialize spatial sampler
        self.spatial_sampler = SpatialSampler(config)

        # Initialize model
        self.model = SpatialBiomarkerTransformer(config, self.biomarker_embedder, num_cell_types)
        self.model.to(self.device)

        # === MODIFICATION START ===
        # Loss functions for all potential tasks are kept.
        # Their use is determined by the weights in config.py.
        self.mask_loss_fn = nn.MSELoss()
        self.celltype_loss_fn = nn.CrossEntropyLoss()
        self.reconstruction_loss_fn = nn.MSELoss()
        # === MODIFICATION END ===

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)

        # Logging
        self.writer = SummaryWriter()

        # Create model save directory
        os.makedirs(config.model_save_path, exist_ok=True)

        # Print masking configuration
        self._print_masking_config()

    def _print_masking_config(self):
        """Print current masking configuration"""
        print("\n" + "=" * 50)
        print("INTENSITY MASKING CONFIGURATION")
        print("=" * 50)
        print(f"Center cell intensity masking: {self.config.enable_center_intensity_masking}")
        if self.config.enable_center_intensity_masking:
            print(f"  - Probability: {self.config.center_intensity_mask_probability}")
            print(
                f"  - Min/Max ratio: {self.config.center_intensity_min_mask_ratio}/{self.config.center_intensity_max_mask_ratio}")
            print(f"  - Strategy: {self.config.center_mask_strategy}")
            print(f"  - Mask value: {self.config.center_intensity_mask_value}")

        print(f"Neighbor intensity masking: {self.config.enable_neighbor_intensity_masking}")
        if self.config.enable_neighbor_intensity_masking:
            print(f"  - Probability: {self.config.neighbor_intensity_mask_probability}")
            print(
                f"  - Min/Max ratio: {self.config.neighbor_intensity_min_mask_ratio}/{self.config.neighbor_intensity_max_mask_ratio}")
            print(f"  - Strategy: {self.config.neighbor_mask_strategy}")
            print(f"  - Mask value: {self.config.neighbor_intensity_mask_value}")

        print(f"Preserve top biomarkers: {self.config.preserve_top_biomarkers}")
        print("=" * 50 + "\n")

    def sample_batch(self, batch_regions):
        """Sample spatial neighborhoods from batch regions"""
        sampled_batch = []
        for region in batch_regions:
            sampled_region = self.spatial_sampler.sample_region(region)
            sampled_batch.append(sampled_region)
        return sampled_batch

    # === MODIFICATION START ===
    # Kept for future use if cell type prediction is re-enabled.
    # def compute_celltype_accuracy(self, celltype_logits, cell_type_labels):
    #     """Compute accuracy for cell type prediction"""
    #     predictions = torch.argmax(celltype_logits, dim=1)
    #     correct = (predictions == cell_type_labels).float()
    #     return correct.mean().item()
    # === MODIFICATION END ===

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_mask_loss = 0.0
        # total_celltype_loss = 0.0 # Commented out for now
        # total_recon_loss = 0.0 # Commented out for now
        # total_celltype_acc = 0.0 # Commented out for now
        num_batches = 0

        # Masking statistics
        epoch_masking_stats = {
            'center_masked_ratio': [],
            'neighbor_masked_ratio': [],
            'center_cells_with_masking': 0,
            'neighbor_cells_with_masking': 0
        }

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_regions in pbar:
            sampled_batch = self.sample_batch(batch_regions)

            if len(sampled_batch) == 0:
                continue

            self.optimizer.zero_grad()

            # Forward pass for all tasks
            results = self.model(sampled_batch, self.cell_type_to_idx, task='all')

            # Collect masking statistics
            masking_stats = results['masking_stats']
            if masking_stats['center_mask_ratio'] > 0:
                epoch_masking_stats['center_masked_ratio'].append(masking_stats['center_mask_ratio'])
                epoch_masking_stats['center_cells_with_masking'] += masking_stats['center_cells_masked']
            if masking_stats['neighbor_mask_ratio'] > 0:
                epoch_masking_stats['neighbor_masked_ratio'].append(masking_stats['neighbor_mask_ratio'])
                epoch_masking_stats['neighbor_cells_with_masking'] += masking_stats['neighbor_cells_masked']

            # Compute losses for all tasks
            mask_loss = self.mask_loss_fn(
                results['mask_predictions'],
                results['mask_targets']
            )

            # === MODIFICATION START ===
            # The following losses are calculated but will be multiplied by a weight of 0,
            # so they won't affect the training. This code is kept for easy reactivation.
            celltype_loss = self.celltype_loss_fn(
                results['celltype_logits'],
                results['cell_type_labels']
            )

            recon_loss = self.reconstruction_loss_fn(
                results['reconstruction'],
                results['reconstruction_targets']
            )

            # Combined loss with weights from config
            total_batch_loss = (
                    self.config.prediction_loss_weight * mask_loss +
                    self.config.celltype_loss_weight * celltype_loss +
                    self.config.reconstruction_loss_weight * recon_loss
            )
            # === MODIFICATION END ===

            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # === MODIFICATION START ===
            # Accuracy calculation is commented out as it's not the current focus.
            # celltype_acc = self.compute_celltype_accuracy(
            #     results['celltype_logits'],
            #     results['cell_type_labels']
            # )
            # === MODIFICATION END ===

            # Update metrics
            total_loss += total_batch_loss.item()
            total_mask_loss += mask_loss.item()
            # total_celltype_loss += celltype_loss.item()
            # total_recon_loss += recon_loss.item()
            # total_celltype_acc += celltype_acc
            num_batches += 1

            # Update progress bar with relevant info
            center_mask_pct = masking_stats['center_mask_ratio'] * 100
            neighbor_mask_pct = masking_stats['neighbor_mask_ratio'] * 100

            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'MaskLoss': f'{mask_loss.item():.4f}',
                # 'CTypeLoss': f'{celltype_loss.item():.4f}', # Commented out
                'CMask%': f'{center_mask_pct:.1f}',
                'NMask%': f'{neighbor_mask_pct:.1f}'
            })

        avg_center_mask_ratio = np.mean(epoch_masking_stats['center_masked_ratio']) if epoch_masking_stats[
            'center_masked_ratio'] else 0
        avg_neighbor_mask_ratio = np.mean(epoch_masking_stats['neighbor_masked_ratio']) if epoch_masking_stats[
            'neighbor_masked_ratio'] else 0

        return {
            'total_loss': total_loss / num_batches,
            'mask_loss': total_mask_loss / num_batches,
            # 'celltype_loss': total_celltype_loss / num_batches,
            # 'reconstruction_loss': total_recon_loss / num_batches,
            # 'celltype_accuracy': total_celltype_acc / num_batches,
            'avg_center_mask_ratio': avg_center_mask_ratio,
            'avg_neighbor_mask_ratio': avg_neighbor_mask_ratio,
        }

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_mask_loss = 0.0
        # total_celltype_loss = 0.0
        # total_recon_loss = 0.0
        # total_celltype_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_regions in tqdm(self.val_loader, desc="Validation"):
                sampled_batch = self.sample_batch(batch_regions)

                if len(sampled_batch) == 0:
                    continue

                # Forward pass
                results = self.model(sampled_batch, self.cell_type_to_idx, task='all')

                # Compute losses
                mask_loss = self.mask_loss_fn(
                    results['mask_predictions'],
                    results['mask_targets']
                )

                celltype_loss = self.celltype_loss_fn(
                    results['celltype_logits'],
                    results['cell_type_labels']
                )

                recon_loss = self.reconstruction_loss_fn(
                    results['reconstruction'],
                    results['reconstruction_targets']
                )

                total_batch_loss = (
                        self.config.prediction_loss_weight * mask_loss +
                        self.config.celltype_loss_weight * celltype_loss +
                        self.config.reconstruction_loss_weight * recon_loss
                )

                # celltype_acc = self.compute_celltype_accuracy(
                #     results['celltype_logits'],
                #     results['cell_type_labels']
                # )

                total_loss += total_batch_loss.item()
                total_mask_loss += mask_loss.item()
                # total_celltype_loss += celltype_loss.item()
                # total_recon_loss += recon_loss.item()
                # total_celltype_acc += celltype_acc
                num_batches += 1

        return {
            'total_loss': total_loss / num_batches,
            'mask_loss': total_mask_loss / num_batches,
            # 'celltype_loss': total_celltype_loss / num_batches,
            # 'reconstruction_loss': total_recon_loss / num_batches,
            # 'celltype_accuracy': total_celltype_acc / num_batches
        }

    def train(self):
        """Main training loop"""
        # === MODIFICATION START ===
        # The model is now saved based on the best (lowest) validation loss,
        # as accuracy is no longer the primary metric.
        best_val_loss = float('inf')
        # best_val_acc = 0.0 # Commented out
        # === MODIFICATION END ===

        print(f"Starting training with {self.config.embedding_method} embeddings...")
        print(f"Biomarker embedding dimension: {self.biomarker_embedder.get_embedding_dim()}")
        print(f"Model dimension: {self.config.d_model}")

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            # Logging
            self.writer.add_scalar('Loss/Train_Total', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Loss/Val_Total', val_metrics['total_loss'], epoch)
            self.writer.add_scalar('Loss/Train_Mask', train_metrics['mask_loss'], epoch)
            self.writer.add_scalar('Loss/Val_Mask', val_metrics['mask_loss'], epoch)
            # self.writer.add_scalar('Accuracy/Val_CellType', val_metrics['celltype_accuracy'], epoch) # Commented out

            # Log masking statistics
            self.writer.add_scalar('Masking/Center_Mask_Ratio', train_metrics['avg_center_mask_ratio'], epoch)
            self.writer.add_scalar('Masking/Neighbor_Mask_Ratio', train_metrics['avg_neighbor_mask_ratio'], epoch)

            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])

            # === MODIFICATION START ===
            # Save best model based on validation loss
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self.save_model('best_model.pth')
                print(f"  -> New best validation loss: {best_val_loss:.4f}. Model saved.")
            # === MODIFICATION END ===

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth')

            print(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"Mask Loss: {train_metrics['mask_loss']:.4f}")
            print(f"        Center Mask: {train_metrics['avg_center_mask_ratio']:.2%}, "
                  f"Neighbor Mask: {train_metrics['avg_neighbor_mask_ratio']:.2%}")
            print(f"Val   - Loss: {val_metrics['total_loss']:.4f}, "
                  f"Mask Loss: {val_metrics['mask_loss']:.4f}")

    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'embedder_state_dict': self.biomarker_embedder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'cell_type_to_idx': self.cell_type_to_idx,
            'num_cell_types': self.num_cell_types
        }
        torch.save(checkpoint, os.path.join(self.config.model_save_path, filename))
        print(f"Model saved: {filename}")


if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()