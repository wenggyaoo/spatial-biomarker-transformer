import matplotlib
matplotlib.use('inline')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

from config import Config
from data_loader import create_data_loaders
from esm_embedder import BiomarkerEmbedder
from spatial_sampler import SpatialSampler
from model_test import SpatialBiomarkerTransformer

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
        # Use biomarkers from CSV instead of data discovery
        self.biomarker_embedder.build_biomarker_vocab(all_biomarkers)
        self.biomarker_embedder.to(self.device)

        # Initialize spatial sampler
        self.spatial_sampler = SpatialSampler(config)

        # Initialize model
        self.model = SpatialBiomarkerTransformer(config, self.biomarker_embedder, self.num_cell_types, is_teacher_model=False)
        self.model.to(self.device)

        self.reconstruction_loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.model.parameters(),
            lr=config.initial_lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.min_lr
            )

        # Logging
        self.writer = SummaryWriter()

        # Create model save directory
        os.makedirs(config.model_save_path, exist_ok=True)

        self.recon_corr_markdown = []

    def calculate_recon_pearson_correlation(self, recon_targets, recon_results):
        targets = recon_targets.tolist()
        predictions = recon_results.tolist()
        x_mean = np.mean(predictions)
        y_mean = np.mean(targets)
        a, b, c = 0, 0, 0
        for i in range(len(predictions)):
            a += (predictions[i] - x_mean) * (targets[i] - y_mean)
            b += np.pow(predictions[i] - x_mean, 2)
            c += np.pow(targets[i] - y_mean, 2)
        return (a / (np.sqrt(b) * np.sqrt(c)))

    def _plot_recon_correlation(self, predictions, targets, epoch_num):
        """Fixed plotting function"""
        # Create the plot
        plt.figure(figsize=(6, 6))
        plt.scatter(predictions, targets,
                    s=50,
                    c='steelblue',
                    alpha=0.7,
                    edgecolors='white',
                    linewidth=0.8)

        plt.xlabel('Predictions')
        plt.ylabel('Ground Truth')
        plt.title(f'Reconstruction Correlation - Epoch {epoch_num}')
        plt.grid(True, alpha=0.3)

        # Calculate range for both axes
        min_val = min(min(predictions), min(targets))
        max_val = max(max(predictions), max(targets))
        padding = (max_val - min_val) * 0.05
        range_min = min_val - padding
        range_max = max_val + padding

        # Set equal ranges
        plt.xlim(range_min, range_max)
        plt.ylim(range_min, range_max)

        # Add y=x reference line
        plt.plot([range_min, range_max], [range_min, range_max],
                 'r--', linewidth=2, label='y = x (Perfect Match)', alpha=0.8)

        plt.legend()
        plt.tight_layout()

        # For Jupyter notebooks
        plt.show()
        plt.close()

    def sample_batch(self, batch_regions):
        """Sample spatial neighborhoods from batch regions"""
        sampled_batch = []
        for region in batch_regions:
            sampled_region = self.spatial_sampler.sample_region(region)
            sampled_batch.append(sampled_region)
        return sampled_batch

    def check_gradients(self):
        total_norm = 0
        param_count = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        total_norm = total_norm ** (1. / 2)
        print(f"Total gradient norm: {total_norm:.6f}")
        return total_norm

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

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

            # Forward pass for student model
            results = self.model(sampled_batch, self.cell_type_to_idx, task='all')
            recon_loss = self.reconstruction_loss_fn(results['recon_target'], results['recon_results'])
            recon_correlation = self.calculate_recon_pearson_correlation(recon_targets = results['recon_target'], recon_results = results['recon_results'])

            # # Collect masking statistics
            # masking_stats = results['masking_stats']
            # if masking_stats['center_mask_ratio'] > 0:
            #     epoch_masking_stats['center_masked_ratio'].append(masking_stats['center_mask_ratio'])
            #     epoch_masking_stats['center_cells_with_masking'] += masking_stats['center_cells_masked']
            # if masking_stats['neighbor_mask_ratio'] > 0:
            #     epoch_masking_stats['neighbor_masked_ratio'].append(masking_stats['neighbor_mask_ratio'])
            #     epoch_masking_stats['neighbor_cells_with_masking'] += masking_stats['neighbor_cells_masked']

            # Combined loss with configurable weights
            total_batch_loss = recon_loss

            # Backward pass
            total_batch_loss.backward()

            grad_norm = self.check_gradients()
            if grad_norm > 10.0:  # Explosion
                print("⚠️ Gradient explosion detected!")
            elif grad_norm < 1e-6:  # Vanishing
                print("⚠️ Gradient vanishing detected!")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += total_batch_loss.item()

            num_batches += 1

            # # Update progress bar with relevant info
            # center_mask_pct = masking_stats['center_mask_ratio'] * 100
            # neighbor_mask_pct = masking_stats['neighbor_mask_ratio'] * 100

            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Recon_Corr': f'{recon_correlation:.3f}',
                # 'CMask%': f'{center_mask_pct:.1f}',
                # 'NMask%': f'{neighbor_mask_pct:.1f}'
            })

        avg_center_mask_ratio = np.mean(epoch_masking_stats['center_masked_ratio']) if epoch_masking_stats[
            'center_masked_ratio'] else 0
        avg_neighbor_mask_ratio = np.mean(epoch_masking_stats['neighbor_masked_ratio']) if epoch_masking_stats[
            'neighbor_masked_ratio'] else 0

        self.scheduler.step()

        return {
            'total_loss': total_loss / num_batches,
            'recon_correlation': recon_correlation,
            'avg_center_mask_ratio': avg_center_mask_ratio,
            'avg_neighbor_mask_ratio': avg_neighbor_mask_ratio,
            'recon_target': results['recon_target'],
            'recon_results': results['recon_results'],
        }

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0

        num_batches = 0

        with torch.no_grad():
            for batch_regions in tqdm(self.val_loader, desc="Validation"):
                sampled_batch = self.sample_batch(batch_regions)

                if len(sampled_batch) == 0:
                    continue

                # Forward pass for student model
                results = self.model(sampled_batch, self.cell_type_to_idx, task='all')
                recon_loss = self.reconstruction_loss_fn(results['recon_target'], results['recon_results'])
                recon_correlation = self.calculate_recon_pearson_correlation(
                    recon_targets=results['recon_target'], recon_results=results['recon_results'])

                # Combined loss with configurable weights
                total_batch_loss = recon_loss

                total_loss += total_batch_loss.item()

                num_batches += 1

        return {
            'total_loss': total_loss / num_batches,
            'recon_correlation': recon_correlation,
        }

    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        best_val_correlation = -1.0  # Track best correlation

        print(f"Starting training with {self.config.embedding_method} embeddings...")
        print(f"Biomarker embedding dimension: {self.biomarker_embedder.get_embedding_dim()}")
        print(f"Model dimension: {self.config.d_model}")

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            train_metrics = self.train_epoch()
            self.recon_corr_markdown.append(train_metrics['recon_correlation'])
            val_metrics = self.validate()

            # Logging
            self.writer.add_scalar('Loss/Train_Total', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Loss/Val_Total', val_metrics['total_loss'], epoch)

            # Log masking statistics
            self.writer.add_scalar('Masking/Center_Mask_Ratio', train_metrics['avg_center_mask_ratio'], epoch)
            self.writer.add_scalar('Masking/Neighbor_Mask_Ratio', train_metrics['avg_neighbor_mask_ratio'], epoch)

            self._plot_recon_correlation(
                predictions=train_metrics['recon_results'].detach().cpu().tolist(),  # Actual data
                targets=train_metrics['recon_target'].detach().cpu().tolist(),  # Actual data
                epoch_num=epoch + 1
            )

            # Learning rate scheduling - for ReduceLROnPlateau
            # self.scheduler.step(val_metrics['total_loss'])

            print(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"Recon-Correlation: {train_metrics['recon_correlation']:.4f}")
            print(f"        Center Mask: {train_metrics['avg_center_mask_ratio']:.2%}, "
                  f"Neighbor Mask: {train_metrics['avg_neighbor_mask_ratio']:.2%}")
            print(f"Val   - Loss: {val_metrics['total_loss']:.4f}, "
                  f"Correlation: {val_metrics['recon_correlation']:.4f}")

        print('Reconstruction task correlation diagram:')
        x_values = list(range(1, len(self.recon_corr_markdown) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, self.recon_corr_markdown,
                 linewidth=2,
                 color='blue',
                 marker='o',
                 markersize=3,
                 alpha=0.7)

        plt.xlabel('Epochs')
        plt.ylabel('Reconstruction Correlation')
        plt.title('Training Progress: Reconstruction Correlation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close()

if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()