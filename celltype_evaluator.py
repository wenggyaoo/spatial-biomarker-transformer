import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE


class CellTypeEvaluator:
    """
    Handles cell-type evaluation tasks:
    1. Linear Probing (with optional backbone freezing)
    2. KNN Classification
    """
    
    def __init__(self, config, model, dataset, cell_type_to_idx):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.cell_type_to_idx = cell_type_to_idx
        self.idx_to_cell_type = {v: k for k, v in cell_type_to_idx.items()}
        # self.scheduler = None
        
        self.train_embeddings = None
        self.train_labels = None
        self.train_label_names = None
        self.test_embeddings = None
        self.test_labels = None
        self.test_label_names = None
        
    def linear_probe(self, train_loader, val_loader, sampler):
        """
        Train linear probe on frozen or unfrozen backbone
        
        Args:
            train_loader: DataLoader for training regions
            val_loader: DataLoader for validation regions
            sampler: SpatialSampler instance
        
        Returns:
            Dictionary with training history and metrics
        """
        print("Starting Linear Probing Evaluation")

        self.model.eval()
        
        # Freeze backbone if configured
        if self.config.freeze_backbone_for_probing:
            print("Freezing backbone parameters...")
            self.model.freeze_backbone()
        else:
            print("Not freezing bakcbone parameters")
        
        # Setup optimizer (only for celltype_head if backbone is frozen)
        if self.config.freeze_backbone_for_probing:
            optimizer = optim.Adam(
                self.model.celltype_head.parameters(),
                lr=self.config.linear_probe_lr
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.linear_probe_lr
            )
        
        # self.scheduler = self._get_scheduler(optimizer, train_loader)
        
        criterion = nn.CrossEntropyLoss()
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(self.config.linear_probe_epochs):
            # Training
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion, sampler
            )
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            val_loss, val_acc = self._validate_epoch(
                val_loader, criterion, sampler
            )
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{self.config.linear_probe_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        # Unfreeze backbone after probing
        if self.config.freeze_backbone_for_probing:
            self.model.unfreeze_backbone()
            print("Backbone unfrozen after linear probing")
        
        print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
        return history
    
    def _train_epoch(self, data_loader, optimizer, criterion, sampler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_regions in tqdm(data_loader, desc="Training"):
            # Sample subgraphs
            sampled_batch = [sampler.sample_region(region) for region in batch_regions]
            
            # Filter out samples without cell types
            valid_samples = [s for s in sampled_batch if len(s['cell_types']) > 0]
            if len(valid_samples) == 0:
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            results = self.model(valid_samples, return_celltype_logits=True)
            
            if 'celltype_logits' not in results:
                continue
            
            # Get labels
            labels = [self.cell_type_to_idx.get(ct, 0) for ct in results['celltype_labels']]
            labels = torch.tensor(labels, dtype=torch.long, device=self.config.device)
            
            # Compute loss
            loss = criterion(results['celltype_logits'], labels)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(
            #     self.model.parameters() if not self.config.freeze_backbone_for_probing else self.model.celltype_head.parameters(),
            #     max_norm=1.0
            # )

            optimizer.step()

            # if hasattr(self.config, 'scheduler_type') and self.config.scheduler_type != 'reduce_on_plateau':
            #     self.scheduler.step()
            #     current_lr = optimizer.param_groups[0]['lr']
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(results['celltype_logits'], 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy
    
    def _validate_epoch(self, data_loader, criterion, sampler):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_regions in tqdm(data_loader, desc="Validation"):
                sampled_batch = [sampler.sample_region(region) for region in batch_regions]
                valid_samples = [s for s in sampled_batch if len(s['cell_types']) > 0]
                
                if len(valid_samples) == 0:
                    continue
                
                results = self.model(valid_samples, return_celltype_logits=True)
                
                if 'celltype_logits' not in results:
                    continue
                
                labels = [self.cell_type_to_idx.get(ct, 0) for ct in results['celltype_labels']]
                labels = torch.tensor(labels, dtype=torch.long, device=self.config.device)
                
                loss = criterion(results['celltype_logits'], labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(results['celltype_logits'], 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy
    
    def knn_evaluation(self, train_loader, test_loader, sampler):
        """
        Perform KNN evaluation on embeddings and store them for visualization
        
        Args:
            train_loader: DataLoader for training regions
            test_loader: DataLoader for test regions
            sampler: SpatialSampler instance
        
        Returns:
            Dictionary with KNN metrics for each K value
        """
        print("Starting KNN Evaluation")
        
        self.model.eval()
        
        # Extract embeddings and labels from training set
        print("Extracting training embeddings...")
        train_embeddings, train_labels, train_label_names = self._extract_embeddings_with_names(
            train_loader, sampler
        )
        
        # Extract embeddings and labels from test set
        print("Extracting test embeddings...")
        test_embeddings, test_labels, test_label_names = self._extract_embeddings_with_names(
            test_loader, sampler
        )
        
        if len(train_embeddings) == 0 or len(test_embeddings) == 0:
            print("Not enough samples for KNN evaluation")
            return {}
        
        # Store for visualization
        self.train_embeddings = np.array(train_embeddings)
        self.train_labels = np.array(train_labels)
        self.train_label_names = train_label_names
        self.test_embeddings = np.array(test_embeddings)
        self.test_labels = np.array(test_labels)
        self.test_label_names = test_label_names
        
        print(f"Train embeddings shape: {self.train_embeddings.shape}")
        print(f"Test embeddings shape: {self.test_embeddings.shape}")
        
        # Evaluate for different K values
        results = {}
        for k in self.config.knn_k_values:
            print(f"\nEvaluating KNN with K={k}...")
            knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            knn.fit(self.train_embeddings, self.train_labels)
            predictions = knn.predict(self.test_embeddings)
            
            accuracy = accuracy_score(self.test_labels, predictions)
            f1 = f1_score(self.test_labels, predictions, average='weighted')
            
            results[f'k{k}'] = {
                'accuracy': accuracy,
                'f1_score': f1
            }
            
            print(f"K={k} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def _extract_embeddings_with_names(self, data_loader, sampler):
        """Extract embeddings, label indices, and label names from a dataset"""
        self.model.eval()

        embeddings = []
        labels = []
        label_names = []
        
        with torch.no_grad():
            for batch_regions in tqdm(data_loader, desc="Extracting embeddings"):
                sampled_batch = [sampler.sample_region(region) for region in batch_regions]
                valid_samples = [s for s in sampled_batch if len(s['cell_types']) > 0]
                
                if len(valid_samples) == 0:
                    continue
                
                results = self.model(valid_samples, return_celltype_logits=False)
                
                # Extract center cell embeddings
                center_embeddings = results['center_encoded'].cpu().numpy()
                embeddings.extend(center_embeddings)
                
                # Extract labels and names
                for sample in valid_samples:
                    cell_type = sample['cell_types'][0]  # Center cell
                    label_idx = self.cell_type_to_idx.get(cell_type, 0)
                    labels.append(label_idx)
                    label_names.append(cell_type)
        
        return embeddings, labels, label_names
    
    def visualize_embeddings(self, method='umap', use_test=True, save_path=None, 
                           figsize=(14, 10), point_size=50, alpha=0.6):
        """
        Visualize embeddings using UMAP or t-SNE
        
        Args:
            method: 'umap' or 'tsne'
            use_test: If True, visualize test set; if False, visualize train set
            save_path: Path to save the figure (optional)
            figsize: Figure size (width, height)
            point_size: Size of scatter points
            alpha: Transparency of points
        
        Returns:
            embeddings_2d: The 2D projected embeddings
        """
        self.model.eval()

        if method == 'umap':
            print("UMAP not available, falling back to t-SNE")
            method = 'tsne'
        
        # Select dataset
        if use_test:
            if self.test_embeddings is None:
                raise ValueError("No test embeddings found. Run knn_evaluation first!")
            embeddings = self.test_embeddings
            label_names = self.test_label_names
            dataset_name = "Test"
        else:
            if self.train_embeddings is None:
                raise ValueError("No train embeddings found. Run knn_evaluation first!")
            embeddings = self.train_embeddings
            label_names = self.train_label_names
            dataset_name = "Train"
        
        print(f"\n{'='*60}")
        print(f"Visualizing {dataset_name} Embeddings using {method.upper()}")
        print(f"{'='*60}")
        print(f"Total samples: {len(embeddings)}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Count samples per cell type
        from collections import Counter
        cell_type_counts = Counter(label_names)
        print(f"\nCell type distribution:")
        for ct, count in sorted(cell_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ct}: {count} ({100*count/len(label_names):.1f}%)")
        
        # Apply dimensionality reduction
        print(f"\nRunning {method.upper()} dimensionality reduction...")
        if method == 'umap':
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=42,
                verbose=True
            )
        elif method == 'tsne':
            reducer = TSNE(
                n_components=2,
                perplexity=min(30, len(embeddings) - 1),  # Adjust perplexity for small datasets
                random_state=42,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'umap' or 'tsne'")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        print("Dimensionality reduction complete!")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique cell types and create color palette
        unique_labels = sorted(set(label_names))
        n_colors = len(unique_labels)
        
        # Use different color palettes based on number of classes
        if n_colors <= 10:
            colors = sns.color_palette('tab10', n_colors=n_colors)
        elif n_colors <= 20:
            colors = sns.color_palette('tab20', n_colors=n_colors)
        else:
            colors = sns.color_palette('husl', n_colors=n_colors)
        
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        # Plot each cell type
        for label in unique_labels:
            mask = np.array(label_names) == label
            count = mask.sum()
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color_map[label]],
                label=f'{label} (n={count})',
                alpha=alpha,
                s=point_size,
                edgecolors='white',
                linewidth=0.5
            )
        
        # Styling
        ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
        ax.set_title(f'{dataset_name} Set Cell Type Embeddings ({method.upper()})', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
                 shadow=True, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        
        plt.show()
        
        return embeddings_2d
    
    def visualize_both_sets(self, method='umap', save_path=None):
        """
        Visualize both train and test embeddings side by side
        
        Args:
            method: 'umap' or 'tsne'
            save_path: Path to save the figure (optional)
        """
        self.model.eval()

        if self.train_embeddings is None or self.test_embeddings is None:
            raise ValueError("Missing embeddings. Run knn_evaluation first!")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Process train set
        print("Processing train set...")
        if method == 'umap':
            reducer_train = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        else:
            reducer_train = TSNE(n_components=2, perplexity=30, random_state=42)
        
        train_2d = reducer_train.fit_transform(self.train_embeddings)
        
        # Process test set
        print("Processing test set...")
        if method == 'umap':
            reducer_test = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        else:
            reducer_test = TSNE(n_components=2, perplexity=30, random_state=42)
        
        test_2d = reducer_test.fit_transform(self.test_embeddings)
        
        # Get all unique cell types across both sets
        all_labels = sorted(set(self.train_label_names + self.test_label_names))
        colors = sns.color_palette('tab10', n_colors=len(all_labels))
        color_map = {label: colors[i] for i, label in enumerate(all_labels)}
        
        # Plot train set
        for label in set(self.train_label_names):
            mask = np.array(self.train_label_names) == label
            ax1.scatter(train_2d[mask, 0], train_2d[mask, 1],
                       c=[color_map[label]], label=label, alpha=0.6, s=50)
        
        ax1.set_title(f'Train Set ({method.upper()})', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'{method.upper()} 1')
        ax1.set_ylabel(f'{method.upper()} 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot test set
        for label in set(self.test_label_names):
            mask = np.array(self.test_label_names) == label
            ax2.scatter(test_2d[mask, 0], test_2d[mask, 1],
                       c=[color_map[label]], label=label, alpha=0.6, s=50)
        
        ax2.set_title(f'Test Set ({method.upper()})', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'{method.upper()} 1')
        ax2.set_ylabel(f'{method.upper()} 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    # def _get_scheduler(self, optimizer, train_loader):
    #     """
    #     Create learning rate scheduler with warmup
    #     """
    #     total_steps = len(train_loader) * self.config.linear_probe_epochs
    #     warmup_steps = int(total_steps * self.config.linear_probe_warmup_ratio)
        
    #     print(f"Total training steps: {total_steps}")
    #     print(f"Warmup steps: {warmup_steps} ({self.config.linear_probe_warmup_ratio*100:.1f}%)")
        
    #     scheduler_type = self.config.linear_probe_scheduler_type
        
    #     if scheduler_type == 'cosine':
    #         from torch.optim.lr_scheduler import LambdaLR
            
    #         # Get min LR
    #         min_lr_ratio = getattr(self.config, 'linear_probe_min_lr_ratio', 0.01)
            
    #         def lr_lambda(current_step):
    #             if current_step < warmup_steps:
    #                 # Linear warmup: 0 → 1
    #                 return float(current_step) / float(max(1, warmup_steps))
    #             else:
    #                 # Cosine decay: 1 → min_lr_ratio
    #                 progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    #                 cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
    #                 return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
            
    #         scheduler = LambdaLR(optimizer, lr_lambda)
    #         print(f"Using: Linear Warmup + Cosine Annealing (min_lr={self.config.linear_probe_lr * min_lr_ratio:.2e})")
        
    #     elif scheduler_type == 'linear':
    #         from torch.optim.lr_scheduler import LambdaLR
            
    #         def lr_lambda(current_step):
    #             if current_step < warmup_steps:
    #                 return float(current_step) / float(max(1, warmup_steps))
    #             else:
    #                 return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
            
    #         scheduler = LambdaLR(optimizer, lr_lambda)
    #         print("Using: Linear Warmup + Linear Decay")
        
    #     elif scheduler_type == 'constant_warmup':
    #         from torch.optim.lr_scheduler import LambdaLR
            
    #         def lr_lambda(current_step):
    #             if current_step < warmup_steps:
    #                 return float(current_step) / float(max(1, warmup_steps))
    #             else:
    #                 return 1.0
            
    #         scheduler = LambdaLR(optimizer, lr_lambda)
    #         print("Using: Linear Warmup + Constant LR")
        
    #     elif scheduler_type == 'reduce_on_plateau':
    #         from torch.optim.lr_scheduler import ReduceLROnPlateau
            
    #         scheduler = ReduceLROnPlateau(
    #             optimizer, 
    #             mode='min', 
    #             factor=0.5, 
    #             patience=5,  # Reduce LR if no improvement for 5 epochs
    #             verbose=True,
    #             min_lr=self.config.linear_probe_lr * 0.01
    #         )
    #         print("Using: ReduceLROnPlateau (no warmup)")
        
    #     else:
    #         raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
    #     return scheduler