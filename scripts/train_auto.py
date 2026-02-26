# train_manual_config.py

import os
import sys
import logging
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- Assume these are your custom modules in the same directory or in PYTHONPATH ---
# If these files are in a different location, you might need to add them to your path:
# sys.path.append('/path/to/your/modules')
from config import Config
from data_loader import SpatialBiomarkerDataset
from spatial_sampler import SpatialSampler
from intensity_masker import IntensityMasker
from esm_embedder import BiomarkerEmbedder
from model import SpatialBiomarkerTransformer

# ==============================================================================
# >> MANUAL CONFIGURATION SECTION <<
# Edit the values below to configure your training run.
# ==============================================================================
class TrainingConfig:
    # --- Path Settings ---
    TRAIN_DATA_ROOT = '/autofs/bal14/khguo/data/train'
    EVAL_DATA_ROOT = '/autofs/bal14/khguo/data/val'
    CHECKPOINT_DIR = '/autofs/bal14/khguo/model_v1/checkpoints/one'
    LOG_DIR = '/autofs/bal14/khguo/model_v1/logs/one'

    # --- Training Hyperparameters ---
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 500

    # --- Data Generation Settings ---
    SUBGRAPHS_PER_REGION = 1000
    SUBGRAPHS_PER_REGION_EVAL = 200

    # --- System Settings ---
    DEVICE = 'cuda:1'  # e.g., "cpu", "cuda:0", "cuda:1"
    RESUME_TRAINING = False  # Set to True to resume from the last checkpoint
# ==============================================================================
# >> END OF CONFIGURATION SECTION <<
# ==============================================================================


def setup_logging(log_dir):
    """Sets up logging to both a file and the console."""
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves the model and training state."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    filepath = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        shutil.copyfile(filepath, best_filepath)
        logging.info(f"Saved new best model to {best_filepath}")

def evaluate(model, eval_dataset, sampler, loss_fn, config, training_config):
    """Evaluates the model on the evaluation dataset."""
    model.eval()
    total_val_loss = 0
    
    eval_subgraphs = []
    for i in range(len(eval_dataset)):
        region = eval_dataset[i]
        for _ in range(training_config.SUBGRAPHS_PER_REGION_EVAL):
            eval_subgraphs.append(sampler.sample_region(region))
    eval_loader = DataLoader(eval_subgraphs, batch_size=training_config.BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)

    with torch.no_grad():
        for batch_regions in eval_loader:
            results = model(batch_regions)
            recon_target = results['recon_target'].to(config.device)
            recon_results = results['recon_results'].to(config.device)
            loss = loss_fn(recon_target, recon_results)
            total_val_loss += loss.item()
            
    return total_val_loss / len(eval_loader)

def main():
    """Main training and evaluation function."""
    # --- Setup Logging ---
    setup_logging(TrainingConfig.LOG_DIR)
    logging.info("Starting training script with manual configuration.")
    
    # Log the configuration
    config_vars = {k: v for k, v in TrainingConfig.__dict__.items() if not k.startswith('__')}
    logging.info(f"Configuration: {config_vars}")

    # --- 1. Model-specific Configuration ---
    logging.info("Setting up model configuration...")
    config = Config()
    config.use_cell_type = False
    config.embedding_method = "onehot"
    config.device = TrainingConfig.DEVICE
    config.num_encoder_layers = 6
    config.dim_feedforward = 1024
    config.initial_lr = TrainingConfig.LEARNING_RATE
    config.lr = TrainingConfig.LEARNING_RATE
    config.weight_decay = 1e-5
    config.spec_index_recon = False
    
    if "cuda" in config.device and not torch.cuda.is_available():
        logging.error("CUDA is not available. Falling back to CPU.")
        config.device = "cpu"
    
    logging.info(f"Using device: {config.device}")

    # --- 2. Data Loading ---
    logging.info("Loading train and validation datasets...")
    train_dataset = SpatialBiomarkerDataset(TrainingConfig.TRAIN_DATA_ROOT, config=config)
    eval_dataset = SpatialBiomarkerDataset(TrainingConfig.EVAL_DATA_ROOT, config=config)
    
    # --- 3. Sampler, Masker, and Embedder Setup ---
    logging.info("Initializing sampler, masker, and embedder...")
    sampler = SpatialSampler(config)
    intensity_masker = IntensityMasker(config)
    biomarker_embedder = BiomarkerEmbedder(config)
    biomarker_embedder.build_biomarker_vocab(train_dataset.all_biomarkers)
    logging.info(f"Biomarker embedding dimension: {biomarker_embedder.embedding_dim}")

    # --- 4. Model Initialization ---
    logging.info("Initializing student and teacher models...")
    student_model = SpatialBiomarkerTransformer(config, biomarker_embedder=biomarker_embedder, intensity_masker=intensity_masker, is_teacher_model=False)
    teacher_model = SpatialBiomarkerTransformer(config, biomarker_embedder=biomarker_embedder, intensity_masker=intensity_masker, is_teacher_model=True)
    teacher_model.load_state_dict(student_model.state_dict())
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.to(config.device)
    student_model.to(config.device)

    # --- 5. Optimizer and Loss Functions ---
    reconstruction_loss_fn = nn.MSELoss()
    distill_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(student_model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay, betas=(0.9, 0.999))

    # --- 6. Data Preparation (Subgraph Generation) ---
    logging.info("Generating subgraphs for training...")
    train_subgraphs = []
    for i in range(len(train_dataset)):
        region = train_dataset[i]
        for _ in range(TrainingConfig.SUBGRAPHS_PER_REGION):
            train_subgraphs.append(sampler.sample_region(region))
    np.random.shuffle(train_subgraphs)
    logging.info(f"Assembled training subgraph dataset with {len(train_subgraphs)} samples.")
    train_loader = DataLoader(train_subgraphs, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

    # --- 7. Checkpoint Loading (Resume Training) ---
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if TrainingConfig.RESUME_TRAINING:
        checkpoint_path = os.path.join(TrainingConfig.CHECKPOINT_DIR, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_path):
            logging.info(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            best_val_loss = checkpoint['best_val_loss']
            student_model.load_state_dict(checkpoint['student_state_dict'])
            teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f"Resuming training from epoch {start_epoch + 1}")
        else:
            logging.warning(f"RESUME_TRAINING is True, but no checkpoint found at '{checkpoint_path}'. Starting from scratch.")

    # --- 8. Training Loop ---
    logging.info("Starting training loop...")
    ema_momentum = 0.999
    warmup_steps = TrainingConfig.WARMUP_STEPS
    peak_lr = config.initial_lr

    for epoch in range(start_epoch, TrainingConfig.EPOCHS):
        student_model.train()
        teacher_model.eval()
        recon_total_loss, distill_total_loss, total_total_loss = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TrainingConfig.EPOCHS}")
        for batch_regions in pbar:
            global_step += 1
            if global_step < warmup_steps:
                lr_scale = global_step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = peak_lr * lr_scale
            
            optimizer.zero_grad()
            student_results = student_model(batch_regions)
            with torch.no_grad():
                teacher_results = teacher_model(batch_regions)

            recon_loss = reconstruction_loss_fn(student_results['recon_target'], student_results['recon_results'])
            distill_loss = distill_loss_fn(teacher_results['center_encoded'], student_results['center_encoded'])
            total_loss = recon_loss * 0.5 + distill_loss * 0.5
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
                    teacher_param.data.mul_(ema_momentum).add_(student_param.data, alpha=1.0 - ema_momentum)

            recon_total_loss += recon_loss.item()
            distill_total_loss += distill_loss.item()
            total_total_loss += total_loss.item()
            pbar.set_postfix({'Loss': total_loss.item(), 'Recon': recon_loss.item(), 'LR': optimizer.param_groups[0]['lr']})

        avg_total_loss = total_total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} Summary | Train Loss: {avg_total_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # --- 9. Validation and Checkpointing ---
        val_loss = evaluate(student_model, eval_dataset, sampler, reconstruction_loss_fn, config, TrainingConfig)
        logging.info(f"Epoch {epoch+1} | Validation Loss: {val_loss:.4f}")

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'global_step': global_step,
            'student_state_dict': student_model.state_dict(),
            'teacher_state_dict': teacher_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, is_best, checkpoint_dir=TrainingConfig.CHECKPOINT_DIR)
        logging.info(f"Saved checkpoint for epoch {epoch+1} to {TrainingConfig.CHECKPOINT_DIR}")

    logging.info("Training finished.")

if __name__ == '__main__':
    # Create directories if they don't exist before starting
    if not os.path.exists(TrainingConfig.CHECKPOINT_DIR):
        os.makedirs(TrainingConfig.CHECKPOINT_DIR)
    if not os.path.exists(TrainingConfig.LOG_DIR):
        os.makedirs(TrainingConfig.LOG_DIR)
        
    main()