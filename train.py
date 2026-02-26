import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from spatial_sampler import SpatialSampler

class Trainer:
    def __init__(self, config, student_model, teacher_model, train_dataset):
        self.config = config
        self.student_model = student_model.to(config.device)
        self.teacher_model = teacher_model.to(config.device)
        self.train_dataset = train_dataset

        self.sampler = SpatialSampler(config)
        self.recon_loss_fn = nn.MSELoss()
        self.distill_loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(
            student_model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        # For LR warmup
        self.global_step = 0
        self.warmup_steps = config.warmup_steps

        # EMA
        self.ema_momentum = config.ema_momentum
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Wandb init
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config=config.__dict__,
            dir=config.model_save_path
        )
        wandb.watch(self.student_model, log="all", log_freq=100)

    def ema_update(self):
        with torch.no_grad():
            for student_param, teacher_param in zip(self.student_model.parameters(), self.teacher_model.parameters()):
                teacher_param.data = self.ema_momentum * teacher_param.data + (1.0 - self.ema_momentum) * student_param.data

    def get_lr(self):
        if self.global_step < self.warmup_steps:
            return self.config.initial_lr + (self.config.lr - self.config.initial_lr) * \
                (self.global_step / max(1, self.warmup_steps))
        else:
            return self.config.lr

    def build_subgraph_dataset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        train_subgraphs = []
        for i in range(len(self.train_dataset)):
            region = self.train_dataset[i]
            for _ in range(self.config.n_subgraphs_per_region):
                subgraph = self.sampler.sample_region(region)
                train_subgraphs.append(subgraph)

        np.random.shuffle(train_subgraphs)
        print(f"Assembled {len(train_subgraphs)} training subgraphs")
        return train_subgraphs

    def train_epoch(self, epoch):
        self.student_model.train()
        self.teacher_model.eval()

        subgraphs = self.build_subgraph_dataset(seed=epoch)
        train_loader = DataLoader(
            subgraphs,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=4,
            pin_memory=True
        )

        recon_losses, distill_losses, total_losses = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        for batch_regions in pbar:
            self.global_step += 1

            # Update LR (warmup)
            current_lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            self.optimizer.zero_grad()

            student_results = self.student_model(batch_regions)
            with torch.no_grad():
                teacher_results = self.teacher_model(batch_regions)

            total_loss = 0.0
            recon_loss = self.recon_loss_fn(
                student_results['recon_target'], student_results['recon_results'])
            total_loss += self.config.reconstruction_loss_weight * recon_loss
            
            distill_loss = self.distill_loss_fn(
                teacher_results['center_encoded'], student_results['center_encoded'])
            total_loss += self.config.distillation_loss_weight * distill_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.ema_update()

            # Logging
            recon_losses.append(recon_loss.item())
            distill_losses.append(distill_loss.item())
            total_losses.append(total_loss.item())

            if self.global_step % 50 == 0:
                pbar.set_postfix({
                    'recon': f'{np.mean(recon_losses[-50:]):.4f}',
                    'distill': f'{np.mean(distill_losses[-50:]):.4f}',
                    'lr': f'{current_lr:.2e}'
                })

            # Wandb log every 10 steps
            if self.global_step % 10 == 0:
                wandb.log({
                    "train/recon_loss": np.mean(recon_losses[-10:]),
                    "train/distill_loss": np.mean(distill_losses[-10:]),
                    "train/total_loss": np.mean(total_losses[-10:]),
                    "train/lr": current_lr,
                    "train/global_step": self.global_step,
                }, step=self.global_step)

        # Epoch summary
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_recon_loss": np.mean(recon_losses),
            "train/epoch_distill_loss": np.mean(distill_losses),
            "train/epoch_total_loss": np.mean(total_losses),
        }, step=self.global_step)

        return np.mean(recon_losses), np.mean(distill_losses)

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'student_state_dict': self.student_model.state_dict(),
            'teacher_state_dict': self.teacher_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        filename = f"checkpoint_epoch_{epoch+1}.pth"
        path = os.path.join(self.config.model_save_path, filename)
        torch.save(checkpoint, path)

    def train(self):
        os.makedirs(self.config.model_save_path, exist_ok=True)

        for epoch in range(self.config.num_epochs):
            epoch_losses = self.train_epoch(epoch)
            self.save_checkpoint(epoch)
            print(f"Epoch {epoch+1} | losses: {epoch_losses}")

        wandb.finish()