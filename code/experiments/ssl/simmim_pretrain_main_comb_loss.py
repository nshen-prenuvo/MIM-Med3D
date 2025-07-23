import torch
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
import importlib
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Optimize for Tensor Cores on CUDA devices
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')

from models import ViTSimMIM
from torch.nn import L1Loss
from monai.inferers import SlidingWindowInferer
# from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from optimizers.lr_scheduler import WarmupCosineSchedule
# from utils.schedulers import LinearWarmupCosineAnnealingLR
import data
import optimizers
import os
from torch.optim import AdamW

class SchedulerMonitorCallback(Callback):
    """Callback to monitor scheduler calls and learning rate changes"""
    
    def __init__(self, log_every_n_steps=10):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Monitor learning rate at the start of each training batch"""
        if self.step_count % self.log_every_n_steps == 0:
            # Get current learning rate
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            print(f"Step {self.step_count}: Current LR = {current_lr:.8f}")
            
            # Check if scheduler exists
            if hasattr(trainer, 'lr_schedulers') and trainer.lr_schedulers:
                scheduler = trainer.lr_schedulers[0]
                print(f"Step {self.step_count}: Scheduler type = {type(scheduler).__name__}")
                
                # For WarmupCosineSchedule, we can check the lambda function
                if hasattr(scheduler, 'lr_lambda'):
                    lambda_value = scheduler.lr_lambda(self.step_count)
                    print(f"Step {self.step_count}: Lambda value = {lambda_value:.6f}")
        
        self.step_count += 1
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Monitor learning rate after scheduler step"""
        if self.step_count % self.log_every_n_steps == 0:
            # Get learning rate after scheduler step
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            print(f"Step {self.step_count}: LR after scheduler = {current_lr:.8f}")
            print("-" * 50)

class SimMIMCombLossTrainer(pl.LightningModule):
    """Pretraining on 3D Imaging with Masked Auto Encoder"""

    def __init__(
        self, 
        model_name: str, 
        model_dict: dict, 
        visualization_frequency: int = 10,
        optimizer_config: dict = None,
        scheduler_config: dict = None,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dict = model_dict
        self.optimizer_config = optimizer_config or {}
        self.scheduler_config = scheduler_config or {}
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

        self.model = ViTSimMIM(**model_dict)

        self.l1_loss = L1Loss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.recon_patches = []
        self.visualization_frequency = visualization_frequency
        self.save_hyperparameters()

    def combined_loss(self, pred_pixel_values, target_patches):
        """Calculate combined L1 + SSIM loss"""
        l1_loss = self.l1_loss(pred_pixel_values, target_patches)
        
        # Reshape patches for SSIM calculation (assuming 3D patches)
        # Get patch size from model config
        patch_size = self.model_dict.get('patch_size')
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        
        # print(f"pred_pixel_values.shape: {pred_pixel_values.shape}")
        # print(f"target_patches.shape: {target_patches.shape}")
        
        # Reshape to 3D patches for SSIM
        patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
        pred_3d = pred_pixel_values.view(pred_pixel_values.shape[0], pred_pixel_values.shape[1], patch_size[0], patch_size[1], patch_size[2])
        # print(f"pred_3d.shape: {pred_3d.shape}")
        target_3d = target_patches.view(target_patches.shape[0], target_patches.shape[1], patch_size[0], patch_size[1], patch_size[2])
        # print(f"target_3d.shape: {target_3d.shape}")
        
        # Calculate SSIM for 3D data by computing 2D SSIM for each slice and averaging
        # This is more stable than trying to compute 3D SSIM directly
        batch_size, num_patches = pred_3d.shape[:2]
        ssim_scores = []
        
        for b in range(batch_size):
            for p in range(num_patches):
                # Take middle slice for SSIM calculation (more stable than averaging all slices)
                mid_slice = patch_size[0] // 2
                pred_slice = pred_3d[b, p, mid_slice, :, :].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                target_slice = target_3d[b, p, mid_slice, :, :].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                
                # Add numerical stability checks
                if torch.isnan(pred_slice).any() or torch.isnan(target_slice).any():
                    continue
                if torch.isinf(pred_slice).any() or torch.isinf(target_slice).any():
                    continue
                
                try:
                    ssim_score = self.ssim_metric(pred_slice, target_slice)
                    if not torch.isnan(ssim_score) and not torch.isinf(ssim_score):
                        ssim_scores.append(ssim_score)
                except:
                    continue
        
        # Calculate average SSIM score
        if len(ssim_scores) > 0:
            ssim_score = torch.stack(ssim_scores).mean()
            ssim_loss_val = 1.0 - ssim_score  # Convert to loss (lower is better)
        else:
            # Fallback to L1 loss only if SSIM calculation fails
            ssim_loss_val = torch.tensor(0.0, device=l1_loss.device, dtype=l1_loss.dtype)
        
        # Add numerical stability check for final loss
        if torch.isnan(ssim_loss_val) or torch.isinf(ssim_loss_val):
            ssim_loss_val = torch.tensor(0.0, device=l1_loss.device, dtype=l1_loss.dtype)
        
        total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss_val
        
        # Final safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Fallback to L1 loss only
            total_loss = l1_loss
            ssim_loss_val = torch.tensor(0.0, device=l1_loss.device, dtype=l1_loss.dtype)
        
        return total_loss, l1_loss, ssim_loss_val

    def visualize_reconstruction(self, image, pred_pixel_values, patches, batch_range, masked_indices, batch_idx):
        """Visualize original vs reconstructed image for one example"""
        if batch_idx % self.visualization_frequency != 0:
            return
            
        # Take first example from batch
        sample_idx = 0
        
        # Get original image and patches for this sample
        orig_image = image[sample_idx]  # Shape: (C, D, H, W)
        orig_patches = patches[sample_idx]  # Shape: (num_patches, patch_dim)
        pred_patches = pred_pixel_values[batch_range[sample_idx], :]  # Shape: (num_masked, patch_dim)
        masked_idx = masked_indices[sample_idx]  # Shape: (num_masked,)
        
        # Create reconstructed patches by copying original and replacing masked ones
        recon_patches = orig_patches.clone()
        # Ensure both tensors have the same dtype
        pred_patches = pred_patches.to(dtype=recon_patches.dtype)
        recon_patches[masked_idx] = pred_patches
        print(f"recon_patches[masked_idx[0]]: {recon_patches[masked_idx[0]]}")
        print(f"recon_patches[masked_idx[1]]: {recon_patches[masked_idx[1]]}")
        
        # Reshape patches back to image
        # We need to get the patch dimensions from the model
        patch_size = self.model_dict.get('patch_size')
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        
        # Calculate spatial dimensions
        img_size = self.model_dict.get('img_size')
        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)
        
        # Calculate number of patches per dimension
        num_patches_d = img_size[0] // patch_size[0]
        num_patches_h = img_size[1] // patch_size[1] 
        num_patches_w = img_size[2] // patch_size[2]
        
        # Reshape patches to spatial layout
        recon_patches_reshaped = recon_patches.view(num_patches_d, num_patches_h, num_patches_w, -1)
        
        # Reshape to image dimensions
        patch_dim = recon_patches_reshaped.shape[-1]
        patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
        
        # Reshape each patch to its original 3D shape
        recon_patches_reshaped = recon_patches_reshaped.view(
            num_patches_d, num_patches_h, num_patches_w, 
            patch_size[0], patch_size[1], patch_size[2]
        )
        
        # Rearrange to get the full image
        recon_image = rearrange(
            recon_patches_reshaped, 
            'pd ph pw d h w -> (pd d) (ph h) (pw w)'
        )
        
        # Add channel dimension back
        recon_image = recon_image.unsqueeze(0)  # (1, D, H, W)
        
        # Print shape information for sanity check
        # print("\nShape information for visualization:")
        # print(f"Original image shape: {orig_image.shape}")
        # print(f"Original patches shape: {orig_patches.shape}")
        # print(f"Predicted patches shape: {pred_patches.shape}")
        # print(f"Masked indices shape: {masked_idx.shape}")
        # print(f"Reconstructed patches shape: {recon_patches.shape}")
        # print(f"Reconstructed image shape: {recon_image.shape}")
        # print(f"Patch size: {patch_size}")
        # print(f"Image size: {img_size}")
        # print(f"Number of patches (D,H,W): ({num_patches_d}, {num_patches_h}, {num_patches_w})")
        
        # Visualize middle slice of each dimension
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image - middle slices
        mid_d = orig_image.shape[1] // 2
        mid_h = orig_image.shape[2] // 2  
        mid_w = orig_image.shape[3] // 2
        
        axes[0, 0].imshow(orig_image[0, mid_d, :, :].cpu().numpy(), cmap='gray')
        axes[0, 0].set_title('Original - D slice')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(orig_image[0, :, mid_h, :].cpu().numpy(), cmap='gray')
        axes[0, 1].set_title('Original - H slice')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(orig_image[0, :, :, mid_w].cpu().numpy(), cmap='gray')
        axes[0, 2].set_title('Original - W slice')
        axes[0, 2].axis('off')
        
        # Reconstructed image - middle slices
        axes[1, 0].imshow(recon_image[0, mid_d, :, :].cpu().numpy(), cmap='gray')
        axes[1, 0].set_title('Reconstructed - D slice')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(recon_image[0, :, mid_h, :].cpu().numpy(), cmap='gray')
        axes[1, 1].set_title('Reconstructed - H slice')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(recon_image[0, :, :, mid_w].cpu().numpy(), cmap='gray')
        axes[1, 2].set_title('Reconstructed - W slice')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save visualization locally in lightning logs folder
        if hasattr(self, 'trainer') and self.trainer is not None:
            # Use Lightning's default log directory
            log_dir = self.trainer.log_dir
            if log_dir:
                # Create visualization subdirectory
                vis_dir = os.path.join(log_dir, 'reconstruction_visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                
                # Save the figure
                # fig_path = os.path.join(vis_dir, f'epoch_{self.trainer.current_epoch}_batch_{batch_idx}.png')
                # Alternative: include global step for more granular naming
                fig_path = os.path.join(vis_dir, f'epoch_{self.trainer.current_epoch}_step_{self.global_step}_batch_{batch_idx}.png')
                fig.savefig(fig_path, dpi=100, bbox_inches='tight')
                print(f"Saved reconstruction visualization to: {fig_path}")
        
        plt.close(fig)

    def training_step(self, batch, batch_idx):
        # --------------------------
        # Debug: Print learning rate every 10 steps
        if self.global_step % 10 == 0:
            current_lr = self.optimizers().param_groups[0]['lr']
            print(f"Training Step {self.global_step}: LR = {current_lr:.8f}")
        
        image = batch["image"]
        pred_pixel_values, patches, batch_range, masked_indices = self.model(image)
        batch_size = pred_pixel_values.shape[0]
        
        # Use combined loss
        total_loss, l1_loss, ssim_loss_val = self.combined_loss(
            pred_pixel_values, patches[batch_range, masked_indices]
        )

        # Debug: Check for NaN values every 50 steps
        if self.global_step % 50 == 0:
            print(f"Step {self.global_step} - L1: {l1_loss.item():.6f}, SSIM: {ssim_loss_val.item():.6f}, Total: {total_loss.item():.6f}")
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"WARNING: NaN/Inf detected at step {self.global_step}!")
                print(f"L1 loss: {l1_loss.item()}, SSIM loss: {ssim_loss_val.item()}")
                print(f"Pred values range: [{pred_pixel_values.min().item():.6f}, {pred_pixel_values.max().item():.6f}]")
                print(f"Target values range: [{patches[batch_range, masked_indices].min().item():.6f}, {patches[batch_range, masked_indices].max().item():.6f}]")

        # Log individual and combined losses
        self.log("train/total_loss", total_loss, batch_size=batch_size, sync_dist=True)
        self.log("train/l1_loss", l1_loss, batch_size=batch_size, sync_dist=True)
        self.log("train/ssim_loss", ssim_loss_val, batch_size=batch_size, sync_dist=True)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        # --------------------------
        image = batch["image"]
        pred_pixel_values, patches, batch_range, masked_indices = self.model(image)
        batch_size = pred_pixel_values.shape[0]
        
        # Use combined loss
        total_loss, l1_loss, ssim_loss_val = self.combined_loss(
            pred_pixel_values, patches[batch_range, masked_indices]
        )

        # Debug: Check for NaN values in validation
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"WARNING: NaN/Inf detected in validation at batch {batch_idx}!")
            print(f"L1 loss: {l1_loss.item()}, SSIM loss: {ssim_loss_val.item()}")

        # Log individual and combined losses
        self.log("val/total_loss", total_loss, batch_size=batch_size, sync_dist=True)
        self.log("val/l1_loss", l1_loss, batch_size=batch_size, sync_dist=True)
        self.log("val/ssim_loss", ssim_loss_val, batch_size=batch_size, sync_dist=True)
        
        # Visualize reconstruction
        self.visualize_reconstruction(image, pred_pixel_values, patches, batch_range, masked_indices, batch_idx)
        
        return {"loss": total_loss}

    def on_validation_epoch_end(self):
        # In newer PyTorch Lightning versions, we don't have direct access to validation outputs
        # The validation loss is already logged in validation_step, so we can just log hyperparams
        self.logger.log_hyperparams(
            params={
                "model": self.model_name,
                **self.model_dict,
                # "data": self.trainer.datamodule.json_path,
                # "ds_ratio": self.trainer.datamodule.downsample_ratio,
                "batch_size": self.trainer.datamodule.batch_size,
                "distribution": self.trainer.datamodule.dist,
                # "benchmark": self.trainer.benchmark,
                "max_epochs": self.trainer.max_epochs,
                "precision": self.trainer.precision,
                "l1_weight": self.l1_weight,
                "ssim_weight": self.ssim_weight,
            },
            # metrics={"total_loss": 0.0, "l1_loss": 0.0, "ssim_loss": 0.0},  # We'll use a placeholder since we can't access outputs directly
        )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler based on YAML configuration"""
        # Dynamically import and create optimizer
        optimizer_class_path = self.optimizer_config.get('class_path', 'torch.optim.AdamW')
        optimizer_init_args = self.optimizer_config.get('init_args', {})
        
        # Import optimizer class
        if optimizer_class_path == 'torch.optim.AdamW':
            optimizer_class = AdamW
        else:
            # Handle other optimizers by importing them
            module_path, class_name = optimizer_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            optimizer_class = getattr(module, class_name)
        
        # Create optimizer
        optimizer = optimizer_class(
            self.parameters(),
            **optimizer_init_args
        )
        
        # Dynamically import and create scheduler
        scheduler_class_path = self.scheduler_config.get('class_path', 'optimizers.lr_scheduler.WarmupCosineSchedule')
        scheduler_init_args = self.scheduler_config.get('init_args', {})
        
        # Import scheduler class
        if scheduler_class_path == 'optimizers.lr_scheduler.WarmupCosineSchedule':
            scheduler_class = WarmupCosineSchedule
        else:
            # Handle other schedulers by importing them
            module_path, class_name = scheduler_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            scheduler_class = getattr(module, class_name)
        
        # Create scheduler
        scheduler = scheduler_class(
            optimizer=optimizer,
            **scheduler_init_args
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step
                "frequency": 1,  # Update every step
            },
        }


if __name__ == "__main__":
    cli = LightningCLI(save_config_kwargs={"overwrite": True})
