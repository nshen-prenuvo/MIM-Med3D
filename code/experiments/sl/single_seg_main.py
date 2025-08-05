from typing import Union, Optional, Sequence

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from models import UNETR
# from models import UperNetSwin, UperNetVAN
# from monai.networks.nets import SegResNet
from monai.data import decollate_batch

import numpy as np
import torch
import data
import optimizers
import os
import nibabel as nib

# import mlflow
# import pytorch_lightning as pl

# from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import MLFlowLogger
# from pytorch_lightning.utilities.cli import LightningCLI

import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI

class SingleSegtrainer(pl.LightningModule):
    def __init__(self, num_classes: int, model_name: str, model_dict: dict, save_examples: int = 3,
                 gradual_unfreeze: bool = True, 
                 progressive_unfreezing_schedule: list = [50, 100, 150],
                 backbone_lr: float = 1e-5, backbone_weight_decay: float = 1e-5,
                 decoder_lr: float = 1e-4, decoder_weight_decay: float = 1e-4,
                 scheduler_T_0: int = 10, scheduler_T_mult: int = 2, scheduler_eta_min: float = 1e-6,
                 # Loss function parameters
                 lambda_dice: float = 0.5, lambda_ce: float = 0.5, class_weights: list = [0.1, 0.9],
                 sliding_window_overlap: float = 0.5):
        super().__init__()
        self.model_name = model_name
        self.model_dict = model_dict
        self.save_examples = save_examples
        self.gradual_unfreeze = gradual_unfreeze
        self.progressive_unfreezing_schedule = progressive_unfreezing_schedule
        
        # Learning rate and weight decay parameters
        self.backbone_lr = backbone_lr
        self.backbone_weight_decay = backbone_weight_decay
        self.decoder_lr = decoder_lr
        self.decoder_weight_decay = decoder_weight_decay
        
        # Scheduler parameters
        self.scheduler_T_0 = scheduler_T_0
        self.scheduler_T_mult = scheduler_T_mult
        self.scheduler_eta_min = scheduler_eta_min
        
        # Loss function parameters
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.class_weights = class_weights
        
        self.sliding_window_overlap = sliding_window_overlap

        if model_name.split("_")[0] == "unetr":
            self.model = UNETR(**model_dict)
        # elif model_name == "segresnet":
        #     self.model = SegResNet(**model_dict)
        # elif model_name.startswith("upernet_swin"):
        #     self.model = UperNetSwin(**model_dict)
        # elif model_name.startswith("upernet_van"):
        #     self.model = UperNetVAN(**model_dict)

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True,
                                        lambda_dice=self.lambda_dice, lambda_ce=self.lambda_ce,
                                        weight=torch.tensor(self.class_weights))
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label = AsDiscrete(to_onehot=num_classes)
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0
        # self.dice_vals = []
        self.metric_values = []
        self.epoch_loss_values = []
        
        # For saving validation examples
        self.validation_examples = []
        self.examples_saved_count = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        batch_size = images.shape[0]
        output = self.forward(images)
        
        # # Dynamic loss weight adjustment - gradually increase dice weight
        # if hasattr(self.trainer, 'max_epochs') and self.trainer.max_epochs > 0:
        #     epoch_ratio = self.current_epoch / self.trainer.max_epochs
        #     # Start with more CE weight, gradually increase dice weight
        #     lambda_dice = 0.5 + 0.3 * epoch_ratio  # 0.5 → 0.8
        #     lambda_ce = 0.5 - 0.3 * epoch_ratio    # 0.5 → 0.2
            
        #     # Update loss function weights
        #     self.loss_function.lambda_dice = lambda_dice
        #     self.loss_function.lambda_ce = lambda_ce
            
        #     # Log weight changes occasionally
        #     if batch_idx == 0:  # Log once per epoch
        #         print(f"Epoch {self.current_epoch}: lambda_dice={lambda_dice:.3f}, lambda_ce={lambda_ce:.3f}")
        
        loss = self.loss_function(output, labels)
        # logging
        self.log(
            "train/dice_loss_step",
            loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        # Log learning rates for monitoring
        if batch_idx == 0:  # Log once per epoch
            optimizers = self.optimizers()
            # Handle case where optimizers() returns a list (when scheduler is present)
            if isinstance(optimizers, list):
                optimizer = optimizers[0]  # Get the first (and only) optimizer
            else:
                optimizer = optimizers
                
            if hasattr(optimizer, 'param_groups'):
                for i, param_group in enumerate(optimizer.param_groups):
                    lr_name = f"train/lr_group_{i}" 
                    if i == 0:
                        lr_name = "train/lr_backbone"
                    elif i == 1:
                        lr_name = "train/lr_decoder"
                    self.log(lr_name, param_group['lr'], on_step=False, on_epoch=True, logger=True)

        return {"loss": loss}

    def on_train_epoch_start(self):
        """Implement gradual unfreezing strategy"""
        # Update data module epoch for progressive augmentation
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'set_epoch'):
            self.trainer.datamodule.set_epoch(self.current_epoch)
            
        if not self.gradual_unfreeze:
            return
            
        if self.current_epoch == 0:
            # Freeze entire ViT backbone initially
            print("\nEpoch 0: Freezing entire ViT backbone")
            for name, param in self.model.named_parameters():
                if 'vit' in name:
                    param.requires_grad = False
                    
            # Count and print trainable vs total parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"\nEpoch {self.current_epoch}: Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
                    
        elif self.current_epoch == self.progressive_unfreezing_schedule[0]:
            # Unfreeze top ViT layers first (last 2 blocks)
            print(f"\nEpoch {self.current_epoch}: Unfreezing top ViT layers")
            for name, param in self.model.named_parameters():
                if 'vit' in name and ('blocks.6' in name or 'blocks.7' in name):  # Top layers
                    param.requires_grad = True
                    
            # Count and print trainable vs total parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"\nEpoch {self.current_epoch}: Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
                    
        elif self.current_epoch == self.progressive_unfreezing_schedule[1]:
            # Unfreeze middle ViT layers (blocks 6-9)
            print(f"\nEpoch {self.current_epoch}: Unfreezing middle ViT layers")
            for name, param in self.model.named_parameters():
                if 'vit' in name and any(f'blocks.{i}' in name for i in [3,4,5]):
                    param.requires_grad = True
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"\nEpoch {self.current_epoch}: Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
                                
        elif self.current_epoch == self.progressive_unfreezing_schedule[2]:
            # Unfreeze all remaining ViT layers
            print(f"\nEpoch {self.current_epoch}: Unfreezing all ViT layers")
            for name, param in self.model.named_parameters():
                if 'vit' in name:
                    param.requires_grad = True
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"\nEpoch {self.current_epoch}: Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
            
                    
        # Log current frozen/unfrozen status
        vit_frozen_params = sum(1 for name, p in self.model.named_parameters() 
                               if 'vit' in name and not p.requires_grad)
        vit_total_params = sum(1 for name, p in self.model.named_parameters() if 'vit' in name)
        total_frozen_params = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total_params = sum(1 for p in self.model.parameters())
        
        print(f"\nEpoch {self.current_epoch}: ViT {vit_frozen_params}/{vit_total_params} frozen, "
              f"Total {total_frozen_params}/{total_params} frozen")

    def on_train_epoch_end(self, outputs=None):
        if outputs is None:
            return
            
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "train/dice_loss_avg",
            avg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def on_validation_start(self):
        """Reset examples collection at the start of each validation epoch"""
        self.validation_examples = []
        self.examples_saved_count = 0

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        # print(f"images.shape: {images.shape}")
        # print(f"labels.shape: {labels.shape}")
        batch_size = images.shape[0]
        roi_size = self.trainer.datamodule.spatial_size
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward,  # the output image will be cropped to the original image size
            overlap=self.sliding_window_overlap
        )
        # print(f"outputs.shape: {outputs.shape}")
        loss = self.loss_function(outputs, labels)
        
        # Convert outputs to predicted masks for saving
        pred_masks = [self.post_pred(i) for i in decollate_batch(outputs)]
        
        # collect examples for saving (only save first few examples)
        if self.examples_saved_count < self.save_examples:
            for i in range(batch_size):
                if self.examples_saved_count >= self.save_examples:
                    break
                    
                # Store original image, true label, and predicted mask
                example = {
                    'image': images[i].detach().cpu().numpy(),
                    'true_label': labels[i].detach().cpu().numpy(), 
                    'pred_mask': pred_masks[i].detach().cpu().numpy(),
                    'batch_idx': batch_idx,
                    'sample_idx': i
                }
                self.validation_examples.append(example)
                self.examples_saved_count += 1
        
        # compute dice score
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        dice = self.dice_metric.aggregate().item()
        # self.dice_metric.reset()
        # compute mean dice score per validation epoch
        # self.dice_vals.append(dice)
        # logging
        self.log(
            "val/dice_loss_step",
            loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"val_loss": loss, "val_number": len(outputs), "dice": dice}

    def save_validation_examples(self):
        """Save validation examples as .nii files"""
        if not self.validation_examples:
            return
            
        # Create output directory in Lightning logs folder
        save_dir = os.path.join(self.trainer.log_dir, "validation_examples", f"epoch_{self.current_epoch:03d}")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nSaving {len(self.validation_examples)} validation examples to {save_dir}")
        
        for idx, example in enumerate(self.validation_examples):
            # Extract data
            image = example['image']  # Shape: (C, H, W, D)
            true_label = example['true_label']  # Shape: (C, H, W, D) 
            pred_mask = example['pred_mask']  # Shape: (C, H, W, D)
            
            # For medical images, we typically want to save as (H, W, D) and take the first channel
            # or handle multi-channel appropriately
            if image.ndim == 4:  # (C, H, W, D)
                image_3d = image[0]  # Take first channel
            else:
                image_3d = image
                
            if true_label.ndim == 4:  # (C, H, W, D) - for multi-class, take argmax
                if true_label.shape[0] > 1:
                    true_label_3d = np.argmax(true_label, axis=0)
                else:
                    true_label_3d = true_label[0]
            else:
                true_label_3d = true_label
                
            if pred_mask.ndim == 4:  # (C, H, W, D) - for multi-class, take argmax  
                if pred_mask.shape[0] > 1:
                    pred_mask_3d = np.argmax(pred_mask, axis=0)
                else:
                    pred_mask_3d = pred_mask[0]
            else:
                pred_mask_3d = pred_mask
            
            # Create filenames
            base_name = f"example_{idx:02d}_batch_{example['batch_idx']:02d}_sample_{example['sample_idx']:02d}"
            
            # Save as .nii files
            try:
                # Original image
                img_nii = nib.Nifti1Image(image_3d.astype(np.float32), affine=np.eye(4))
                nib.save(img_nii, os.path.join(save_dir, f"{base_name}_image.nii.gz"))
                
                # True label
                label_nii = nib.Nifti1Image(true_label_3d.astype(np.int16), affine=np.eye(4))
                nib.save(label_nii, os.path.join(save_dir, f"{base_name}_true_label.nii.gz"))
                
                # Predicted mask
                pred_nii = nib.Nifti1Image(pred_mask_3d.astype(np.int16), affine=np.eye(4))
                nib.save(pred_nii, os.path.join(save_dir, f"{base_name}_pred_mask.nii.gz"))
                
                print(f"Saved example {idx}: {base_name}")
                
            except Exception as e:
                print(f"Error saving example {idx}: {e}")
                continue

    def on_validation_epoch_end(self, outputs=None):
        # Always compute and log validation metrics
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        
        # Compute average validation loss if outputs are available
        if outputs is not None:
            val_loss, num_items = 0, 0
            for output in outputs:
                val_loss += output["val_loss"].sum().item()
                num_items += output["val_number"]
            mean_val_loss = torch.tensor(val_loss / num_items)
        else:
            # Use a default value or compute from other sources
            mean_val_loss = torch.tensor(0.0)  # Placeholder
            
        # Save validation examples as .nii files
        self.save_validation_examples()
            
        # logging
        self.log(
            "val/dice_loss_avg",
            mean_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val/dice_score_avg",
            mean_val_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.logger.log_hyperparams(
            params={
                "model": self.model_name,
                **self.model_dict,
                "data": self.trainer.datamodule.val_json_path,
                # "ds_ratio": self.trainer.datamodule.downsample_ratio,
                "batch_size": self.trainer.datamodule.batch_size,
                "distribution": self.trainer.datamodule.dist,
                # "benchmark": self.trainer.benchmark,
                "max_epochs": self.trainer.max_epochs,
                "precision": self.trainer.precision,
            },
            metrics={"dice_loss": mean_val_loss, "dice_score": mean_val_dice},
        )

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.metric_values.append(mean_val_dice)
        
        # Monitor fine-tuning progress
        self.monitor_finetuning_progress()

    def monitor_finetuning_progress(self):
        """Monitor fine-tuning progress and parameter statistics"""
        # Count trainable parameters by component
        vit_trainable = sum(p.numel() for name, p in self.model.named_parameters() 
                           if 'vit' in name and p.requires_grad)
        vit_total = sum(p.numel() for name, p in self.model.named_parameters() if 'vit' in name)
        
        decoder_trainable = sum(p.numel() for name, p in self.model.named_parameters() 
                               if 'vit' not in name and p.requires_grad)
        decoder_total = sum(p.numel() for name, p in self.model.named_parameters() if 'vit' not in name)
        
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print("\n" + "="*60)
        print(f"FINE-TUNING PROGRESS - Epoch {self.current_epoch}")
        print("="*60)
        print(f"ViT Backbone:  {vit_trainable:,} / {vit_total:,} trainable ({vit_trainable/vit_total*100:.1f}%)")
        print(f"Decoder:       {decoder_trainable:,} / {decoder_total:,} trainable ({decoder_trainable/decoder_total*100:.1f}%)")
        print(f"Total:         {total_trainable:,} / {total_params:,} trainable ({total_trainable/total_params*100:.1f}%)")
        print(f"Best Dice:     {self.best_val_dice:.4f} (epoch {self.best_val_epoch})")
        
        # Log current loss weights
        print(f"Loss weights:  λ_dice={self.loss_function.lambda_dice:.3f}, λ_ce={self.loss_function.lambda_ce:.3f}")
        print("="*60 + "\n")

    def configure_optimizers(self):
        """Configure optimizer with differential learning rates for backbone vs decoder"""
        # Separate parameters for backbone (ViT) vs decoder components
        backbone_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            if 'vit' in name:  # ViT backbone parameters
                backbone_params.append(param)
            else:  # Encoder/decoder parameters
                decoder_params.append(param)
        
        print(f"Backbone parameters: {sum(p.numel() for p in backbone_params):,}")
        print(f"Decoder parameters: {sum(p.numel() for p in decoder_params):,}")
        
        # Use significantly lower learning rate for pretrained backbone
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.backbone_lr, 'weight_decay': self.backbone_weight_decay},  # Lower LR for pretrained
            {'params': decoder_params, 'lr': self.decoder_lr, 'weight_decay': self.decoder_weight_decay}    # Higher LR for decoder
        ])
        
        # Cosine annealing scheduler (as mentioned it's already there)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=self.scheduler_T_0,  # Restart every 10 epochs
            T_mult=self.scheduler_T_mult,
            eta_min=self.scheduler_eta_min
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    # def test_step(self, batch, batch_idx):
    #     images, labels = batch["image"], batch["label"]
    #     batch_size = images.shape[0]
    #     roi_size = (96, 96, 96)
    #     sw_batch_size = 4
    #     outputs = sliding_window_inference(
    #         images,
    #         roi_size,
    #         sw_batch_size,
    #         self.forward,  # the output image will be cropped to the original image size
    #     )
    #     loss = self.loss_function(outputs, labels)
    #     # compute dice score
    #     outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
    #     labels = [self.post_label(i) for i in decollate_batch(labels)]
    #     self.dice_metric(y_pred=outputs, y=labels)
    #     # dice = self.dice_metric.aggregate().item()

    #     # return {"dice": dice}

    # def test_epoch_end(self, outputs):
    #     # dice_vals = []
    #     # for output in outputs:
    #     #     dice_vals.append(output["dice"])
    #     # mean_val_dice = np.mean(dice_vals)
    #     # mean_val_dice = self.dice_metric_test.aggregate().item()
    #     # self.dice_metric.reset()

    #     # print(f"avg dice score: {mean_val_dice} ")
    #     mean_val_dice = torch.nanmean(self.dice_metric.get_buffer(), dim=0)
    #     print(mean_val_dice)
    #     print(torch.mean(mean_val_dice))


if __name__ == "__main__":
    cli = LightningCLI(save_config_kwargs={"overwrite": True})
