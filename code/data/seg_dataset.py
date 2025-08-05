import logging
from typing import Union, Sequence

import numpy as np
import lightning as pl
# from lightning.utilities.cli import DATAMODULE_REGISTRY
import torch.distributed as dist

from monai.data import (
    CacheDataset,
    Dataset,
    partition_dataset,
    DataLoader,
    PersistentDataset,
    load_decathlon_datalist,
)
from monai.transforms import (
    Compose,
    Lambdad,
    SpatialPadd,
    RandSpatialCropSamplesd,
    EnsureChannelFirstd,
    SplitDimd,
    ConcatItemsd,
    DeleteItemsd,
    SelectItemsd,
    MapTransform,
    # Essential progressive augmentation transforms
    RandFlipd,
    RandRotated,
    RandGaussianNoised,
    RandAdjustContrastd,
)
from monai.data.utils import pad_list_data_collate
import json

# from .utils import ConvertToMultiChannelBasedOnBratsClassesd, StackStuff, get_modalities

def load_npy(path):
    array = np.load(path, allow_pickle=True)
    return array.astype(np.float32)

def fill_nan_with_zero(array):
    """Fill NaN values with 0 in the array"""
    if np.isnan(array).any():
        array = np.nan_to_num(array, nan=0.0)
    return array


class changToImage(MapTransform):
    def __call__(self, data):
        d = dict(data)
        d["image"] = d[self.keys[0]]
        del d[self.keys[0]]

        return d


# @DATAMODULE_REGISTRY
class FomoSegDataset(pl.LightningDataModule):
    def __init__(
        self,
        train_json_path: str,
        val_json_path: str,
        cache_dir: str,
        # modality: str,
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 8,
        cache_num: int = 0,
        cache_rate: float = 0.0,
        spatial_size: Sequence[int] = (96, 96, 96),
        num_samples: int = 4,
        dist: bool = False,
        # Progressive augmentation epoch thresholds
        light_to_medium_epoch: int = 50,
        medium_to_full_epoch: int = 100,
    ):
        super().__init__()
        self.train_json_path = train_json_path
        self.val_json_path = val_json_path
        self.cache_dir = cache_dir
        # self.modality = modality
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.cache_num = cache_num
        self.cache_rate = cache_rate
        self.spatial_size = spatial_size
        self.num_samples = num_samples
        self.dist = dist
        self.current_epoch = 0  # Initialize current epoch
        
        # Progressive augmentation thresholds
        self.light_to_medium_epoch = light_to_medium_epoch
        self.medium_to_full_epoch = medium_to_full_epoch

        with open(self.train_json_path, 'r') as f:
            self.train_list = json.load(f)
        with open(self.val_json_path, 'r') as f:
            self.valid_list = json.load(f)


        # self.common_transform_list = Compose([
        #     Lambdad(keys="everything", func=load_npy),
        #     SplitDimd(keys="everything", dim=0, keepdim=False, output_postfixes=["DWI", "T2FLAIR", "SWI_OR_T2STAR", "LABEL_MASK"]),  # Split into separate channels
        #     ConcatItemsd(keys=['everything_DWI'], name='image'),  # First channel becomes image
        #     ConcatItemsd(keys=['everything_LABEL_MASK'], name='label'),  # Last channel becomes label
        #     SelectItemsd(keys=['image', 'label']),  # Keep only image and label
        #     EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),  # Ensure channels are first dimension
        # ])
        self.common_transform_list = Compose([
            Lambdad(keys=['image', 'label'], func=load_npy),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
        ])

    def train_transforms(self):
        # Extract transforms from the common_transform_list Compose object and convert to list
        common_transforms = list(self.common_transform_list.transforms)
        
        # Get progressive augmentations based on current epoch
        stage = self.get_augmentation_stage()
        if stage == "light":
            progressive_augmentations = self.get_light_augmentations()
        elif stage == "medium":
            progressive_augmentations = self.get_medium_augmentations()
        else:  # full
            progressive_augmentations = self.get_full_augmentations()
        
        # Combine with additional training transforms
        all_transforms = common_transforms + progressive_augmentations + [
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=self.spatial_size,
                mode="constant"
            ),
            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=self.spatial_size,
                random_size=False,
                num_samples=self.num_samples,
            )
        ]
        
        transforms = Compose(all_transforms)
        return transforms

    def val_transforms(self):
        return self.common_transform_list

    def test_transforms(self):
        return self.common_transform_list

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            if self.dist and dist.is_initialized():
                train_partition = partition_dataset(
                    data=self.train_list,
                    num_partitions=dist.get_world_size(),
                    shuffle=True,
                    even_divisible=True,
                )[dist.get_rank()]

                valid_partition = partition_dataset(
                    data=self.valid_list,
                    num_partitions=dist.get_world_size(),
                    shuffle=False,
                    even_divisible=True,
                )[dist.get_rank()]
            else:
                train_partition = self.train_list
                valid_partition = self.valid_list

            if any([self.cache_num, self.cache_rate]) > 0:
                self.train_ds = CacheDataset(
                    train_partition,
                    cache_num=self.cache_num,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                    transform=self.train_transforms(),
                )
                self.valid_ds = CacheDataset(
                    valid_partition,
                    cache_num=self.cache_num,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                    transform=self.val_transforms(),
                )
            else:
                logging.info("Loading Persistent Dataset...")
                self.train_ds = PersistentDataset(
                    train_partition,
                    transform=self.train_transforms(),
                    cache_dir=self.cache_dir,
                )
                self.valid_ds = PersistentDataset(
                    valid_partition,
                    transform=self.val_transforms(),
                    cache_dir=self.cache_dir,
                )

        # self.test_ds = Dataset(self.test_dict, transform=self.test_transforms())

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
            shuffle=False,
            drop_last=False,
            # prefetch_factor=4,
        )

    def set_epoch(self, epoch: int):
        """Called by trainer to update current epoch for progressive augmentation"""
        old_stage = self.get_augmentation_stage()
        self.current_epoch = epoch
        new_stage = self.get_augmentation_stage()
        
        if old_stage != new_stage:
            print(f"\nDataModule: Epoch {epoch} - Switching from {old_stage.upper()} to {new_stage.upper()} augmentations")
    
    def get_augmentation_stage(self) -> str:
        """Return current augmentation stage based on epoch"""
        if self.current_epoch < self.light_to_medium_epoch:
            return "light"
        elif self.current_epoch < self.medium_to_full_epoch:
            return "medium"
        else:
            return "full"

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_ds, batch_size=1, num_workers=self.num_workers, pin_memory=True,
    #     )

    def get_light_augmentations(self):
        """Light augmentations for early training (epochs 0-{})""".format(self.light_to_medium_epoch)
        return [
            # Basic geometric transforms only
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotated(keys=["image", "label"], prob=0.3, range_x=0.1, range_y=0.1, range_z=0.1),
        ]

    def get_medium_augmentations(self):
        """Medium augmentations for middle training (epochs {}-{})""".format(self.light_to_medium_epoch, self.medium_to_full_epoch)
        return [
            # Basic geometric transforms
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotated(keys=["image", "label"], prob=0.3, range_x=0.1, range_y=0.1, range_z=0.1),
            
            # Light intensity transforms
            RandGaussianNoised(keys="image", prob=0.3, std=0.1),
            RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.8, 1.2)),
        ]

    def get_full_augmentations(self):
        """Full augmentations for late training (epochs {}+)""".format(self.medium_to_full_epoch)
        return [
            # Geometric transforms
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotated(keys=["image", "label"], prob=0.3, range_x=0.1, range_y=0.1, range_z=0.1),
            
            # Intensity transforms
            RandGaussianNoised(keys="image", prob=0.3, std=0.1),
            RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.8, 1.2)),
        ]

