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
    ToTensord,
    EnsureChannelFirstd,
    # AddChanneld,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandScaleIntensityd,
    Spacingd,
    RandShiftIntensityd,
    CropForegroundd,
    SpatialPadd,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld,
    MapTransform,
)
from monai.data.utils import pad_list_data_collate
import json

# from .utils import ConvertToMultiChannelBasedOnBratsClassesd, StackStuff, get_modalities

def load_npy(path):
    array = np.load(path)
    return array.astype(np.float32)


class changToImage(MapTransform):
    def __call__(self, data):
        d = dict(data)
        d["image"] = d[self.keys[0]]
        del d[self.keys[0]]

        return d


# @DATAMODULE_REGISTRY
class Fomo60kDataset(pl.LightningDataModule):
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

        with open(self.train_json_path, 'r') as f:
            self.train_list = json.load(f)
        with open(self.val_json_path, 'r') as f:
            self.valid_list = json.load(f)


        self.common_transform_list = [
            Lambdad(keys="image", func=load_npy),
            EnsureChannelFirstd(keys="image", channel_dim="no_channel"),  # adds channel dim
            ToTensord(keys="image")  # converts to torch.Tensor
        ]

    def train_transforms(self):
        transforms = Compose(
            self.common_transform_list
            + [
                SpatialPadd(
                    keys=["image"],
                    spatial_size=self.spatial_size,
                    mode="constant"
                ),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=self.spatial_size,
                    random_size=False,
                    num_samples=self.num_samples,
                )
            ]
        )
        return transforms

    def val_transforms(self):
        return self.train_transforms()

    def test_transforms(self):
        return self.train_transforms()

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

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_ds, batch_size=1, num_workers=self.num_workers, pin_memory=True,
    #     )
