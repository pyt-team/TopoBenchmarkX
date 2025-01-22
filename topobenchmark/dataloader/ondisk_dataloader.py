"OnDiskTBDataloader class."

from typing import Any

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from topobenchmark.dataloader.ondisk_dataload_dataset import (
    OnDiskDataloadDataset,
)
from topobenchmark.dataloader.utils import collate_fn


class OnDiskTBDataloader(LightningDataModule):
    r"""This class takes care of returning the dataloaders for the training, validation, and test datasets.

    It also handles the collate function. The class is designed to work with the `torch` dataloaders.

    Parameters
    ----------
    dataset : OnDiskDataloadDataset
        The entire dataset.
    train_indices : np.ndarray
        The training dataset indices.
    val_indices : np.ndarray, optional
        The validation dataset indices (default: None).
    test_indices : np.ndarray, optional
        The test dataset indices (default: None).
    batch_size : int, optional
        The batch size for the dataloader (default: 1).
    num_workers : int, optional
        The number of worker processes to use for data loading (default: 0).
    pin_memory : bool, optional
        If True, the data loader will copy tensors into pinned memory before returning them (default: False).
    **kwargs : optional
        Additional arguments.

    References
    ----------
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset: OnDiskDataloadDataset,  # Base dataset that knows how to load from disk
        train_indices: np.array,
        val_indices: np.array = None,
        test_indices: np.array = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["dataset", "train_indices", "val_indices", "test_indices"],
        )

        self.dataset = dataset
        self.train_indices = train_indices
        self.batch_size = batch_size

        if val_indices is None and test_indices is None:
            # Transductive setting
            self.val_indices = train_indices
            self.test_indices = train_indices
            assert (
                self.batch_size == 1
            ), "Batch size must be 1 for transductive setting."
        else:
            self.val_indices = val_indices
            self.test_indices = test_indices

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = kwargs.get("persistent_workers", False)

    def train_dataloader(self) -> DataLoader:
        """Create and return the OnDisk train dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The train dataloader.
        """
        return DataLoader(
            dataset=OnDiskDataloadDataset(self.dataset, self.train_indices),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the OnDisk validation dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The validation dataloader.
        """
        return DataLoader(
            dataset=OnDiskDataloadDataset(self.dataset, self.val_indices),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the OnDisk test dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The test dataloader.
        """
        if self.test_indices is None:
            raise ValueError("There is no test dataloader.")
        return DataLoader(
            dataset=OnDiskDataloadDataset(self.dataset, self.test_indices),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
        )
