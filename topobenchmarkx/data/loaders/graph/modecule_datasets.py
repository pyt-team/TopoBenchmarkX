"""Loaders for Molecule datasets (ZINC and AQSOL)."""

import os
from pathlib import Path
from typing import ClassVar

import numpy as np
from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import AQSOL, ZINC

from topobenchmarkx.data.loaders.base import AbstractLoader


class MoleculeDatasetLoader(AbstractLoader):
    """Load molecule datasets (ZINC and AQSOL) with predefined splits.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "molecule")
    """

    VALID_DATASETS: ClassVar[set] = {"ZINC", "AQSOL"}
    VALID_TYPES: ClassVar[set] = {"ZINC", "AQSOL"}

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        self._validate_parameters()
        self.datasets: list[Dataset] = []

    def _validate_parameters(self) -> None:
        """Validate the input parameters."""
        if self.parameters.data_name not in self.VALID_DATASETS:
            raise ValueError(
                f"Dataset '{self.parameters.data_name}' not supported. "
                f"Must be one of: {', '.join(sorted(self.VALID_DATASETS))}"
            )

        if self.parameters.data_type not in self.VALID_TYPES:
            raise ValueError(
                f"Data type '{self.parameters.data_type}' not supported. "
                f"Must be one of: {', '.join(sorted(self.VALID_TYPES))}"
            )

    def load_dataset(self) -> Dataset:
        """Load the molecule dataset with predefined splits.

        Returns
        -------
        Dataset
            The combined dataset with predefined splits.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        self._load_splits()
        split_idx = self._prepare_split_idx()
        combined_dataset = self._combine_splits()
        combined_dataset.split_idx = split_idx
        return combined_dataset

    def _load_splits(self) -> None:
        """Load the dataset splits for the specified dataset."""
        for split in ["train", "val", "test"]:
            if self.parameters.data_name == "ZINC":
                self.datasets.append(
                    ZINC(
                        root=str(self.root_data_dir),
                        subset=True,
                        split=split,
                    )
                )
            elif self.parameters.data_name == "AQSOL":
                self.datasets.append(
                    AQSOL(
                        root=str(self.root_data_dir),
                        split=split,
                    )
                )

    def _prepare_split_idx(self) -> dict[str, np.ndarray]:
        """Prepare the split indices for the dataset.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping split names to index arrays.
        """
        split_idx = {"train": np.arange(len(self.datasets[0]))}
        split_idx["valid"] = np.arange(
            len(self.datasets[0]),
            len(self.datasets[0]) + len(self.datasets[1]),
        )
        split_idx["test"] = np.arange(
            len(self.datasets[0]) + len(self.datasets[1]),
            len(self.datasets[0])
            + len(self.datasets[1])
            + len(self.datasets[2]),
        )
        return split_idx

    def _combine_splits(self) -> Dataset:
        """Combine the dataset splits into a single dataset.

        Returns
        -------
        Dataset
            The combined dataset containing all splits.
        """
        return self.datasets[0] + self.datasets[1] + self.datasets[2]

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return os.path.join(self.root_data_dir, self.parameters.data_name)