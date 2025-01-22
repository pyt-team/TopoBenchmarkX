"""OnDisk preprocessor for datasets."""

import json
import os

import hydra
import numpy as np
import torch_geometric
from torch_geometric.data import Data, OnDiskDataset

from topobenchmark.data.utils import (
    ensure_serializable,
    load_inductive_split_indices,
    load_inductive_splits,
    load_transductive_splits,
    make_hash,
)
from topobenchmark.dataloader import DataloadDataset
from topobenchmark.transforms.data_transform import DataTransform


class OnDiskPreProcessor(OnDiskDataset):
    """Preprocessor for datasets.

    Parameters
    ----------
    dataset : OnDiskDataset
        Dataset.
    data_dir : str
        Path to the directory containing the data.
    transforms_config : DictConfig, optional
        Configuration parameters for the transforms (default: None).
    **kwargs : optional
        Optional additional arguments.
    """

    def __init__(self, dataset, data_dir, transforms_config=None, **kwargs):
        self.dataset = dataset

        # Not sure if it is working for transforms yet.
        if transforms_config is not None:
            self.transforms_applied = True
            pre_transform = self.instantiate_pre_transform(
                data_dir, transforms_config
            )
            super().__init__(
                self.processed_data_dir, None, pre_transform, **kwargs
            )
            self.save_transform_parameters()
        else:
            self.transforms_applied = False

        super().__init__(data_dir, None, None, **kwargs)

        # Some datasets have fixed splits
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx

    def __len__(self) -> int:
        """Return the number of graphs in the dataset.

        Returns
        -------
        int
            Number of graphs in the dataset.
        """
        return len(self.dataset)

    def get(self, idx: int) -> Data:
        """Load a single graph from disk. Copied from H36MDataset.

        Parameters
        ----------
        idx : int
            Index of the graph to load.

        Returns
        -------
        Data
            The loaded graph.
        """
        return self.dataset.get(idx)

    def __getitem__(self, idx: int) -> Data:
        """Load a single graph from disk so that split_utils doesn't need changing.

        Parameters
        ----------
        idx : int
            Index of the graph to load.

        Returns
        -------
        Data
            The loaded graph.
        """
        return self.get(idx)

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory. Copied from PreProcessor.

        Returns
        -------
        str
            Path to the processed directory.
        """
        if self.transforms_applied:
            return self.root
        else:
            return os.path.join(self.root, "processed")

    def process(self) -> None:
        """Method that applies transformation to the data.

        TBH no idea if this works.
        """
        if self.pre_transform is not None:
            print("Applying transform to data...")
            print("NOT IMPLEMENTED YET")
            # # Process each graph and save to new location
            # for idx in range(len(self)):
            #     # Load original graph
            #     data = self.get(idx)

            #     # Apply transform
            #     transformed_data = self.pre_transform(data)

            #     # Save transformed graph
            #     filename = f"transformed_data_{idx}.pt"
            #     filepath = os.path.join(self.processed_dir, filename)
            #     torch.save(transformed_data, filepath)

            #     # Add to database
            #     size = os.path.getsize(filepath)
            #     self.cursor.execute(
            #         "INSERT OR REPLACE INTO data(idx, file_name, size) VALUES (?, ?, ?)",
            #         (idx, filename, size),
            #     )

            # # Commit database changes
            # self.connection.commit()

        print("Done processing.")

    def instantiate_pre_transform(
        self, data_dir, transforms_config
    ) -> torch_geometric.transforms.Compose:
        """Instantiate the pre-transforms. Copied from PreProcessor.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.

        Returns
        -------
        torch_geometric.transforms.Compose
            Pre-transform object.
        """
        pre_transforms_dict = hydra.utils.instantiate(transforms_config)
        pre_transforms_dict = {
            key: DataTransform(**value)
            for key, value in transforms_config.items()
        }
        pre_transforms = torch_geometric.transforms.Compose(
            list(pre_transforms_dict.values())
        )
        self.set_processed_data_dir(
            pre_transforms_dict, data_dir, transforms_config
        )
        return pre_transforms

    def set_processed_data_dir(
        self, pre_transforms_dict, data_dir, transforms_config
    ) -> None:
        """Set the processed data directory. Copied from PreProcessor.

        Parameters
        ----------
        pre_transforms_dict : dict
            Dictionary containing the pre-transforms.
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.
        """
        repo_name = "_".join(list(transforms_config.keys()))
        transforms_parameters = {
            transform_name: transform.parameters
            for transform_name, transform in pre_transforms_dict.items()
        }
        params_hash = make_hash(transforms_parameters)
        self.transforms_parameters = ensure_serializable(transforms_parameters)
        self.processed_data_dir = os.path.join(
            *[data_dir, repo_name, f"{params_hash}"]
        )

    def save_transform_parameters(self) -> None:
        """Save the transform parameters. Copied from PreProcessor."""
        path_transform_parameters = os.path.join(
            self.processed_data_dir, "path_transform_parameters_dict.json"
        )
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transforms_parameters, f, indent=4)
        else:
            with open(path_transform_parameters) as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != self.transforms_parameters:
                raise ValueError(
                    "Different transform parameters for the same data_dir"
                )

            print(
                f"Transform parameters are the same, using existing data_dir: {self.processed_data_dir}"
            )

    def load_dataset_splits(
        self, split_params
    ) -> tuple[
        DataloadDataset, DataloadDataset | None, DataloadDataset | None
    ]:
        """Load the dataset splits InMemory. Copied from PreProcessor.

        Parameters
        ----------
        split_params : dict
            Parameters for loading the dataset splits.

        Returns
        -------
        tuple
            A tuple containing the train, validation, and test datasets.
        """
        if not split_params.get("learning_setting", False):
            raise ValueError("No learning setting specified in split_params")

        if split_params.learning_setting == "inductive":
            return load_inductive_splits(self, split_params)
        elif split_params.learning_setting == "transductive":
            return load_transductive_splits(self, split_params)
        else:
            raise ValueError(
                f"Invalid '{split_params.learning_setting}' learning setting.\
                Please define either 'inductive' or 'transductive'."
            )

    def load_dataset_split_indices(
        self, split_params
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Load the dataset splits. So things can happen OnDisk.

        Parameters
        ----------
        split_params : dict
            Parameters for loading the dataset splits.

        Returns
        -------
        tuple
            A tuple containing the train, validation, and test split indices.
        """
        if split_params.get("learning_setting") != "inductive":
            raise NotImplementedError(
                "Non-inductive splits are not yet implemented for OnDiskDatasets."
            )

        return load_inductive_split_indices(self, split_params)

    def __del__(self):
        """Close database connection when object is deleted."""
        if hasattr(self, "connection"):
            self.connection.close()
