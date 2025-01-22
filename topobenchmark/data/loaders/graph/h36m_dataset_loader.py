"""Loader for Human3.6M dataset."""

from omegaconf import DictConfig

from topobenchmark.data.datasets import H36MDataset
from topobenchmark.data.loaders.base import AbstractLoader


class H36MDatasetLoader(AbstractLoader):
    """Load Human3.6M dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_domain: Domain of data, 'graph'.
            - data_type: Type of data, 'motion'.
            - data_name: Name of the dataset, 'H36MDataset'.
            - data_dir: Root directory for the data.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> H36MDataset:
        """Load the H36M dataset.

        Returns
        -------
        H36MDataset
            The loaded H3.6M dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        # This is where I sketchily add whether or not to force reloading!
        # I can't find out how to do this in the abstract dataloader class
        # as none of the other ones seem to have this functionality.
        force_reload = True
        dataset = H36MDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
            force_reload=force_reload,
        )
        return dataset
