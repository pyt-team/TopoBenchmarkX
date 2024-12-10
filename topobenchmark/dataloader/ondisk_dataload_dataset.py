"""OnDisk Dataset class compatible with TBDataloader."""

import torch_geometric


class OnDiskDataloadDataset(torch_geometric.data.Dataset):
    """Custom OnDisk dataset to return all the values added to the dataset object.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Dataset.
    valid_indices : np.ndarray
        List of valid indices.
    """

    def __init__(self, dataset, valid_indices):
        super().__init__()
        self.dataset = dataset
        self.valid_indices = valid_indices

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self.dataset)})"

    def get(self, idx):
        """Get data object from data list.

        Parameters
        ----------
        idx : int
            Index of the data object to get.

        Returns
        -------
        tuple
            Tuple containing a list of all the values for the data and the corresponding keys.
        """
        # Wasn't working with numpy.int64
        data = self.dataset.get(idx)
        keys = list(data.keys())
        return ([data[key] for key in keys], keys)

    def len(self):
        """Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.dataset)
