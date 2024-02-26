import json
import os

import hydra
import torch_geometric

from topobenchmarkx.io.load.utils import ensure_serializable, make_hash


class Preprocessor(torch_geometric.data.InMemoryDataset):
    def __init__(self, data_dir, data_list, transforms_config):
        if isinstance(data_list, torch_geometric.data.Dataset):
            data_list = [data_list.get(idx) for idx in range(len(data_list))]
        elif isinstance(data_list, torch_geometric.data.Data):
            data_list = [data_list]
        self.data_list = data_list
        pre_transform = self.instantiate_pre_transform(data_dir, transforms_config)
        super().__init__(self.processed_data_dir, None, pre_transform)
        self.save_transform_parameters()
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return self.root

    @property
    def processed_file_names(self):
        return "data.pt"

    def instantiate_pre_transform(self, data_dir, transforms_config):
        pre_transforms_dict = hydra.utils.instantiate(transforms_config)
        pre_transforms = torch_geometric.transforms.Compose(
            list(pre_transforms_dict.values())
        )
        self.set_processed_data_dir(pre_transforms_dict, data_dir, transforms_config)
        return pre_transforms

    def set_processed_data_dir(self, pre_transforms_dict, data_dir, transforms_config):
        # Use self.transform_parameters to define unique save/load path for each transform parameters
        repo_name = "_".join(list(transforms_config.keys()))
        transforms_parameters = {
            transform_name: transform.parameters
            for transform_name, transform in pre_transforms_dict.items()
        }
        params_hash = make_hash(transforms_parameters)
        self.transforms_parameters = ensure_serializable(transforms_parameters)
        self.processed_data_dir = os.path.join(*[data_dir, repo_name, f"{params_hash}"])

    def save_transform_parameters(self):
        # Check if root/params_dict.json exists, if not, save it
        path_transform_parameters = os.path.join(
            self.processed_data_dir, "path_transform_parameters_dict.json"
        )
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transforms_parameters, f)
        else:
            # If path_transform_parameters exists, check if the transform_parameters are the same
            with open(path_transform_parameters, "r") as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != self.transforms_parameters:
                raise ValueError("Different transform parameters for the same data_dir")
            else:
                print(
                    f"Transform parameters are the same, using existing data_dir: {self.processed_data_dir}"
                )

    def process(self):
        self.data_list = [self.pre_transform(d) for d in self.data_list]

        self.data, self.slices = self.collate(self.data_list)
        self._data_list = None  # Reset cache.

        assert isinstance(self._data, torch_geometric.data.Data)
        self.save(self.data_list, self.processed_paths[0])