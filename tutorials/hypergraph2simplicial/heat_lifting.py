# %% Imports
import numpy as np
import toponetx as tnx
import torch_geometric

from modules.data.load.loaders import GraphLoader, HypergraphLoader
from modules.data.preprocess.preprocessor import PreProcessor
from modules.transforms.liftings.hypergraph2simplicial.heat_lifting import normalize_hg
from modules.utils.utils import (
    describe_data,
    describe_hypergraph,
    describe_simplicial_complex,
    load_dataset_config,
    load_model_config,
    load_transform_config,
)

# %% Load the data set
dataset_name = "manual_hypergraph"
# dataset_name = "senate_committee" # or "contact_primary_school",
dataset_config = load_dataset_config(dataset_name)
loader = HypergraphLoader(dataset_config)
dataset = loader.load()  # stored in datasets/**

# %% Print information about it
assert dataset.incidence_hyperedges.shape == (12, 24)
describe_data(dataset)

## This doesn't work as expected, but seems to be a an issue with the describe function itself
# describe_hypergraph(dataset)

# dataset.data.incidence_hyperedges.to_dense()
# %% Configure the lifting
transform_type = "liftings"
transform_id = "hypergraph2simplicial/heat_lifting"
transform_config = {"lifting": load_transform_config(transform_type, transform_id)}

# %% Perform the lift to the simplicial target domain
lifted_dataset = PreProcessor(dataset, transform_config, loader.data_dir)

# %% Print info about it
## There's too many things wrong the describe functions to warrant getting them to work
# describe_data(lifted_dataset)
# describe_simplicial_complex(lifted_dataset)

# %% Running a Model over the Lifted Dataset
import toponetx as tnx
from topomodelx.utils.sparse import from_sparse

from modules.models.simplicial.san import SAN, SANModel

## Configure
dataset_config.num_features = 1
dataset_config.num_classes = 2

## This doesn't seem fine enough
# model_type = "simplicial"
# model_id = "san"
# model_config = load_model_config(model_type, model_id)
# model = SANModel(model_config, dataset_config)

# %% Prep inputs for a model
import torch

x_0 = lifted_dataset.x_0.T.float()
x_1 = lifted_dataset.x_1.T.float()
x = x_1 + torch.sparse.mm(lifted_dataset.incidence_1.T, x_0)
y = torch.tensor(np.random.choice([0, 1], size=dataset.num_nodes))
y_true = np.zeros((len(y), 2))
y_true[:, 0] = y
y_true[:, 1] = 1 - y
y_train = y_true[: int(len(y) * 0.80)]
y_test = y_true[int(len(y) * 0.80) :]
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

D1 = lifted_dataset.incidence_1
D2 = lifted_dataset.incidence_2
dn_laplacian = D1.T @ D1
up_laplacian = D2 @ D2.T


class Network(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=1):
        super().__init__()
        self.base_model = SAN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        )
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, laplacian_up, laplacian_down):
        x = self.base_model(x, laplacian_up, laplacian_down)
        x = self.linear(x)
        return torch.sigmoid(x)


# %%  Network architecture
model = Network(in_channels=1, hidden_channels=8, out_channels=2, n_layers=2)
test_interval = 10
num_epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch_i in range(1, num_epochs + 1):
    epoch_loss = []
    model.train()
    optimizer.zero_grad()
    y_hat_edge = model(x, up_laplacian, dn_laplacian)
    y_hat = torch.softmax(torch.sparse.mm(D1, y_hat_edge), dim=1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        y_hat[: len(y_train)].float(), y_train.float()
    )
    epoch_loss.append(loss.item())
    loss.backward()
    optimizer.step()

    y_pred = torch.where(y_hat > 0.5, torch.tensor(1), torch.tensor(0))
    accuracy = (y_pred[: len(y_train)] == y_train).all(dim=1).float().mean().item()
    print(
        f"Epoch: {epoch_i} loss: {np.mean(epoch_loss):.4f} Train_acc: {accuracy:.4f}",
        flush=True,
    )
    if epoch_i % test_interval == 0:
        with torch.no_grad():
            y_hat_edge_test = model(
                x, laplacian_up=up_laplacian, laplacian_down=dn_laplacian
            )
            # Projection to node-level
            y_hat_test = torch.softmax(torch.sparse.mm(D1, y_hat_edge_test), dim=1)
            y_pred_test = torch.where(
                y_hat_test > 0.5, torch.tensor(1), torch.tensor(0)
            )
            test_accuracy = (
                torch.eq(y_pred_test[-len(y_test) :], y_test)
                .all(dim=1)
                .float()
                .mean()
                .item()
            )
            print(f"Test_acc: {test_accuracy:.4f}", flush=True)

# %%
