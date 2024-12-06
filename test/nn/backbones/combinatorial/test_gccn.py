"""Unit tests for TopoTune."""

import pytest
import torch
from torch_geometric.data import Data
from test._utils.nn_module_auto_test import NNModuleAutoTest
from topobenchmark.nn.backbones.combinatorial.gccn import TopoTune, interrank_boundary_index, get_activation
from torch_geometric.nn import GCNConv
from omegaconf import OmegaConf

class MockGNN(torch.nn.Module):
    """Mock GNN module for testing purposes.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index):
        """Forward pass of the MockGNN.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        edge_index : torch.Tensor
            Edge indices.

        Returns
        -------
        torch.Tensor
            Output of the GCN layer.
        """
        return self.conv(x, edge_index)

def create_mock_complex_batch():
    """Create a mock complex batch for testing.

    Returns
    -------
    Data
        A PyTorch Geometric Data object representing a mock complex batch.
    """
    # 3 nodes, 3 edges, 1 face
    x_0 = torch.randn(3, 16)  # 3 nodes
    x_1 = torch.randn(3, 16)  # 3 edges
    x_2 = torch.randn(1, 16)  # 1 face
    
    batch = Data(x_0=x_0, x_1=x_1, x_2=x_2)

    # Incidence matrices
    incidence_1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 1, 2, 0, 2],  # node indices
                              [0, 0, 1, 1, 2, 2]]),  # edge indices
        values=torch.ones(6),
        size=(3, 3)  # (num_nodes, num_edges)
    ).coalesce()
    batch["down_incidence-1"] = incidence_1

    incidence_2 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 2],  # edge indices
                              [0, 0, 0]]),  # face index
        values=torch.ones(3),
        size=(3, 1)  # (num_edges, num_faces)
    ).coalesce()
    batch["down_incidence-2"] = incidence_2

    # Adjacency matrices (remain unchanged)
    adjacency_0 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 1, 1, 2, 2],
                              [1, 2, 0, 2, 0, 1]]),
        values=torch.ones(6),
        size=(3, 3)  # (num_nodes, num_nodes)
    ).coalesce()
    batch["up_adjacency-0"] = adjacency_0

    adjacency_1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 1, 1, 2, 2],
                              [1, 2, 0, 2, 0, 1]]),
        values=torch.ones(6),
        size=(3, 3)  # (num_edges, num_edges)
    ).coalesce()
    batch["up_adjacency-1"] = adjacency_1

    adjacency_2 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0], [0]]),
        values=torch.ones(1),
        size=(1, 1)  # (num_faces, num_faces)
    ).coalesce()
    batch["up_adjacency-2"] = adjacency_2

    cell_statistics = torch.tensor([[3, 3, 1]]) 
    batch["cell_statistics"] = cell_statistics
    return batch

class ModifiedNNModuleAutoTest(NNModuleAutoTest):
    """Modified NNModuleAutoTest class for TopoTune testing."""

    def assert_return_tensor(self, result):
        """Assert that the result contains a dictionary with tensor values.

        Parameters
        ----------
        result : Any
            The result to check.
        """
        assert any(isinstance(r, dict) and any(isinstance(v, torch.Tensor) for v in r.values()) for r in result)

    def assert_equal_output(self, module, result, result_2):
        """Assert that two outputs are equal.

        Parameters
        ----------
        module : torch.nn.Module
            The module being tested.
        result : Any
            The first result to compare.
        result_2 : Any
            The second result to compare.
        """
        assert len(result) == len(result_2)

        for i, r1 in enumerate(result):
            r2 = result_2[i]
            if isinstance(r1, dict) and isinstance(r2, dict):
                assert r1.keys() == r2.keys(), f"Dictionaries have different keys at index {i}"
                for key in r1.keys():
                    assert torch.allclose(r1[key], r2[key], atol=1e-6), f"Tensors not equal for key {key} at index {i}"
            elif isinstance(r1, torch.Tensor):
                assert torch.allclose(r1, r2, atol=1e-6), f"Tensors not equal at index {i}"
            else:
                assert r1 == r2, f"Values not equal at index {i}"

def test_topotune():
    """Test the TopoTune module using ModifiedNNModuleAutoTest."""
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)
    neighborhoods = OmegaConf.create(["up_adjacency-0", "up_adjacency-1", "down_incidence-1", "down_incidence-2"])#[[[0, 0], "adjacency"], [[1, 1], "adjacency"], [[1, 0], "boundary"], [[2, 1], "boundary"]])
    
    auto_test = ModifiedNNModuleAutoTest([
        {
            "module": TopoTune,
            "init": {
                "GNN": gnn,
                "neighborhoods": neighborhoods,
                "layers": 2,
                "use_edge_attr": False,
                "activation": "relu"
            },
            "forward": (batch,),
        }
    ])
    auto_test.run()

def test_topotune_methods():
    """Test individual methods of the TopoTune module."""
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)
    neighborhoods = OmegaConf.create(["up_adjacency-0", "down_incidence-1"])#[[[0, 0], "adjacency"], [[1, 0], "boundary"]])
    topotune = TopoTune(GNN=gnn, neighborhoods=neighborhoods, layers=2, use_edge_attr=False, activation="relu")

    # Test generate_membership_vectors
    membership = topotune.generate_membership_vectors(batch)
    assert 0 in membership and 1 in membership and 2 in membership
    assert membership[0].shape == (batch.x_0.shape[0],)
    assert membership[1].shape == (batch.x_1.shape[0],)
    assert membership[2].shape == (batch.x_2.shape[0],)

    # Test get_nbhd_cache
    nbhd_cache = topotune.get_nbhd_cache(batch)
    assert (1, 0) in nbhd_cache
    assert isinstance(nbhd_cache[(1, 0)], tuple)
    assert len(nbhd_cache[(1, 0)]) == 2

    # Test intrarank_expand
    expanded = topotune.intrarank_expand(batch, 0, "up_adjacency-0")
    assert isinstance(expanded, Data)
    assert expanded.x.shape == (3, 16)
    assert expanded.edge_index.shape[0] == 2

    # Test intrarank_gnn_forward
    output = topotune.intrarank_gnn_forward(expanded, 0, 0)
    assert output.shape == (3, 16) 

    # Test interrank_expand
    membership = topotune.generate_membership_vectors(batch)
    expanded = topotune.interrank_expand(batch, 1, 0, nbhd_cache[(1, 0)], membership)
    assert isinstance(expanded, Data)
    assert expanded.x.shape[1] == 16
    assert expanded.edge_index.shape[0] == 2

    # Test interrank_gnn_forward
    output = topotune.interrank_gnn_forward(expanded, 0, 0, 3)
    assert output.shape == (3, 16)  

    # Test aggregate_inter_nbhd
    x_out_per_route = {0: torch.randn(3, 16), 1: torch.randn(3, 16)}
    aggregated = topotune.aggregate_inter_nbhd(x_out_per_route)
    assert 0 in aggregated
    assert aggregated[0].shape == (3, 16)

def test_interrank_boundary_index():
    """Test the interrank_boundary_index function."""
    x_src = torch.randn(15, 16)
    boundary_index = [torch.randint(0, 10, (30,)), torch.randint(0, 15, (30,))]
    n_dst_nodes = 10
    
    edge_index, edge_attr = interrank_boundary_index(x_src, boundary_index, n_dst_nodes)
    
    assert edge_index.shape == (2, 30)
    assert edge_attr.shape == (30, 16)

def test_get_activation():
    """Test the get_activation function."""
    relu_func = get_activation("relu")
    assert callable(relu_func)
    
    relu_module = get_activation("relu", return_module=True)
    assert issubclass(relu_module, torch.nn.Module)
    
    with pytest.raises(NotImplementedError):
        get_activation("invalid_activation")