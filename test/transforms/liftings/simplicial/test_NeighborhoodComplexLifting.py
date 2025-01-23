"""Test the message passing module."""

import torch
import torch_geometric

from modules.transforms.liftings.graph2simplicial.neighborhood_lifting import (
    NeighborhoodComplexLifting,
)


def create_test_graph():
    num_nodes = 5
    x = [1] * num_nodes
    edge_index = [[0, 0, 1, 1, 2, 2, 3], [1, 4, 2, 3, 3, 4, 4]]
    y = [0, 0, 1, 1, 0]

    return torch_geometric.data.Data(
        x=torch.tensor(x).float().reshape(-1, 1),
        edge_index=torch.Tensor(edge_index),
        num_nodes=num_nodes,
        y=torch.tensor(y),
    )


class TestSimplicialCliqueLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = create_test_graph()  # load_manual_graph()

        # Initialise the SimplicialCliqueLifting class
        self.lifting_signed = NeighborhoodComplexLifting(signed=True)
        self.lifting_unsigned = NeighborhoodComplexLifting(signed=False)

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())
        print(lifted_data_signed)

        expected_incidence_1 = torch.tensor(
            [
                [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            ]
        )

        assert (
            abs(expected_incidence_1) == lifted_data_unsigned.incidence_1.to_dense()
        ).all(), f"Something is wrong with unsigned incidence_1 (nodes to edges).\n{abs(expected_incidence_1) - lifted_data_unsigned.incidence_1.to_dense()}"
        assert (
            expected_incidence_1 == lifted_data_signed.incidence_1.to_dense()
        ).all(), "Something is wrong with signed incidence_1 (nodes to edges)."

        expected_incidence_2 = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        assert (
            abs(expected_incidence_2) == lifted_data_unsigned.incidence_2.to_dense()
        ).all(), f"Something is wrong with unsigned incidence_2 (edges to triangles).\n{abs(expected_incidence_2) - lifted_data_unsigned.incidence_2.to_dense()}"
        assert (
            expected_incidence_2 == lifted_data_signed.incidence_2.to_dense()
        ).all(), f"Something is wrong with signed incidence_2 (edges to triangles).\n{lifted_data_signed.incidence_2.to_dense()}"
