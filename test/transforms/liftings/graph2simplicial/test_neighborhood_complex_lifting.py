import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2simplicial.neighborhood_complex_lifting import (
    NeighborhoodComplexLifting,
)


class TestNeighborhoodComplexLifting:
    """Test the NeighborhoodComplexLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = load_manual_graph()

        # Initialize the NeighborhoodComplexLifting class for dim=3
        self.lifting_signed = NeighborhoodComplexLifting(complex_dim=3, signed=True)
        self.lifting_unsigned = NeighborhoodComplexLifting(complex_dim=3, signed=False)

        # Intialize an empty graph for testing purpouses
        self.empty_graph = nx.empty_graph(10)
        self.empty_data = from_networkx(self.empty_graph)
        self.empty_data["x"] = torch.rand((10, 10))

        # Intialize a random graph for testing purpouses
        self.random_graph = nx.fast_gnp_random_graph(10, 0.5)
        self.random_data = from_networkx(self.random_graph)
        self.random_data["x"] = torch.rand((10, 10))

    def test_lift_topology_random_graph(self):
        """ Verifies that the lifting procedure works on
        random graphs, that is, checks that the simplices
        generated share a neighbour.
        """
        lifted_data = self.lifting_unsigned.forward(self.random_data)
        # For each set of simplices
        for r in range(1, self.lifting_unsigned.complex_dim):
            idx_str = f"x_idx_{r}"

            # Found maximum dimension
            if idx_str not in lifted_data:
                break

            # Iterate over the composing nodes of each simplex
            for simplex_points in lifted_data[idx_str]:
                share_neighbour = True
                # For each node in the graph
                for node in self.random_graph.nodes:
                    # Not checking the nodes themselves
                    if node in simplex_points:
                        continue
                    share_neighbour = True
                    # For each node in the simplex
                    for simplex_point in simplex_points:
                        # If the node does not have a common neighbour
                        if not self.random_graph.has_edge(node, simplex_point):
                            share_neighbour = False
                            break
                    # There is at least 1 node that has a common neighbour
                    # with the nodes in the simplex
                    if share_neighbour:
                        break
                assert share_neighbour, f"The simplex {simplex_points} does not have a common neighbour with all the nodes."

    def test_lift_topology_empty_graph(self):
        """ Test the lift_topology method with an empty graph.
        """

        lifted_data_signed = self.lifting_signed.forward(self.empty_data)

        assert lifted_data_signed.incidence_1.shape[1] == 0, "Something is wrong with signed incidence_1 (nodes to edges)."

        assert lifted_data_signed.incidence_2.shape[1] == 0, "Something is wrong with signed incidence_2 (edges to triangles)."

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [
                [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.],
                [ 1.,  0.,  0.,  0., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.],
                [ 0.,  1.,  0.,  0.,  1.,  0., -1., -1., -1., -1., -1.,  0.,  0.,  0., 0.],
                [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0., -1.,  0.,  0., 0.],
                [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0., 0.],
                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0., -1., -1., 0.],
                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  0., -1.],
                [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., 1.]
            ]
        )
        assert (
            abs(expected_incidence_1) == lifted_data_unsigned.incidence_1.to_dense()
        ).all(), "Something is wrong with unsigned incidence_1 (nodes to edges)."
        assert (
            expected_incidence_1 == lifted_data_signed.incidence_1.to_dense()
        ).all(), "Something is wrong with signed incidence_1 (nodes to edges)."

        expected_incidence_2 = torch.tensor(
            [
                [ 0.],
                [ 0.],
                [ 0.],
                [ 0.],
                [ 0.],
                [ 0.],
                [ 0.],
                [ 0.],
                [ 0.],
                [ 1.],
                [-1.],
                [ 0.],
                [ 0.],
                [ 0.],
                [ 1.]
            ]
        )

        assert (
            abs(expected_incidence_2) == lifted_data_unsigned.incidence_2.to_dense()
        ).all(), "Something is wrong with unsigned incidence_2 (edges to triangles)."
        assert (
            expected_incidence_2 == lifted_data_signed.incidence_2.to_dense()
        ).all(), "Something is wrong with signed incidence_2 (edges to triangles)."
