"""Test the message passing module."""

import torch

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2simplicial.neighborhood_lifting import (
    NeighborhoodComplexLifting,
)


class TestSimplicialCliqueLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = load_manual_graph()
        print(self.data)

        # Initialise the SimplicialCliqueLifting class
        self.lifting_signed = NeighborhoodComplexLifting(signed=True)
        self.lifting_unsigned = NeighborhoodComplexLifting(signed=False)

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        )

        print(lifted_data_unsigned.incidence_1.to_dense())

        assert (
            abs(expected_incidence_1) == lifted_data_unsigned.incidence_1.to_dense()
        ).all(), f"Something is wrong with unsigned incidence_1 (nodes to edges).\n{abs(expected_incidence_1) - lifted_data_unsigned.incidence_1.to_dense()}"
        assert (
            expected_incidence_1 == lifted_data_signed.incidence_1.to_dense()
        ).all(), "Something is wrong with signed incidence_1 (nodes to edges)."

        expected_incidence_2 = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        assert (
            abs(expected_incidence_2) == lifted_data_unsigned.incidence_2.to_dense()
        ).all(), "Something is wrong with unsigned incidence_2 (edges to triangles)."
        assert (
            expected_incidence_2 == lifted_data_signed.incidence_2.to_dense()
        ).all(), "Something is wrong with signed incidence_2 (edges to triangles)."

        expected_incidence_3 = torch.tensor(
            [[-1.0], [1.0], [-1.0], [0.0], [1.0], [0.0]]
        )

        assert (
            abs(expected_incidence_3) == lifted_data_unsigned.incidence_3.to_dense()
        ).all(), (
            "Something is wrong with unsigned incidence_3 (triangles to tetrahedrons)."
        )
        assert (
            expected_incidence_3 == lifted_data_signed.incidence_3.to_dense()
        ).all(), (
            "Something is wrong with signed incidence_3 (triangles to tetrahedrons)."
        )
