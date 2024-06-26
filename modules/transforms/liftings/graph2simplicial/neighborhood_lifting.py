import torch
import torch_geometric
from toponetx.classes import SimplicialComplex
from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting

class NeighborhoodComplexLifting(Graph2SimplicialLifting):
    r"""Liftss graphs to simplicial complex domain by constructing the neighborhood complex[1].

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        self.contains_edge_attr = False
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to a simplicial complex by identifying the cliques as k-simplices.

        Parameters
        ----------
        data : torch_geometric.data.Dataa
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        undir_edge_index = torch_geometric.utils.to_undirected(data.edge_index)

        simplices = [
            set(
                undir_edge_index[1, j].tolist()
                for j in torch.nonzero(undir_edge_index[0] == i).squeeze()
            )
            for i in torch.unique(undir_edge_index[0])
        ]

        node_features = {i: data.x[i, :] for i in range(data.x.shape[0])}

        simplicial_complex = SimplicialComplex(simplices)
        self.complex_dim = simplicial_complex.dim
        simplicial_complex.set_simplex_attributes(node_features, name="features")

        graph = simplicial_complex.graph_skeleton()

        return self._get_lifted_topology(simplicial_complex, graph)
