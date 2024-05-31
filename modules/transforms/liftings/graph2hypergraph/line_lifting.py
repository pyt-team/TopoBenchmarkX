import torch
import torch_geometric

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting


class HypergraphLineLifting(Graph2HypergraphLifting):
    r"""Lifts graphs to hypergraph domain by considering line hypergraph.

    Line hypergraph is a hypergraph in which the vertices are the edges in the initial graph
    and a hyperedge connects the vertices in there's a vertex in the initial graph
    which is adjacent to the corresponding edges

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to hypergraph domain via line hypergraph construction.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """

        # nodes are the edges in the original graph
        num_nodes = data.edge_index.shape[1]
        # hyperedges are the vertices in the original graph
        num_hyperedges = data.x.shape[0]

        x_0 = torch.Tensor(
            [torch.mean(val, dtype=torch.float) for val in data.edge_index.T]
        ).reshape(-1, 1)

        incidence_1 = torch.zeros(num_nodes, num_hyperedges)

        for i, val in enumerate(data.edge_index.T):
            incidence_1[i, val[0]] = 1
            incidence_1[i, val[1]] = 1

        incidence_1 = incidence_1.to_sparse_coo()

        return {
            "incidence_hyperedges": incidence_1,
            # "num_nodes": num_nodes,
            "num_hyperedges": num_hyperedges,
            "x_0": x_0,
            "x_1": data.x,
        }
