import networkx as nx
import torch
import torch_geometric
from toponetx.classes import SimplicialComplex

from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting


class SimplicialLineLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to a simplicial complex domain by considering line simplicial complex.

    Line simplicial complex is a clique complex of the line graph. Line graph is a graph, in which
    the vertices are the edges in the initial graph, and two vertices are adjacent if the corresponding
    edges are adjacent in the initial graph.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to simplicial domain via line simplicial complex construction.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """

        graph = self._generate_graph_from_data(data)
        line_graph = nx.line_graph(graph)

        cliques = nx.find_cliques(line_graph)
        simplices = list(map(lambda x: set(x), cliques))

        simplicial_complex = SimplicialComplex(simplices=simplices)
        self.complex_dim = simplicial_complex.dim

        node_features = {
            node: (
                (data.x[next(iter(node))[0], :] + data.x[next(iter(node))[1], :]) / 2
            ).item()
            for node in list(simplicial_complex.nodes)
        }

        simplicial_complex.set_simplex_attributes(node_features, name="features")

        return self._get_lifted_topology(simplicial_complex, line_graph)
