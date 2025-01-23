r"""This module implements the line lifting.

This lifting constructs a simplicial complex called the *Line simplicial complex*.
This is a generalization of the so-called
`Line graph <https://en.wikipedia.org/wiki/Line_graph_.
In vanilla line graph, nodes are edges from the original graph and two nodes are
connected with an edge if corresponding edges are adjacent in the original graph.

The line simplicial complex is a clique complex of the line graph.
That is, if several edges share a node in the original graph,
the corresponding vertices in the line complex are going to be connected with
a simplex.

So, the line lifting performs the following:

1. For a graph :math:`G` we create its *line graph* :math:`L(G)`.
  The *line* (or *dual*) *graph* is a graph created from the initial one
  by considering the edges in :math:`G` as the vertices in :math:`L(G)`
  and the edges in :math:`L(G)` correspond to the vertices in :math:`G`.

2. During this procedure, we obtain a graph.
  It is easy to see that such graph contains cliques for basically each
  node in initial graph :math:`G`. That is, if :math:`v \in G`
  has a degree :math:`d`, then there's a clique on :math:`d` vertices in
  math:`L(G)`.

3. Therefore let's consider a clique complex
  :math:`X(L(G))` of math:`L(G)` creating :math:`(d - 1)`-simplices for each
  clique on :math:`d` vertices, that is, for each node in math:`G` of degree
  :math:`d`.


When creating a line graph, we need to transfer the features from :math:`G` to
:math:`L(G)`. That is, for a vertex :math:`v \in L(G)`, which correponds to
an edge :math:`e \in G` we need to set a feature vector.
This is basically done as a mean feature of the nodes that are adjacent to :math:`e`,
that is, if `:math:`e = ( a , b )`, then

.. math::

    f v := f a + f b 2 .
"""

import networkx as nx
import torch
from toponetx.classes import SimplicialComplex

from topobenchmark.transforms.liftings.base import LiftingMap


class SimplicialLineLifting(LiftingMap):
    r"""Lifts graphs to a simplicial complex domain by considering line simplicial complex.

    Line simplicial complex is a clique complex of the line graph. Line graph is a graph, in which
    the vertices are the edges in the initial graph, and two vertices are adjacent if the corresponding
    edges are adjacent in the initial graph.
    """

    def lift(self, domain):
        r"""Lift the topology of a graph to a simplicial complex.

        Parameters
        ----------
        domain : nx.Graph
            Graph to be lifted.

        Returns
        -------
        toponetx.SimplicialComplex
            Lifted simplicial complex.
        """
        graph = domain
        line_graph = nx.line_graph(graph)

        node_features = {
            node: (
                (
                    torch.tensor(graph.nodes[node[0]]["x"])
                    + torch.tensor(graph.nodes[node[1]]["x"])
                )
                / 2
            )
            for node in list(line_graph.nodes)
        }

        cliques = nx.find_cliques(line_graph)
        simplices = list(cliques)

        # we need to rename simplices here since now vertices are named as pairs
        rename_vertices_dict = {
            node: i for i, node in enumerate(line_graph.nodes)
        }
        renamed_simplices = [
            {rename_vertices_dict[vertex] for vertex in simplex}
            for simplex in simplices
        ]

        renamed_node_features = {
            rename_vertices_dict[node]: value
            for node, value in node_features.items()
        }

        simplicial_complex = SimplicialComplex(simplices=renamed_simplices)
        simplicial_complex.set_simplex_attributes(
            renamed_node_features, name="x"
        )

        return simplicial_complex
