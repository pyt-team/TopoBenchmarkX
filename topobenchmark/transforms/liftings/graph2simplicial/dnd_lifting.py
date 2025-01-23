"""This modules implements the DnD lifting.

The DnD lifting introduces a novel, non-deterministic, and somewhat
lighthearted approach to transforming graphs into simplicial complexes.
Inspired by the game mechanics of Dungeons & Dragons (D&D), this method
incorporates elements of randomness and character attributes to determine
the formation of simplices. This lifting aims to add an element of whimsy
and unpredictability to the graph-to-simplicial complex transformation
process, while still providing a serious and fully functional methodology.

Each vertex in the graph is assigned the following attributes: degree centrality,
clustering coefficient, closeness centrality, eigenvector centrality,
betweenness centrality, and pagerank.
Simplices are created based on the neighborhood within a distance determined by
a D20 dice roll + the attribute value. The randomness from the dice roll,
modified by the node's attributes, ensures a non-deterministic process for each
lifting. The dice roll is influenced by different attributes based on the level
of the simplex being formed. The different attributes for different levels of
simplices are used in the order shown above, based on the role of those attributes
in the context of the graph structure.
"""

import random
from itertools import combinations

import networkx as nx
import torch
from toponetx.classes import SimplicialComplex

from topobenchmark.transforms.liftings import LiftingMap


class SimplicialDnDLifting(LiftingMap):
    """Lifts graphs to simplicial complex domain

    Uses a Dungeons & Dragons inspired system.

    Parameters
    ----------
    complex_dim : int
        Dimension of the subcomplex.
    """

    def __init__(self, complex_dim=2):
        super().__init__()
        self.complex_dim = complex_dim

    def lift(self, domain):
        """Lifts the topology of a graph to a simplicial complex.

        Uses Dungeons & Dragons (D&D) inspired mechanics.

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

        simplicial_complex = SimplicialComplex()

        characters = self._assign_attributes(graph)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]

        for node in graph.nodes:
            simplicial_complex.add_node(
                node, x=torch.tensor(domain.nodes[node]["x"])
            )

        for node in graph.nodes:
            character = characters[node]
            for k in range(1, self.complex_dim):
                dice_roll = self._roll_dice(character, k)
                neighborhood = list(
                    nx.single_source_shortest_path_length(
                        graph, node, cutoff=dice_roll
                    ).keys()
                )
                for combination in combinations(neighborhood, k + 1):
                    simplices[k - 1].add(tuple(sorted(combination)))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        # because ComplexData pads unexisting dimensions with empty matrices
        simplicial_complex.practical_dim = self.complex_dim

        return simplicial_complex

    def _assign_attributes(self, graph):
        """Assign D&D-inspired attributes based on node properties."""
        degrees = nx.degree_centrality(graph)
        clustering = nx.clustering(graph)
        closeness = nx.closeness_centrality(graph)
        eigenvector = nx.eigenvector_centrality(graph)
        betweenness = nx.betweenness_centrality(graph)
        pagerank = nx.pagerank(graph)

        attributes = {}
        for node in graph.nodes:
            attributes[node] = {
                "Degree": degrees[node],
                "Clustering": clustering[node],
                "Closeness": closeness[node],
                "Eigenvector": eigenvector[node],
                "Betweenness": betweenness[node],
                "Pagerank": pagerank[node],
            }
        return attributes

    def _roll_dice(self, attributes, k):
        """Simulate a D20 dice roll influenced by node attributes.

        A different attribute is used based on the simplex level.
        """

        attribute = None
        if k == 1:
            attribute = attributes["Degree"]
        elif k == 2:
            attribute = attributes["Clustering"]
        elif k == 3:
            attribute = attributes["Closeness"]
        elif k == 4:
            attribute = attributes["Eigenvector"]
        elif k == 5:
            attribute = attributes["Betweenness"]
        else:
            attribute = attributes["Pagerank"]

        base_roll = random.randint(1, 20)
        modifier = int(attribute * 20)
        return base_roll + modifier
