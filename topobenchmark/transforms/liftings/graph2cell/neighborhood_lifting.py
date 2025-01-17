import torch_geometric
from toponetx.classes import CellComplex

from modules.transforms.liftings.graph2cell.base import Graph2CellLifting


class NeighborhoodLifting(Graph2CellLifting):
    r"""Lifts graphs to cell complexes by identifying the cycles as 2-cells.

    Parameters
    ----------
    max_cell_length : int, optional
        The maximum length of the cycles to be lifted. Default is None.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_cell_length=None, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = 2
        self.max_cell_length = max_cell_length

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Finds the cycles of a graph and lifts them to 2-cells.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """

        G = self._generate_graph_from_data(data)

        cell_complex = CellComplex(G)

        vertices = list(G.nodes())
        for v in vertices:
            cell_complex.add_node(v, rank=0)

        edges = list(G.edges())
        for edge in edges:
            cell_complex.add_cell(edge, rank=1)

        for v in vertices:
            neighbors = list(G.neighbors(v))
            if len(neighbors) > 1:
                two_cell = [v, *neighbors]
                if (
                    self.max_cell_length is not None
                    and len(two_cell) > self.max_cell_length
                ):
                    pass
                else:
                    cell_complex.add_cell(two_cell, rank=2)

        return self._get_lifted_topology(cell_complex, G)
