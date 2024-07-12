import networkx as nx
import torch
from toponetx.classes import ColoredHyperGraph, SimplicialComplex

# from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.lifting import HypergraphLifting


class Hypergraph2SimplicialLifting(HypergraphLifting):
    r"""Abstract class for lifting hyper graphs to simplicial complexes.

    Parameters
    ----------
    complex_dim : int, optional
        The dimension of the simplicial complex to be generated. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = complex_dim
        self.type = "hypergraph2simplicial"
        self.signed = kwargs.get("signed", False)
