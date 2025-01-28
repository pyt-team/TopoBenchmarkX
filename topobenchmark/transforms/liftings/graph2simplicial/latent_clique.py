from topobenchmark.transforms.liftings.base import ComposedLiftingMap
from topobenchmark.transforms.liftings.graph2graph.latent_clique import (
    LatentGraphLifting,
)
from topobenchmark.transforms.liftings.graph2simplicial.clique import (
    SimplicialCliqueLifting,
)


class LatentCliqueLifting(ComposedLiftingMap):
    r"""Lifts graphs to cell complexes by identifying the cycles as 2-cells.

    Parameters
    ----------
    edge_prob_mean : float = 0.9
        Mean of the prior distribution of pie ~ Beta
        where edge_prob_mean must be in (0, 1).
        When edge_prob_mean is one, the value of edge_prob is fixed and not sampled.
    edge_prob_var : float = 0.05
        Uncertainty of the prior distribution of pie ~ Beta(a, b)
        where edge_prob_var must be in [0, inf). When edge_prob_var is zero,
        the value of edge_prob is fixed and not sampled. It is require dthat
        edge_prob_var < edge_prob_mean * (1 - edge_prob_mean). When this is not the case
        the value of edge_prob_var is set to edge_prob_mean * (1 - edge_prob_mean) - 1e-6.
    it : int, optional
        Number of iterations for sampling, by default None.
    init : str, optional
        Initialization method for the clique cover matrix, by default "edges".
    verbose : bool, optional
        Whether to display verbose output, by default False.
    complex_dim : int
        Dimension of the subcomplex.
    """

    def __init__(
        self,
        edge_prob_mean=0.9,
        edge_prob_var=0.05,
        it=None,
        init="edges",
        verbose=False,
        complex_dim=2,
    ):
        graph2graph = LatentGraphLifting(
            edge_prob_mean=edge_prob_mean,
            edge_prob_var=edge_prob_var,
            it=it,
            init=init,
            verbose=verbose,
        )
        graph2simplicial = SimplicialCliqueLifting(complex_dim=complex_dim)
        super().__init__([graph2graph, graph2simplicial])
