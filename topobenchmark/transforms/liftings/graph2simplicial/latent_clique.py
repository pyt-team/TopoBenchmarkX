r"""This module implements the LatentCliqueLifting class.

In the context of Topological Deep Learning [PBB2024]_[HZP2023]_,
and the very recently emerged paradigm of Latent Topology Inference
(LTI) [BST2023]_, it is natural to look at the model in [WT2020]_ as a
novel LTI method able to infer a random latent simplicial complex from an
input graph. Or, in other words, to use [WT2020]_ as a novel random lifting
procedure from graphs to simplicial complexes.

To summarize, this is:

* a non-deterministic lifting
* not present in the literature as a lifting procedure
* based on connectivity
* | modifying the initial connectivity of the graph by
  | adding edges (thus, this can be also considered as a graph rewiring method).

The lifting ensures both 1) small-world property and 2) edge/cell sparsity.
Combining these two properties is very attractive for Topological Deep Learning (TDL)
because it ensures computational efficiency due to the reduced number of higher-order
connections: only a few message-passing layers connect any two nodes.

References
----------
.. [WT2020] Williamson, S.A., Tec, M., 2020. Random Clique Covers for Graphs with Local
    Density and Global Sparsity, in: Proceedings of The 35th Uncertainty in Artificial
    Intelligence Conference.
    Presented at the Uncertainty in Artificial Intelligence, PMLR, pp. 228--238.
    http://proceedings.mlr.press/v115/williamson20a/williamson20a.pdf
.. [PBB2024] Papamarkou, T., Birdal, T., Bronstein, M., et al., 2024.
    Position Paper: Challenges and Opportunities in Topological Deep Learning.
    https://doi.org/10.48550/arXiv.2402.08871
.. [HZP2023] Hajij, M., Zamzmi, G., Papamarkou, T., et al., 2023.
    Topological Deep Learning: Going Beyond Graph Data.
    https://doi.org/10.48550/arXiv.2206.00606
.. [BST2023] Battiloro, C., Spinelli, I., Telyatnikov, L., et al., 2023.
    From Latent Graph to Latent Topology Inference: Differentiable Cell
    Complex Module. Presented at the The Twelfth International Conference
    on Learning Representations.
"""

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
