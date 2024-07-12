import itertools as it
from array import array
from collections import Counter, defaultdict
from collections.abc import Generator, Sized
from functools import cache, reduce
from math import comb, factorial
from operator import or_

import numpy as np
import toponetx as tnx
import torch
import torch_geometric
from scipy.sparse import coo_array, sparray
from scipy.sparse.linalg import eigsh

from modules.transforms.liftings.hypergraph2simplicial.base import (
    Hypergraph2SimplicialLifting,
)


def spectral_bounds(L: sparray) -> tuple:
    """Estimates the spectral gap and spectral radius of a given Laplacian matrix"""
    radius = eigsh(L, k=1, which="LM", return_eigenvectors=False).item()
    gap = max(1.0 / L.shape[0] ** 4, 0.5 * (2 / np.sum(L.diagonal())) ** 2)
    return {"gap": gap, "radius": radius}


def faces(simplex: tuple, proper: bool = False) -> Generator:
    """Generates the faces of a given simplex, optionally including itself."""
    max_dim = len(simplex) - int(proper)
    faces_gen = (it.combinations(simplex, d) for d in range(1, max_dim + 1))
    yield from it.chain(*faces_gen)


def dim(simplex: Sized) -> int:
    """Returns the dimension of sized object, which is given by its length - 1"""
    return len(simplex) - 1


def base_map(dim: int) -> dict:
    """Base dimension -> weight mapping defined by sending the k faces of an n-simplex to n! / (n - k)!"""
    if dim <= 8:
        return Counter(
            {
                d: 1.0 / (factorial(dim) / factorial(dim - k))
                for d, k in enumerate(range(dim + 1))
            }
        )
    typ_wghts = Counter(
        {d: 1.0 / (factorial(dim) / factorial(dim - k)) for d, k in enumerate(range(9))}
    )
    near_zero = Counter(
        {d: np.finfo(np.float64).eps for d, k in enumerate(range(9, dim + 1))}
    )
    return typ_wghts + near_zero


def weighted_simplex(simplex: tuple) -> dict:
    """Constructs a dictionary mapping faces of 'sigma' to *topological weights*.

    The resulting simplex->weight mapping obeys the property that every simplex's weight
    is strictly positive and is identical to the sum of its cofacet weights. Moreover,
    every vertex weight is equal to the number of times it appears in a maximal face,
    and descending order of the weights respect the face poset of the simplex.

    This relation is preserved under addition non-enclosing simplex->weight mappings.

    Parameters:
      simplex = tuple of vertex labels

    Returns:
      dictionary mapping simplices to strictly positive topological.

    """
    weights = defaultdict(float)
    base_weights = base_map(dim(simplex))
    for f in faces(simplex, proper=True):
        weights[f] += base_weights[dim(f)]
    weights[tuple(simplex)] = 1 / factorial(dim(simplex))
    return Counter(weights)


def unit_simplex(sigma: tuple, c: float = 1.0, closure: bool = False) -> dict:
    """Constructs a dictionary mapping 'sigma' or its closure to a constant weight.

    This can be used in conjunction with 'weighted_simplex' to encode hypergraphs as
    weighted simplicial complexes in such a way that the hypergraph is fully recoverable,
    though for learning purposes one might not want to do this.
    """
    weights = defaultdict(float)
    if closure:
        for f in faces(sigma, proper=True):
            weights[f] += c
    weights[tuple(sigma)] = c
    return Counter(weights)


## From: https://stackoverflow.com/questions/42138681/faster-numpy-solution-instead-of-itertools-combinations
@cache
def _combs(n: int, k: int) -> np.ndarray:
    """Faster numpy-version of itertools.combinations over the standard indest set {0, 1, ..., n}"""
    if n < k:
        return np.empty(shape=(0,), dtype=int)
    a = np.ones((k, n - k + 1), dtype=int)
    a[0] = np.arange(n - k + 1)
    for j in range(1, k):
        reps = (n - k + j) - a[j - 1]
        a = np.repeat(a, reps, axis=1)
        ind = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1 - reps[1:]
        a[j, 0] = j
        a[j] = np.add.accumulate(a[j])
    return a


def downward_closure(H: list, d: int = 1, coeffs: bool = False):
    """Constructs a simplicial complex from a hypergraph by taking its downward closure, optionally counting higher order interactions.

    This function implicitly converts a hypergraph into a simplicial complex by taking the downward closure of each hyperedge
    and collecting the corresponding d-simplices. Note that only the d-simplices are returned (maximal p-simplices for p < d
    won't be included!). If coeffs = True, a n x D sparse matrix is returned whose non-zero values at index (i,j) count the number of
    times the corresponding i-th d-simplex appeared in a j-dimensional hyperedge.

    The output of this function is meant to be used in conjunction with top_weights to compute topological weights.

    Parameters:
      H = list of hyperedges
      d = simplex dimension to extract
      coeffs = whether to extract the membership coefficients

    Returns:
      list of the maximal d-simplices in the downward closure of H. If coeffs = True, higher-order membership counts are also returned.
    """
    assert isinstance(d, int), "simplex dimension must be integral"
    H = normalize_hg(H)

    ## If coeffs = False, just do the restriction operation
    if not coeffs:
        S = set()  # about 15x faster than sortedset
        for he in H:
            d_simplices = map(tuple, it.combinations(he, d + 1))
            S.update(d_simplices)
        S = np.array(list(S), dtype=(int, (d + 1,)))
        S.sort(axis=1)
        return S

    ## Extract the lengths of the hyperedges and how many d-simplices we may need
    from hirola import HashTable

    H_sizes = np.array([len(he) for he in H])
    MAX_HT_SIZE = int(np.sum([comb(sz, d + 1) for sz in H_sizes]))

    ## Allocate the two output containers
    S = HashTable(int(MAX_HT_SIZE * 1.20) + 8, dtype=(int, d + 1))
    card_memberships = [array("I") for _ in range(np.max(H_sizes))]
    for he in (he for he in H if len(he) > d):
        d_simplices = he[_combs(len(he), d + 1)].T
        s_keys = S.add(d_simplices)
        card_memberships[len(he) - 1].extend(s_keys.flatten())

    ## Construct the coauthorship coefficients
    R, C, X = array("I"), array("I"), array("I")
    for j, members in enumerate(card_memberships):
        cc = Counter(members)
        R.extend(cc.keys())
        C.extend(np.full(len(cc), j))
        X.extend(cc.values())
    coeffs = coo_array((X, (R, C)), shape=(len(S), len(card_memberships)))
    coeffs.eliminate_zeros()
    S = S.keys.reshape(len(S.keys), d + 1)
    return S, coeffs


def normalize_hg(H: list):
    """Normalizes a list of hyperedges to a canonical form.

    This maps all node ids back to the standard index set [0, 1, ..., n - 1].
    """
    V = np.fromiter(reduce(or_, map(set, H)), dtype=int)
    return [np.unique(np.searchsorted(V, np.sort(he).astype(int))) for he in H]


def top_weights(simplices: np.ndarray, coeffs: sparray, normalize: bool = False):
    """Computes topological weights from higher-order interaction data."""
    assert isinstance(coeffs, sparray), "Coefficients must be sparse matrix"
    assert coeffs.shape[0] == len(
        simplices
    ), "Invalid shape; must have a set of coefficients for each simplex"
    coeffs = coeffs.tocoo() if not hasattr(coeffs, "row") else coeffs
    simplices = np.atleast_2d(simplices)
    N = simplices.shape[1]
    topo_weights = np.zeros(coeffs.shape[0])
    if normalize:
        c, d = 1.0 / factorial(N - 1), N - 1
        _coeff_weights = c * np.array(
            [p / comb(a, d) for p, a in zip(coeffs.data, coeffs.col, strict=True)]
        )
        np.add.at(topo_weights, coeffs.row, _coeff_weights)
    else:
        base_weights = np.array([base_map(d)[N - 1] for d in range(coeffs.shape[1])])
        np.add.at(topo_weights, coeffs.row, coeffs.data * base_weights[coeffs.col])
    return Counter(dict(zip(map(tuple, simplices), topo_weights, strict=True)))


def vertex_counts(H: list) -> np.ndarray:
    """Returns the number of times a"""
    N = np.max([np.max(he) for he in normalize_hg(H)]) + 1
    v_counts = np.zeros(N)
    for he in normalize_hg(H):
        v_counts[he] += 1
    return v_counts


class HypergraphHeatLifting(Hypergraph2SimplicialLifting):
    r"""Lifts hypergraphs to the simplicial complex domain by assigning positive topological weights to the downward closure of the hypergraph.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data) -> dict:
        r"""Lifts the topology of a hypergraph to a simplicial complex by taking the downward closure and weighting the simplices.

        Parameters
        ----------
        data : torch_geometric.data
            The input hypergraph to be 'lifted' to a simplicial complex

        Returns
        -------
        dict
            The lifted topology.
        """
        print("Lifting to weighted simplicial complex")

        ## Convert incidence to simple list of hyperedges
        R, C = data.incidence_hyperedges.coalesce().indices().detach().numpy()
        col_sort = np.argsort(C)
        R, C = R[col_sort], C[col_sort]
        hyperedges = np.split(R, np.cumsum(np.unique(C, return_counts=True)[1])[:-1])
        hyperedges = normalize_hg(hyperedges)

        ## Compute the simplex -> topological weight map using the downward closure of the hyperedges
        max_dim = data.get("max_dim", 2)
        SC = tnx.SimplicialComplex()
        SC_map = {}
        for d in range(max_dim + 1):
            simplex_map = top_weights(*downward_closure(hyperedges, d, coeffs=True))
            SC.add_simplices_from(simplex_map.keys())
            SC_map.update(simplex_map)

        ## Build the boundary matrices, save the weights
        lifted_topology = {}
        for d in range(max_dim + 1):
            _, CI, D = SC.incidence_matrix(d, index=True)
            D = D.tocoo()
            lifted_topology[f"incidence_{d}"] = torch.sparse_coo_tensor(
                D.nonzero(), D.data, D.shape
            )
            lifted_topology[f"weights_{d}"] = torch.tensor(
                np.array([SC_map[s] for s in CI])
            )
            lifted_topology[f"x_{d}"] = torch.atleast_2d(
                lifted_topology[f"weights_{d}"]
            )
        return torch_geometric.data.Data(**lifted_topology)
