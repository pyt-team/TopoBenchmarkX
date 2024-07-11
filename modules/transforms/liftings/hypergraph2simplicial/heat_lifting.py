import numpy as np
import networkx as nx
import torch_geometric
import itertools as it

from functools import cache, reduce
from operator import or_
from array import array
from scipy.sparse import coo_array
from collections import Counter
from math import factorial, comb
from scipy.sparse import sparray
from toponetx.classes import SimplicialComplex, ColoredHyperGraph
from modules.transforms.liftings.hypergraph2simplicial.base import Hypergraph2SimplicialLifting


## From: https://stackoverflow.com/questions/42138681/faster-numpy-solution-instead-of-itertools-combinations
@cache
def _combs(n: int, k: int) -> np.ndarray:
  if n < k: return np.empty(shape=(0,),dtype=int)
  a = np.ones((k, n-k+1), dtype=int)
  a[0] = np.arange(n-k+1)
  for j in range(1, k):
    reps = (n-k+j) - a[j-1]
    a = np.repeat(a, reps, axis=1)
    ind = np.add.accumulate(reps)
    a[j, ind[:-1]] = 1-reps[1:]
    a[j, 0] = j
    a[j] = np.add.accumulate(a[j])
  return a

def normalize_hg(H: list):
  """Normalizes a set of hyperedges to a canonical form"""
  V = np.fromiter(reduce(or_, map(set, H)), dtype=int)
  H = [np.unique(np.searchsorted(V, np.sort(he).astype(int))) for he in H]
  return H

def top_weights(simplices: np.ndarray, coeffs: sparray, normalize: bool = False):
  """Computes topological weights from higher-order interaction data."""
  assert isinstance(coeffs, sparray), "Coefficients must be sparse matrix"
  assert coeffs.shape[0] == len(simplices), "Invalid shape; must have a set of coefficients for each simplex"
  coeffs = coeffs.tocoo() if not hasattr(coeffs, "row") else coeffs
  simplices = np.atleast_2d(simplices)
  n, N = simplices.shape
  topo_weights = np.zeros(coeffs.shape[0])
  if normalize: 
    c, d = 1.0 / factorial(N-1), N-1
    _coeff_weights = c * np.array([p / comb(a, d) for p, a in zip(coeffs.data, coeffs.col)])
    np.add.at(topo_weights, coeffs.row, _coeff_weights)
  else: 
    base_weights = np.array([base_map(d)[N-1] for d in range(coeffs.shape[1])])
    np.add.at(topo_weights, coeffs.row, coeffs.data * base_weights[coeffs.col])  
  return Counter(dict(zip(map(tuple, simplices), topo_weights))) 

def vertex_counts(H: list) -> np.ndarray:
  """Returns the number of times a """
  N = np.max([np.max(he) for he in normalize_hg(H)])+1
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

    def lift_topology(self, data: ColoredHyperGraph) -> dict:
        r"""Lifts the topology of a graph to a simplicial complex by identifying the cliques as k-simplices.

        Parameters
        ----------
        data : ColoredHyperGraph
            The input hypgraph to be 'lifted' to a simplicial complex

        Returns
        -------
        dict
            The lifted topology.
        """
        ## Features in xp
        ## hodge_laplacian_rank0
        return dict()
