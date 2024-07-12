import numpy as np
import networkx as nx
import torch_geometric
import itertools as it
import toponetx as tnx
import torch

from functools import cache, reduce
from operator import or_
from array import array
from scipy.sparse import coo_array, coo_array
from collections import Counter, defaultdict
from typing import Sized, Generator
from math import factorial, comb
from scipy.sparse import sparray
from modules.transforms.liftings.hypergraph2simplicial.base import Hypergraph2SimplicialLifting

def faces(simplex: tuple, proper: bool = False) -> Generator:
  """Generates the faces of a given simplex, optionally including itself."""
  max_dim = len(simplex) - int(proper)
  faces_gen = (it.combinations(simplex, d) for d in range(1, max_dim+1))
  yield from it.chain(*faces_gen)

def dim(simplex: Sized) -> int:
  return len(simplex) - 1

def base_map(dim: int) -> dict:
  """Reciprocal of n! / (n - k)! """
  if dim <= 8: 
    return Counter({d : 1.0/(factorial(dim) / factorial(dim-k)) for d, k in enumerate(range(dim+1))})
  else: 
    typ_wghts = Counter({d : 1.0/(factorial(dim) / factorial(dim-k)) for d, k in enumerate(range(9))})
    near_zero = Counter({ d : np.finfo(np.float64).eps for d, k in enumerate(range(9, dim+1)) })
    return typ_wghts + near_zero

def weighted_simplex(sigma: tuple) -> dict:
  """Constructs a dictionary mapping faces of 'sigma' to *topological weights*.
  
  The resulting simplex->weight mapping obeys the property that every simplex's weight 
  is strictly positive and is identical to the sum of its cofacet weights. Moreover, 
  every vertex weight is equal to the number of times it appears in a maximal face, 
  and descending order of the weights respect the face poset of the simplex.
  
  This relation is preserved under addition non-enclosing simplex->weight mappings.
  """
  weights = defaultdict(float)
  base_weights = base_map(dim(sigma))
  for f in faces(sigma, proper=True):
    weights[f] += base_weights[dim(f)]
  weights[tuple(sigma)] = 1 / factorial(dim(sigma))
  return Counter(weights)

## From: https://stackoverflow.com/questions/42138681/faster-numpy-solution-instead-of-itertools-combinations
@cache
def _combs(n: int, k: int) -> np.ndarray:
  """Faster numpy-version of itertools.combinations over the standard indest set {0, 1, ..., n}"""
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

def downward_closure(H: list, d: int = 1, coeffs: bool = False):
  """Constructs a simplicial complex from a hypergraph by taking its downward closure.
  
  Parameters: 
    H = list of hyperedges / subsets of a set
    d = simplex dimension to extract
    coeffs = whether to extract the membership coefficients 

  Returns: 
    list of the maximal d-simplices in the downward closure of H, if coeffs = False. 
  """
  assert isinstance(d, int), "simplex dimension must be integral"
  H = normalize_hg(H)
  if not coeffs:
    S = set() # about 15x faster than sortedset 
    for he in H:
      d_simplices = map(tuple, it.combinations(he, d+1)) 
      S.update(d_simplices)
    S = np.array(list(S), dtype=(int, (d+1,)))
    S.sort(axis=1)
    # S = S[np.lexsort(np.rot90(S))]
    return S
  else:
    from hirola import HashTable

    ## Extract the lengths of the hyperedges and how many d-simplices we may need
    H_sizes = np.array([len(he) for he in H])
    MAX_HT_SIZE = int(np.sum([comb(sz, d+1) for sz in H_sizes]))
    
    ## Allocate the two output containers
    S = HashTable(int(MAX_HT_SIZE * 1.20) + 8, dtype=(int,d+1))
    card_memberships = [array('I') for _ in range(np.max(H_sizes))]
    for he in (he for he in H if len(he) > d):
      d_simplices = he[_combs(len(he), d+1)].T
      s_keys = S.add(d_simplices)
      card_memberships[len(he)-1].extend(s_keys.flatten())

    ## Construct the coauthorship coefficients
    from collections import Counter
    I, J, X = array('I'), array('I'), array('I')
    for j, members in enumerate(card_memberships):
      cc = Counter(members)
      I.extend(cc.keys())
      J.extend(np.full(len(cc), j))
      X.extend(cc.values())
    coeffs = coo_array((X, (I,J)), shape=(len(S), len(card_memberships)))
    coeffs.eliminate_zeros()
    return S.keys.reshape(len(S.keys), d+1), coeffs 

def normalize_hg(H: list):
  """Normalizes a list of hyperedges to a canonical form.

  This maps all node ids back to the standard index set [0, 1, ..., n - 1].
  """
  V = np.fromiter(reduce(or_, map(set, H)), dtype=int)
  H = [np.unique(np.searchsorted(V, np.sort(he).astype(int))) for he in H]
  return H

def top_weights(simplices: np.ndarray, coeffs: sparray, normalize: bool = False):
  """Computes topological weights from higher-order interaction data."""
  assert isinstance(coeffs, sparray), "Coefficients must be sparse matrix"
  assert coeffs.shape[0] == len(simplices), "Invalid shape; must have a set of coefficients for each simplex"
  coeffs = coeffs.tocoo() if not hasattr(coeffs, "row") else coeffs
  simplices = np.atleast_2d(simplices)
  N = simplices.shape[1]
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
        I, J = data.incidence_hyperedges.coalesce().indices().detach().numpy()
        col_sort = np.argsort(J)
        I,J = I[col_sort], J[col_sort]
        hyperedges = np.split(I, np.cumsum(np.unique(J, return_counts=True)[1])[:-1])
        hyperedges = normalize_hg(hyperedges)

        ## Compute the simplex -> topological weight map using the downward closure of the hyperedges
        max_dim = data.get("max_dim", 2)
        SC = tnx.SimplicialComplex()
        SC_map = {}
        for d in range(max_dim):
          simplex_map = top_weights(*downward_closure(hyperedges, d, coeffs=True))
          SC.add_simplices_from(simplex_map.keys())
          SC_map.update(simplex_map)

        ## Build the boundary matrices, save the weights
        lifted_topology = {}
        for d in range(max_dim+1):
          _, CI, D = SC.incidence_matrix(d, index=True)
          D = D.tocoo()
          lifted_topology[f"incidence_{d}"] = torch.sparse_coo_tensor(D.nonzero(), D.data, D.shape)
          lifted_topology[f"weights_{d}"] = torch.tensor(np.array([SC_map[s] for s in CI.keys()]))
          lifted_topology[f"x_{d}"] = torch.atleast_2d(lifted_topology[f"weights_{d}"])
        # print(lifted_topology)
        sc_data = torch_geometric.data.Data(**lifted_topology)
        return sc_data
