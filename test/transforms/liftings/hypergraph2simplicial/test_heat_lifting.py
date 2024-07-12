from collections import Counter

import numpy as np

from modules.transforms.liftings.hypergraph2simplicial.heat_lifting import (
    downward_closure,
    top_weights,
    unit_simplex,
    vertex_counts,
    weighted_simplex,
)


## Ensure the cofacet relations hold; this should hold for all simplices with base weights / "topological weight"
def cofacet_constraint(
    S: dict, d: int = None, relation: str = ">=", verbose: bool = False
) -> bool:
    assert isinstance(S, dict), "Input mut be a simplicial weight map"
    relation_holds: bool = True
    for s in S.keys():
        s_weight = S[s]
        s_cofacets = [
            c for c in S.keys() if len(c) == len(s) + 1 and np.all(np.isin(s, c))
        ]
        c_weight = np.sum([S[c] for c in s_cofacets])
        relation_holds &= (
            eval(f"s_weight {relation} c_weight")
            or np.isclose(s_weight, c_weight)
            or len(s_cofacets) == 0
        )
        if verbose and (not relation_holds) and len(s_cofacets) > 0:
            print(
                f"simplex {s} weight {s_weight:.3f} !({relation}) cofacet weight {c_weight:.3f}"
            )
    return relation_holds


def coauthorship_constraint(S: dict, v_counts: np.ndarray) -> bool:
    ## Constraint: the sum of edges of the vertices matches the number of times they appear in the hyper edges
    same_weight = np.array(
        [np.isclose(S[(i,)], v_counts[i]) for i in np.arange(len(v_counts))]
    )
    return np.all(same_weight)


def positivity_constraint(S: dict) -> bool:
    return np.all([v > 0 for v in S.values()])


class TestHypergraphHeatLifting:
    ## Ensure equality constraints hold for maximal simplices
    def test_maximal_constraints(self):
        H = [(0, 1, 2), (1, 2, 3)]
        weights = sum(map(weighted_simplex, H), Counter())
        assert positivity_constraint(weights)
        assert coauthorship_constraint(weights, vertex_counts(H))
        assert cofacet_constraint(weights, d=0, relation="==")

    ## Ensure the vertex/edge constraint, positivity, and coauthorship constraints hold under >= relation
    def test_nonmaximal_constraints(self):
        H = [(0, 1, 2), (1, 2, 3), (1, 2)]
        weights = sum(map(weighted_simplex, H), Counter())
        assert positivity_constraint(weights)
        assert coauthorship_constraint(weights, vertex_counts(H))
        assert cofacet_constraint(weights, relation=">=")
        assert not cofacet_constraint(weights, relation="==")

    ## Test (trivial) reconstruction with known maximal simplices
    def test_recover_hypergraph(self):
        H = [(0, 1, 2), (1, 2, 3), (1, 2)]
        weights = sum(map(weighted_simplex, H[:2]), Counter())
        weights += unit_simplex(H[2])
        H_recon = weights - sum(map(weighted_simplex, H[:2]), Counter())
        assert H_recon == unit_simplex(H[2])

    ## Larger test of all the relation the weight mapping should obey
    def test_relations(self):
        H = [
            (0,),
            (0, 1),
            (1, 3),
            (1, 2, 3),
            (0, 1, 2, 3),
            (0, 1, 4),
            (0, 1, 3),
            (2, 5),
            (0, 2, 5),
            (0, 2, 4, 5),
        ]
        sc_lift = sum(map(weighted_simplex, H), Counter())
        assert positivity_constraint(sc_lift)
        assert coauthorship_constraint(sc_lift, vertex_counts(H))
        assert cofacet_constraint(sc_lift, d=0, relation=">=")
        assert cofacet_constraint(sc_lift, relation=">=")
        assert not cofacet_constraint(sc_lift, relation="==")

    ## Ensure we can mimick the same weight mapping with the fast d-skeleton code
    def test_downward_closure(self):
        H = [
            (0,),
            (0, 1),
            (1, 3),
            (1, 2, 3),
            (0, 1, 2, 3),
            (0, 1, 4),
            (0, 1, 3),
            (2, 5),
            (0, 2, 5),
            (0, 2, 4, 5),
        ]
        sc_lift = sum(map(weighted_simplex, H), Counter())
        for d in range(3):
            d_map = top_weights(*downward_closure(H, d=d, coeffs=True))
            assert np.all([np.isclose(sc_lift[s], w) for s, w in d_map.items()])
