import abc

import networkx as nx
import numpy as np
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from toponetx.classes import CellComplex, SimplicialComplex
from torch_geometric.utils.undirected import is_undirected, to_undirected

from topobenchmark.data.utils.domain import ComplexData
from topobenchmark.data.utils.utils import (
    generate_zero_sparse_connectivity,
    select_neighborhoods_of_interest,
)


class Adapter(abc.ABC):
    """Adapt between data structures representing the same domain."""

    def __call__(self, domain):
        """Adapt domain's data structure."""
        return self.adapt(domain)

    @abc.abstractmethod
    def adapt(self, domain):
        """Adapt domain's data structure."""


class IdentityAdapter(Adapter):
    """Identity adaptation.

    Retrieves same data structure for domain.
    """

    def adapt(self, domain):
        """Adapt domain."""
        return domain


class Data2Graph(Adapter):
    """Data to nx.Graph adaptation.

    Parameters
    ----------
    preserve_edge_attr : bool
        Whether to preserve edge attributes.
    """

    def __init__(self, preserve_edge_attr=False):
        self.preserve_edge_attr = preserve_edge_attr

    def _data_has_edge_attr(self, data: torch_geometric.data.Data) -> bool:
        r"""Check if the input data object has edge attributes.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        bool
            Whether the data object has edge attributes.
        """
        return hasattr(data, "edge_attr") and data.edge_attr is not None

    def adapt(self, domain: torch_geometric.data.Data) -> nx.Graph:
        r"""Generate a NetworkX graph from the input data object.

        Parameters
        ----------
        domain : torch_geometric.data.Data
            The input data.

        Returns
        -------
        nx.Graph
            The generated NetworkX graph.
        """
        # Check if data object have edge_attr, return list of tuples as [(node_id, {'features':data}, 'dim':1)] or ??
        nodes = [
            (n, dict(features=domain.x[n], dim=0))
            for n in range(domain.x.shape[0])
        ]

        if self.preserve_edge_attr and self._data_has_edge_attr(domain):
            # In case edge features are given, assign features to every edge
            # TODO: confirm this is the desired behavior
            if is_undirected(domain.edge_index, domain.edge_attr):
                edge_index, edge_attr = (domain.edge_index, domain.edge_attr)
            else:
                edge_index, edge_attr = to_undirected(
                    domain.edge_index, domain.edge_attr
                )

            edges = [
                (i.item(), j.item(), dict(features=edge_attr[edge_idx], dim=1))
                for edge_idx, (i, j) in enumerate(
                    zip(edge_index[0], edge_index[1], strict=False)
                )
            ]

        else:
            # If edge_attr is not present, return list list of edges
            edges = [
                (i.item(), j.item(), {})
                for i, j in zip(
                    domain.edge_index[0], domain.edge_index[1], strict=False
                )
            ]
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph


class Complex2ComplexData(Adapter):
    """toponetx.Complex to ComplexData adaptation.

    NB: order of features plays a crucial role, as ``ComplexData``
    simply stores them as lists (i.e. the reference to the indices
    of the simplex are lost).

    Parameters
    ----------
    neighborhoods : list, optional
        List of neighborhoods of interest.
    signed : bool, optional
        If True, returns signed connectivity matrices.
    transfer_features : bool, optional
        Whether to transfer features.
    """

    def __init__(
        self,
        neighborhoods=None,
        signed=False,
        transfer_features=True,
    ):
        super().__init__()
        self.neighborhoods = neighborhoods
        self.signed = signed
        self.transfer_features = transfer_features

    def adapt(self, domain):
        """Adapt toponetx.Complex to ComplexData.

        Parameters
        ----------
        domain : toponetx.Complex

        Returns
        -------
        ComplexData
        """
        practical_dim = (
            domain.practical_dim
            if hasattr(domain, "practical_dim")
            else domain.dim
        )
        dim = domain.dim

        signed = self.signed
        neighborhoods = self.neighborhoods

        connectivity_infos = [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "coadjacency",
            "hodge_laplacian",
        ]

        practical_shape = list(
            np.pad(
                list(domain.shape), (0, practical_dim + 1 - len(domain.shape))
            )
        )
        data = {
            connectivity_info: [] for connectivity_info in connectivity_infos
        }
        for rank in range(practical_dim + 1):
            for connectivity_info in connectivity_infos:
                try:
                    data[connectivity_info].append(
                        from_sparse(
                            getattr(domain, f"{connectivity_info}_matrix")(
                                rank=rank, signed=signed
                            )
                        )
                    )
                except ValueError:
                    if connectivity_info == "incidence":
                        data[connectivity_info].append(
                            generate_zero_sparse_connectivity(
                                m=practical_shape[rank - 1],
                                n=practical_shape[rank],
                            )
                        )
                    else:
                        data[connectivity_info].append(
                            generate_zero_sparse_connectivity(
                                m=practical_shape[rank],
                                n=practical_shape[rank],
                            )
                        )

        # TODO: handle this
        if neighborhoods is not None:
            data = select_neighborhoods_of_interest(data, neighborhoods)

        if self.transfer_features:
            if isinstance(domain, SimplicialComplex):
                get_features = domain.get_simplex_attributes
            elif isinstance(domain, CellComplex):
                get_features = domain.get_cell_attributes
            else:
                raise ValueError("Can't transfer features.")

            data["features"] = []
            for rank in range(dim + 1):
                rank_features_dict = get_features("features", rank)
                if rank_features_dict:
                    rank_features = torch.stack(
                        list(rank_features_dict.values())
                    )
                else:
                    rank_features = None
                data["features"].append(rank_features)

            for _ in range(dim + 1, practical_dim + 1):
                data["features"].append(None)

        return ComplexData(**data)


class ComplexData2Dict(Adapter):
    """ComplexData to dict adaptation."""

    def adapt(self, domain):
        """Adapt ComplexData to dict.

        Parameters
        ----------
        domain : ComplexData

        Returns
        -------
        dict
        """
        data = {}
        connectivity_infos = [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "coadjacency",
            "hodge_laplacian",
        ]
        for connectivity_info in connectivity_infos:
            info = getattr(domain, connectivity_info)
            for rank, rank_info in enumerate(info):
                data[f"{connectivity_info}_{rank}"] = rank_info

        data["shape"] = domain.shape

        for index, values in enumerate(domain.features):
            if values is not None:
                data[f"x_{index}"] = values

        return data


class HypergraphData2Dict(Adapter):
    """HypergraphData to dict adaptation."""

    def adapt(self, domain):
        """Adapt HypergraphData to dict.

        Parameters
        ----------
        domain : HypergraphData

        Returns
        -------
        dict
        """
        hyperedges_key = domain.rank_keys()[-1]
        return {
            "incidence_hyperedges": domain.incidence[hyperedges_key],
            "num_hyperedges": domain.num_hyperedges,
            "x_0": domain.features[0],
            "x_hyperedges": domain.features[hyperedges_key],
        }


class AdapterComposition(Adapter):
    """Composed adapter."""

    def __init__(self, adapters):
        super().__init__()
        self.adapters = adapters

    def adapt(self, domain):
        """Adapt domain"""
        for adapter in self.adapters:
            domain = adapter(domain)

        return domain


class Complex2Dict(AdapterComposition):
    """toponetx.Complex to dict adaptation.

    Parameters
    ----------
    neighborhoods : list, optional
        List of neighborhoods of interest.
    signed : bool, optional
        If True, returns signed connectivity matrices.
    transfer_features : bool, optional
        Whether to transfer features.
    """

    def __init__(
        self,
        neighborhoods=None,
        signed=False,
        transfer_features=True,
    ):
        tnxcomplex2complex = Complex2ComplexData(
            neighborhoods=neighborhoods,
            signed=signed,
            transfer_features=transfer_features,
        )
        complex2dict = ComplexData2Dict()
        super().__init__(adapters=(tnxcomplex2complex, complex2dict))
