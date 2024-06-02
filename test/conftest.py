import pytest
import torch, torch_geometric
import networkx as nx

@pytest.fixture
def mocker_fixture(mocker):
    return mocker

@pytest.fixture
def simple_graph_1():
    """Create a manual graph for testing purposes."""
    # Define the vertices (just 8 vertices)
    vertices = [i for i in range(8)]
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    # Define the edges
    edges = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 2],
        [2, 3],
        [5, 2],
        [5, 6],
        [6, 3],
        [5, 7],
        [2, 7],
        [0, 7],
    ]

    # Define the tetrahedrons
    tetrahedrons = [[0, 1, 2, 4]]

    # Add tetrahedrons
    for tetrahedron in tetrahedrons:
        for i in range(len(tetrahedron)):
            for j in range(i + 1, len(tetrahedron)):
                edges.append([tetrahedron[i], tetrahedron[j]])  # noqa: PERF401

    # Create a graph
    G = nx.Graph()

    # Add vertices
    G.add_nodes_from(vertices)

    # Add edges
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()

    # Generate feature from 0 to 9
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()

    data = torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
    )
    return data


@pytest.fixture
def simple_graph_2():
    """Create a manual graph for testing purposes."""
    # Define the vertices (just 9 vertices)
    vertices = [i for i in range(9)]
    y = [0, 1, 1, 1, 0, 0, 0, 0, 0]
    # Define the edges
    edges = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 3],
        [2, 3],
        [5, 2],
        [5, 6],
        [6, 3],
        [2, 6],
        [5, 7],
        [2, 8],
        [0, 8],
    ]

    # Define the tetrahedrons
    tetrahedrons = [[0, 1, 2, 3], [0, 1, 2, 4]]

    # Add tetrahedrons
    for tetrahedron in tetrahedrons:
        for i in range(len(tetrahedron)):
            for j in range(i + 1, len(tetrahedron)):
                edges.append([tetrahedron[i], tetrahedron[j]])  # noqa: PERF401

    # Create a graph
    G = nx.Graph()

    # Add vertices
    G.add_nodes_from(vertices)

    # Add edges
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()

    # Generate feature from 0 to 9
    x = (
        torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000, 10000])
        .unsqueeze(1)
        .float()
    )

    data = torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
    )
    return data