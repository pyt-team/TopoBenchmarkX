"""Data IO utilities."""

import os.path as osp
import pickle
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_sparse import coalesce


def get_file_id_from_url(url):
    """Extract the file ID from a Google Drive file URL.

    Parameters
    ----------
    url : str
        The Google Drive file URL.

    Returns
    -------
    str
        The file ID extracted from the URL.

    Raises
    ------
    ValueError
        If the provided URL is not a valid Google Drive file URL.
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "id" in query_params:  # Case 1: URL format contains '?id='
        file_id = query_params["id"][0]
    elif (
        "file/d/" in parsed_url.path
    ):  # Case 2: URL format contains '/file/d/'
        file_id = parsed_url.path.split("/")[3]
    else:
        raise ValueError(
            "The provided URL is not a valid Google Drive file URL."
        )
    return file_id


def download_file_from_drive(
    file_link, path_to_save, dataset_name, file_format="tar.gz"
):
    """Download a file from a Google Drive link and saves it to the specified path.

    Parameters
    ----------
    file_link : str
        The Google Drive link of the file to download.
    path_to_save : str
        The path where the downloaded file will be saved.
    dataset_name : str
        The name of the dataset.
    file_format : str, optional
        The format of the downloaded file. Defaults to "tar.gz".

    Raises
    ------
    None
    """
    file_id = get_file_id_from_url(file_link)

    download_link = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_link)

    output_path = f"{path_to_save}/{dataset_name}.{file_format}"
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Failed to download the file.")


def read_us_county_demos(path, year=2012, y_col="Election"):
    """Load US County Demos dataset.

    Parameters
    ----------
    path : str
        Path to the dataset.
    year : int, optional
        Year to load the features (default: 2012).
    y_col : str, optional
        Column to use as label. Can be one of ['Election', 'MedianIncome',
        'MigraRate', 'BirthRate', 'DeathRate', 'BachelorRate', 'UnemploymentRate'] (default: "Election").

    Returns
    -------
    torch_geometric.data.Data
        Data object of the graph for the US County Demos dataset.
    """
    edges_df = pd.read_csv(f"{path}/county_graph.csv")
    stat = pd.read_csv(
        f"{path}/county_stats_{year}.csv", encoding="ISO-8859-1"
    )

    keep_cols = [
        "FIPS",
        "DEM",
        "GOP",
        "MedianIncome",
        "MigraRate",
        "BirthRate",
        "DeathRate",
        "BachelorRate",
        "UnemploymentRate",
    ]

    # Select columns, replace ',' with '.' and convert to numeric
    stat = stat.loc[:, keep_cols]
    stat["MedianIncome"] = stat["MedianIncome"].replace(",", ".", regex=True)
    stat = stat.apply(pd.to_numeric, errors="coerce")

    # Step 2: Substitute NaN values with column mean
    for column in stat.columns:
        if column != "FIPS":
            mean_value = stat[column].mean()
            stat[column] = stat[column].fillna(mean_value)
    stat = stat[keep_cols].dropna()

    # Delete edges that are not present in stat df
    unique_fips = stat["FIPS"].unique()

    src_ = edges_df["SRC"].apply(lambda x: x in unique_fips)
    dst_ = edges_df["DST"].apply(lambda x: x in unique_fips)

    edges_df = edges_df[src_ & dst_]

    # Remove rows from stat df where edges_df['SRC'] or edges_df['DST'] are not present
    stat = stat[
        stat["FIPS"].isin(edges_df["SRC"]) & stat["FIPS"].isin(edges_df["DST"])
    ]
    stat = stat.reset_index(drop=True)

    # Remove rows where SRC == DST
    edges_df = edges_df[edges_df["SRC"] != edges_df["DST"]]

    # Get torch_geometric edge_index format
    edge_index = torch.tensor(
        np.stack([edges_df["SRC"].to_numpy(), edges_df["DST"].to_numpy()])
    )

    # Make edge_index undirected
    edge_index = torch_geometric.utils.to_undirected(edge_index)

    # Convert edge_index back to pandas DataFrame
    edges_df = pd.DataFrame(edge_index.numpy().T, columns=["SRC", "DST"])

    del edge_index

    # Map stat['FIPS'].unique() to [0, ..., num_nodes]
    fips_map = {fips: i for i, fips in enumerate(stat["FIPS"].unique())}
    stat["FIPS"] = stat["FIPS"].map(fips_map)

    # Map edges_df['SRC'] and edges_df['DST'] to [0, ..., num_nodes]
    edges_df["SRC"] = edges_df["SRC"].map(fips_map)
    edges_df["DST"] = edges_df["DST"].map(fips_map)

    # Get torch_geometric edge_index format
    edge_index = torch.tensor(
        np.stack([edges_df["SRC"].to_numpy(), edges_df["DST"].to_numpy()])
    )

    # Remove isolated nodes (Note: this function maps the nodes to [0, ..., num_nodes] automatically)
    edge_index, _, mask = torch_geometric.utils.remove_isolated_nodes(
        edge_index
    )

    # Conver mask to index
    index = np.arange(mask.size(0))[mask]
    stat = stat.iloc[index]
    stat = stat.reset_index(drop=True)

    # Get new values for FIPS from current index
    # To understand why please print stat.iloc[[516, 517, 518, 519, 520]] for 2012 year
    # Basically the FIPS values has been shifted
    stat["FIPS"] = stat.reset_index()["index"]

    # Create Election variable
    stat["Election"] = (stat["DEM"] - stat["GOP"]) / (
        stat["DEM"] + stat["GOP"]
    )

    # Drop DEM and GOP columns and FIPS
    stat = stat.drop(columns=["DEM", "GOP", "FIPS"])

    # Prediction col
    x_col = list(stat.columns)
    x_col.remove(y_col)

    x = torch.tensor(stat[x_col].to_numpy(), dtype=torch.float32)
    y = torch.tensor(stat[y_col].to_numpy(), dtype=torch.float32)

    data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_index)

    return data


def load_hypergraph_pickle_dataset(data_dir, data_name):
    """Load hypergraph datasets from pickle files.

    Parameters
    ----------
    data_dir : str
        Path to data.
    data_name : str
        Name of the dataset.

    Returns
    -------
    torch_geometric.data.Data
        Hypergraph dataset.
    """
    data_dir = osp.join(data_dir, data_name)

    # Load node features:

    with open(osp.join(data_dir, "features.pickle"), "rb") as f:
        features = pickle.load(f)
        features = features.todense()

    # Load node labels:
    with open(osp.join(data_dir, "labels.pickle"), "rb") as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f"number of nodes:{num_nodes}, feature dimension: {feature_dim}")

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # Load hypergraph.
    with open(osp.join(data_dir, "hypergraph.pickle"), "rb") as f:
        # Hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f"number of hyperedges: {len(hypergraph)}")

    edge_idx = 0  # num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph:
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    # check that every node is in some hyperedge
    if len(np.unique(node_list)) != num_nodes:
        # add self hyperedges to isolated nodes
        isolated_nodes = np.setdiff1d(
            np.arange(num_nodes), np.unique(node_list)
        )

        for node in isolated_nodes:
            node_list += [node]
            edge_list += [edge_idx]
            edge_idx += 1
            hypergraph[f"Unique_additonal_he_{edge_idx}"] = [node]

    edge_index = np.array([node_list, edge_list], dtype=int)
    edge_index = torch.LongTensor(edge_index)

    data = Data(
        x=features,
        x_0=features,
        edge_index=edge_index,
        incidence_hyperedges=edge_index,
        y=labels,
    )

    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )

    n_x = num_nodes
    num_class = len(np.unique(labels.numpy()))

    # Add parameters to attribute
    data.n_x = n_x
    data.num_hyperedges = len(hypergraph)
    data.num_class = num_class

    data.incidence_hyperedges = torch.sparse_coo_tensor(
        data.edge_index,
        values=torch.ones(data.edge_index.shape[1]),
        size=(data.num_nodes, data.num_hyperedges),
    )

    # Print some info
    print("Final num_hyperedges", data.num_hyperedges)
    print("Final num_nodes", data.num_nodes)
    print("Final num_class", data.num_class)

    return data, data_dir
