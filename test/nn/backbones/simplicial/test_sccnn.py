"""Unit tests for SCCNN"""

import torch
from torch_geometric.utils import get_laplacian
from ...._utils.nn_module_auto_test import NNModuleAutoTest
from topobenchmark.nn.backbones.simplicial import SCCNNCustom
from topobenchmark.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


def test_SCCNNCustom(simple_graph_1):
    lifting_signed = SimplicialCliqueLifting(
            complex_dim=3, signed=True
        )
    data = lifting_signed(simple_graph_1)
    out_dim = 4
    conv_order = 1
    sc_order = 3
    laplacian_all = (
            data.hodge_laplacian_0,
            data.down_laplacian_1,
            data.up_laplacian_1,
            data.down_laplacian_2,
            data.up_laplacian_2,
        )
    incidence_all = (data.incidence_1, data.incidence_2)
    expected_shapes = [(data.x.shape[0], out_dim), (data.x_1.shape[0], out_dim), (data.x_2.shape[0], out_dim)]

    auto_test = NNModuleAutoTest([
        {
            "module" : SCCNNCustom, 
            "init": ((data.x.shape[1], data.x_1.shape[1], data.x_2.shape[1]), (out_dim, out_dim, out_dim), conv_order, sc_order),
            "forward":  ((data.x, data.x_1, data.x_2), laplacian_all, incidence_all),
            "assert_shape": expected_shapes
        },
    ])
    auto_test.run()
