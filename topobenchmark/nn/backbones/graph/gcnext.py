"""Class implementing the GCNext model (https://ojs.aaai.org/index.php/AAAI/article/view/28375)."""

from typing import ClassVar

import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import nn


class GCNext(nn.Module):
    """GCNext model from AAAI 2024 paper (https://ojs.aaai.org/index.php/AAAI/article/view/28375).

    This model processes motion sequences represented as graphs, where nodes represent
    joints and edges represent skeletal connections. It uses dynamic routing to learn
    different types of graph convolutions. Implementation has been minimally altered to handle sparse (edge_list) input
    and torch_geometric batching convention.

    Parameters
    ----------
    config : dict
        Configuration containing:
        - motion_dataset: Settings for input data dimensions
        - motion_mlp: Settings for the dynamic layers
    dyna_idx : tuple of int, optional
        Start and end indices for which layers should use dynamic routing.
        For example, (2, 4) means layers 2-4 will use dynamic routing.
    """

    def __init__(self, config, dyna_idx=None):
        super().__init__()

        self.config = config
        self.out_channels = 7

        self.n_frames = config.motion_dataset.n_frames
        self.n_joints = config.motion_dataset.n_joints
        self.n_channels = config.motion_dataset.n_channels

        self.n_nodes_per_frame = self.n_joints * self.n_channels

        self.n_nodes = self.n_frames * self.n_joints * self.n_channels

        # seq = 50  # love hard coding

        self.arr0 = Rearrange("b n d -> b d n")
        self.arr1 = Rearrange("b d n -> b n d")

        self.dynamic_layers = TransGraphConvolution(config)

        self.temporal_fc_in = True  # False  # config.motion_fc_in.temporal_fc
        self.temporal_fc_out = False  # config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            # if true then layer is 50x50
            self.motion_fc_in = nn.Linear(self.n_frames, self.n_frames)
        else:
            # else its 66x66
            self.motion_fc_in = nn.Linear(
                self.n_nodes_per_frame, self.n_nodes_per_frame
            )  # nn.Linear(66,66)
            # then this?
            self.in_weight = nn.Parameter(torch.eye(50, 50))
            # initialised as identity, ie, no temporal relationships
            # then model can learn to mix info from diff frames

        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(
                self.n_frames,
                self.n_frames,
            )
        else:
            self.motion_fc_out = nn.Linear(
                self.n_nodes_per_frame, self.n_nodes_per_frame
            )  # nn.Linear(66,66)
            self.out_weight = nn.Parameter(torch.eye(50, 50))

        self.reset_parameters()

        # this is the layer that chooses which convolution
        self.mlp = nn.Parameter(torch.empty(50, 4))
        nn.init.xavier_uniform_(self.mlp, gain=1e-8)

        # idk this
        self.dyna_idx = dyna_idx

    def reset_parameters(self):
        """Idk."""
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, x, edge_index, **kwargs):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch*frames*joints*channels, 1].
        edge_index : torch.Tensor
            Edge indices; this implementation assumes fully connected & doesn't care.
        **kwargs : optional
            Optional additional arguments including edge_weight.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        # Debug input at top level

        tau = 1

        batch_size = x.size(0) // self.n_nodes

        # Reshape to expected format (256, 50, 66) (batch_size, n_frames, n_nodes)
        motion_input = x.view(
            batch_size, self.n_frames, self.n_nodes_per_frame
        )

        # We don't care about edge_index at all.
        # This is sketchy, but fine.

        # Original processing

        # Step 1:
        # Processes the raw input motion sequence
        # Helps learn initial feature representations
        # Applied before the graph convolution layers

        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            # Transform across time
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            # Transform across features (66,66) with temporal weighting
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)  # now its (bs, 66, 50)
            motion_feats = torch.einsum(
                "bvt,tj->bvj", motion_feats, self.in_weight
            )
        # motion_feats = motion_input

        # Shape of motion_feats is now (batch_size, 66, 50)
        # Reshape to be batch 3D blocks (bs, 22, 3, 50)
        motion_feats = motion_feats.reshape(
            batch_size, self.n_joints, self.n_channels, self.n_frames
        )

        # Process through dynamic layers
        # Process through dynamic layers
        for i in range(len(self.dynamic_layers.layers)):
            if_make_dynamic = self.dyna_idx[0] <= i <= self.dyna_idx[1]
            motion_feats = self.dynamic_layers.layers[i](
                motion_feats, self.mlp, if_make_dynamic, tau
            )

        # Reshape to be batch old way (bs, 22*3, 50)
        motion_feats = motion_feats.reshape(
            batch_size, self.n_nodes_per_frame, self.n_frames
        )

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = torch.einsum(
                "btv,tj->bjv", motion_feats, self.out_weight
            )

        # Reshape back to torch_geometric format [batch*num_nodes, features]
        motion_feats = motion_feats.reshape(-1, 1)

        return motion_feats


class GCBlock(nn.Module):
    """Graph convolution block combining multiple types of convolutions.

    Parameters
    ----------
    n_joints : int
        Number of joints (dim1).
    n_channels : int
        Number of chanels (dim2).
    n_frames : int
        Number of frames (dim3).
    use_norm : bool
        Whether to use normalisation layer.
    use_spatial_fc : bool
        Whether to use spatial fully connected layer or temporal.
    layernorm_axis : str
        Which axis to normalize over ('spatial', 'temporal', or 'all').
    use_skeletal_hyperedges : bool
            Whether to include limb hyperedges in skeletal convolution.
    """

    def __init__(
        self,
        n_joints: int,
        n_channels: int,
        n_frames: int,
        use_norm: bool = True,
        use_spatial_fc: bool = False,
        layernorm_axis: str = "spatial",
        use_skeletal_hyperedges: bool = False,
    ):
        super().__init__()

        self.n_joints = n_joints
        self.n_channels = n_channels
        self.n_frames = n_frames

        # Initialize different convolution types
        self.skeletal_conv = SkeletalConvolution(
            use_hyperedges=use_skeletal_hyperedges
        )
        self.temporal_conv = TemporalConvolution(n_frames=n_frames)
        self.coordinate_conv = JointCoordinateConvolution()
        self.temp_joint_conv = TemporalJointConvolution(
            n_joints=n_joints, n_channels=n_channels, n_frames=n_frames
        )

        # Update layer - either spatial or temporal fully-connected
        if use_spatial_fc:
            self.fc = nn.Linear(n_joints * n_channels, n_joints * n_channels)
            self.arr0 = Rearrange("b j c t -> b t (j c)")
            self.arr1 = Rearrange(
                "b t (j c) -> b j c t", j=n_joints, c=n_channels
            )
        else:
            self.fc = nn.Linear(n_frames, n_frames)
            self.arr0 = Rearrange("b j c t -> b (j c) t")
            self.arr1 = Rearrange(
                "b (j c) t -> b j c t", j=n_joints, c=n_channels
            )

        self.norm = (
            self._create_norm_layer(layernorm_axis)
            if use_norm
            else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        nn.init.xavier_uniform_(self.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc.bias, 0)

    def _create_norm_layer(self, axis: str) -> nn.Module:
        """Create appropriate normalisation layer.

        Parameters
        ----------
        axis : str
            Which axis to normalize over ('spatial', 'temporal', or 'all').

        Returns
        -------
        nn.Module
            Normalisation layer.
        """
        if axis == "spatial":
            # Normalise each frame independently
            return nn.GroupNorm(num_groups=1, num_channels=self.n_joints)
        elif axis == "temporal":
            # Normalise each joint-channel pair independently
            return nn.LayerNorm([self.n_frames])
        elif axis == "all":
            # Normalise everything
            return nn.LayerNorm(
                [self.n_joints, self.n_channels, self.n_frames]
            )
        else:
            raise ValueError(f"Unknown normalization axis: {axis}")

    def forward(
        self,
        x: torch.Tensor,
        mlp: torch.Tensor,
        if_make_dynamic: bool,
        tau: float,
    ) -> torch.Tensor:
        """Combine different graph convolutions using dynamic routing.

        Parameters
        ----------
        x : torch.Tensor [batch, vertices, time]
            Input features.
        mlp : torch.Tensor
            MLP weights for dynamic routing.
        if_make_dynamic : bool
            Whether to use dynamic routing.
        tau : float
            Temperature for Gumbel softmax.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """

        x1 = self.skeletal_conv(x)
        x2 = self.temporal_conv(x)
        x3 = self.coordinate_conv(x)
        x4 = self.temp_joint_conv(x)

        # Dynamic routing
        b, j, c, t = x.shape
        x_reshaped = x.reshape(b, j * c, t)
        prob = torch.einsum("bj,jk->bk", x_reshaped.mean(1), mlp)
        gate = (
            nn.functional.gumbel_softmax(prob, tau=tau, hard=True)
            if if_make_dynamic
            else torch.tensor([1.0, 0.0, 0.0, 0.0])
            .unsqueeze(0)
            .expand(x.shape[0], -1)
            .to(x.device)
        )

        # Combine convolution results
        # Kinda sketchily looks like they're always including x1...???
        # Biases model towards skeletal convolution
        x_opts = torch.stack([torch.zeros_like(x1), x2, x3, x4], dim=1)
        x_combined = torch.einsum("bo,bojct->bjct", gate, x_opts)

        # Send through fully connected layer and normalise
        squished_x = self.arr0(x1 + x_combined)
        squished_x = self.fc(squished_x)
        x_out = self.arr1(squished_x)
        x_out = self.norm(x_out)

        # Residual connection
        return x + x_out


class TransGraphConvolution(nn.Module):
    """Blah.

    Blah.

    Parameters
    ----------
    config : dict
        Configuration containing:
        - motion_dataset:
            - n_joints
            - n_channels
            - n_frames
        - motion_mlp:
            - use_norm: blah
            - use_spatial_fc: blah
            - num_layers: blah
            - layernorm_axis: blah
            - use_skeletal_hyperedges: blkah
    """

    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                GCBlock(
                    n_joints=config.motion_dataset.n_joints,
                    n_channels=config.motion_dataset.n_channels,
                    n_frames=config.motion_dataset.n_frames,
                    use_norm=config.motion_mlp.with_normalization,
                    use_spatial_fc=config.motion_mlp.spatial_fc_only,
                    layernorm_axis=config.motion_mlp.norm_axis,
                    use_skeletal_hyperedges=config.motion_mlp.use_skeletal_hyperedges,
                )
                for i in range(config.motion_mlp.num_layers)
            ]
        )

    def forward(self, x, mlp, if_make_dynamic, tau):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch*frames*joints*channels, 1].
        mlp : str
            Blan.
        if_make_dynamic : str
            Bhal.
        tau : str
            Blah.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        x = self.layers(x, mlp, if_make_dynamic, tau)
        return x


###############################################################
#### Different types of graph convolutions used in GCNext. ####
###############################################################
class BaseGraphConvolution(nn.Module):
    """Base class for all 3D convolution types.

    Parameters
    ----------
    n_dim1 : int
        Size of first dimension (ie, n_joints).
    n_dim2 : int
        Size of second dimension (ie, n_channels).
    n_dim3 : int
        Size of third dimension (ie, n_frames).
    """

    def __init__(self, n_dim1: int, n_dim2: int, n_dim3: int):
        super().__init__()

        self.n_dim1 = n_dim1
        self.n_dim2 = n_dim2
        self.n_dim3 = n_dim3

        self.weights = None
        self.mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution along desired dimensions.

        Parameters
        ----------
        x : torch.Tensor [batch, dim1, dim2, dim3]
            Input features.

        Returns
        -------
        torch.Tensor [batch, dim1, dim2, dim3]
            Features after convolution.
        """
        raise NotImplementedError


class SkeletalConvolution(nn.Module):
    """Convolution along skeletal connections between joints in H36MSkeleton.

    Parameters
    ----------
    use_hyperedges : bool
        Whether to include limb hyperedges in convolution.
    """

    def __init__(self, use_hyperedges=False):
        super().__init__()

        self.skl = H36MSkeleton()
        if use_hyperedges:
            self.register_buffer("skl_mask", self.skl.skl_mask_hyper)
        else:
            self.register_buffer("skl_mask", self.skl.skl_mask)

        self.weights = nn.Parameter(
            torch.randn(self.skl.NUM_JOINTS, self.skl.NUM_JOINTS),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial convolution with skeleton mask.

        Parameters
        ----------
        x : torch.Tensor [batch, n_joints, n_channels, n_frames]
            Input features.

        Returns
        -------
        torch.Tensor [batch, n_joints, n_channels, n_frames]
            Features after spatial convolution with skeleton mask.
        """

        masked_weights = self.weights.mul(self.skl_mask)  # (22,22)
        # (22,22) x (batch,22,3,50) -> (batch,22,3,50)
        return torch.einsum("vj, bjct->bvct", masked_weights, x)


class TemporalConvolution(nn.Module):
    """Convolution along temporal connections.

    Parameters
    ----------
    n_frames : int
        Number of frames, length of time dimension.
    """

    def __init__(self, n_frames: int):
        super().__init__()

        self.weights = nn.Parameter(torch.zeros(n_frames, n_frames))

        # Only allows connections to adjacent time frames while preventing self-connections
        local_temporal_mask = torch.tril(
            torch.ones(n_frames, n_frames), 1
        ) * torch.triu(torch.ones(n_frames, n_frames), -1)
        local_temporal_mask.diagonal().fill_(0)
        self.register_buffer("local_temporal_mask", local_temporal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal graph convolution.

        Parameters
        ----------
        x : torch.Tensor [batch, n_joints, n_channels, n_frames]
            Input features.

        Returns
        -------
        torch.Tensor [batch, n_joints, n_channels, n_frames]
            Features after temporal convolution.
        """
        b, j, c, t = x.shape

        x = x.reshape(b, j * c, t)

        masked_weights = self.weights.mul(self.local_temporal_mask)

        # (n_frames, n_frames) x (batch, joint*channel, n_frames) -> (batch, joint*channel, n_frames)
        # (50, 50) x (256, 22, 3, 50) -> (256, 66, 50)
        x_out = torch.einsum(
            "ft,bnt->bnf",
            masked_weights,
            x,
        )
        return x_out.reshape(b, j, c, t)


class JointCoordinateConvolution(nn.Module):
    """Convolution between joints and coordinates."""

    def __init__(self):
        super().__init__()
        self.skl = H36MSkeleton()
        self.weights = nn.Parameter(
            torch.zeros(
                self.skl.NUM_JOINTS,
                self.skl.NUM_CHANNELS,
                self.skl.NUM_CHANNELS,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply joint-coordinate graph convolution.

        Parameters
        ----------
        x : torch.Tensor [batch, n_joints, n_channels, n_frames]
            Input features.

        Returns
        -------
        torch.Tensor [batch, n_joints, n_channels, n_frames]
            Features after joint-coordinate convolution.
        """
        return torch.einsum("jkc,bjct->bjkt", self.weights, x)


class TemporalJointConvolution(nn.Module):
    """Convolution combining temporal and joint relationships.

    Parameters
    ----------
    n_joints : int
        Number of joints (22).
    n_channels : int
        Number of channels per joint (3).
    n_frames : int
        Number of frames, length of time dimension.
    """

    def __init__(self, n_joints: int, n_channels: int, n_frames: int):
        super().__init__()
        self.weights = nn.Parameter(
            torch.zeros(n_joints * n_channels, n_frames, n_frames)
        )

        # Add this to the base class!!
        local_temporal_mask = torch.tril(
            torch.ones(n_frames, n_frames), 1
        ) * torch.triu(torch.ones(n_frames, n_frames), -1)
        local_temporal_mask.diagonal().fill_(0)
        self.register_buffer("local_temporal_mask", local_temporal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal-joint graph convolution.

        Parameters
        ----------
        x : torch.Tensor [batch, n_joints, n_channels, n_frames]
            Input features.

        Returns
        -------
        torch.Tensor [batch, n_joints, n_channels, n_frames]
            Features after temporal-joint convolution.
        """
        b, j, c, t = x.shape

        x = x.reshape(b, j * c, t)

        # (66, 50, 50) x (50, 50) -> # (66, 50, 50)
        masked_weights = self.weights.mul(self.local_temporal_mask)

        # (66, n_frames, n_frames) x (batch, 66, n_frames) -> (batch, n_frames, 66)
        # (66, 50, 50) x (batch, 66, 50) -> (batch, 50, 66)
        x_out = torch.einsum("nft,bnt->bnf", masked_weights, x)
        return x_out.reshape(b, j, c, t)


class H36MSkeleton:
    r"""Class for connections in Human3.6M Skeleton.

    Attributes
    ----------
    NUM_JOINTS (int): Number of joints in skeleton.
    NUM_CHANNELS (int): Number of channels per joint.
    USED_JOINT_INDICES (np.array[np.int64]): Numpy array containing relevant joint indices.
    BONE_LINKS (list[tup[int]]): ONE-INDEXED list defining bone connections.
    LIMB_LINKS (list[list[int]]): List defining limbs.
    """

    NUM_JOINTS: ClassVar = 22
    NUM_CHANNELS: ClassVar = 3

    # Labels from here: https://github.com/qxcv/pose-prediction/blob/master/H36M-NOTES.md
    USED_JOINT_INDICES: ClassVar = np.array(
        [
            2,  # RHip 1
            3,  # RKnee 2
            4,  # RAnkle 3
            5,  # RFoot 4
            7,  # LHip 5
            8,  # LKnee 6
            9,  # LAnkle 7
            10,  # LFoot 8
            12,  # Pelvis? 9
            13,  # Torso 10
            14,  # Base of neck (same as 17, 25?) 11
            15,  # Head low 12
            17,  # Base of neck (same as 14, 25?)
            18,  # LShoulder 14
            19,  # LElbow 15
            21,  # LWrist 16
            22,  # LHand 17
            25,  # Base of neck (same as 14, 17?)
            26,  # RShoulder 19
            27,  # RElbow 20
            29,  # RWrist 21
            30,  # RHand 22
        ]
    )

    BONE_LINKS: ClassVar = [
        (1, 2),  # WHY IS THIS ONE INDEXED!??!?!??!?!
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (7, 8),
        (1, 9),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (10, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (15, 17),
        (10, 18),
        (18, 19),
        (19, 20),
        (20, 21),
        (20, 22),
    ]

    LIMB_LINKS: ClassVar = [
        [1, 2, 3, 4],  # Right leg?
        [5, 6, 7, 8],  # Left leg?
        [9, 10, 11, 12, 13],  # torso?
        [14, 15, 16, 17],  # Left arm?
        [19, 20, 21, 22],  # Right arm?
    ]

    def __init__(self):
        r"""H36M skeleton with 22 joints."""

        self.bone_list = self.generate_bone_list()
        self.skl_mask = self.generate_skl_mask()

        self.skl_mask_hyper = self.generate_skl_mask_hyper()

    def compute_flat_index(self, t, j, c):
        r"""Compute flat index for motion matrix of shape (T,J,C).

        Parameters
        ----------
        t : int
            Time index in 3d matrix.
        j : int
            Joint index in 3d matrix.
        c : int
            Channel index in 3d matrix.

        Returns
        -------
        int
            Flat index in T*J*C vector.
        """
        return (
            t * self.NUM_JOINTS * self.NUM_CHANNELS + j * self.NUM_CHANNELS + c
        )

    def generate_bone_list(self):
        r"""Generate bones in H36M skeleton with 22 joints.

        Returns
        -------
        list[tup[int]]
            Edge list with bone links and self links.
        """
        self_links = [(i, i) for i in range(self.NUM_JOINTS)]
        return self_links + [(i - 1, j - 1) for (i, j) in self.BONE_LINKS]

    def generate_skl_mask(self):
        r"""Get skeleton mask for H36M skeleton with 22 joints.

        Returns
        -------
        list[tup[int]]
            Edge list with bone links and self links.
        """
        # Create adjacency matrix
        skl_mask = torch.zeros(
            self.NUM_JOINTS, self.NUM_JOINTS, requires_grad=False
        )
        for i, j in self.bone_list:
            skl_mask[i, j] = 1
            skl_mask[j, i] = 1
        return skl_mask

    def generate_skl_mask_hyper(self):
        r"""Get hyperedge skeleton mask for H36M skeleton with 22 joints.

        Returns
        -------
        list[tup[int]]
            Edge list with limb links, bone links and self links.
        """
        # Create adjacency matrix
        skl_mask = torch.zeros(
            self.NUM_JOINTS, self.NUM_JOINTS, requires_grad=False
        )
        for i, j in self.bone_list:
            skl_mask[i, j] = 1
            skl_mask[j, i] = 1

        for limb in self.LIMB_LINKS:
            # Connect all joints within each limb to each other
            for i in limb:
                for j in limb:
                    if (
                        i != j
                    ):  # Skip self connections as they're already handled
                        skl_mask[i - 1, j - 1] = (
                            1  # -1 since joints are 1-indexed in LIMB_LINKS
                        )
        return skl_mask

    # def generate_time_edges(self, n_times):
    #     r"""Generate list of edges only through time.

    #     Parameters
    #     ----------
    #     n_times : int
    #         Number of frames to consider.

    #     Returns
    #     -------
    #     torch.tensor
    #         Time edges.
    #     """
    #     time_edges = []
    #     for c in range(self.NUM_CHANNELS):
    #         for j in range(self.NUM_JOINTS):
    #             for t1 in range(n_times):
    #                 for t2 in range(n_times):
    #                     edge = [
    #                         self.compute_flat_index(t1, j, c),
    #                         self.compute_flat_index(t2, j, c),
    #                     ]
    #                     time_edges.append(edge)
    #     return torch.tensor(time_edges).T

    # def generate_bone_edges(self, n_times):
    #     """Generate list of edges along bones across all frames.

    #     Parameters
    #     ----------
    #     n_times : int
    #         Number of frames to consider.

    #     Returns
    #     -------
    #     torch.tensor
    #         Bone edges spanning all frames, shape [2, num_edges].
    #     """
    #     bone_edges = []

    #     # For each frame
    #     for t in range(n_times):
    #         # For each channel
    #         for c in range(self.NUM_CHANNELS):
    #             # For each bone connection
    #             for j1, j2 in self.bone_list:
    #                 edge = [
    #                     self.compute_flat_index(t, j1, c),
    #                     self.compute_flat_index(t, j2, c),
    #                 ]
    #                 bone_edges.append(edge)

    #     return torch.tensor(bone_edges).T
