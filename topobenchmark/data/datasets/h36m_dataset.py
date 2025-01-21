"""Dataset class for Human3.6M dataset (http://vision.imar.ro/human3.6m/description.php)."""

import os
import os.path as osp
import shutil
import sqlite3
from typing import ClassVar

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, OnDiskDataset, extract_zip


class H36MDataset(OnDiskDataset):
    r"""Dataset class for Human3.6M dataset. Maybe use on disk dataset?.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.
    force_reload : bool
        Whether to re-process the dataset. Default is False.

    Attributes
    ----------
    URLS (dict): Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT (dict): Dictionary containing the file formats for the dataset.
    RAW_FILE_NAMES (dict): Dictionary containing the raw file names for the dataset.
    SUBJ_NAMES (list[str]): List of subjects to consider.
    VAL_SUBJ (str): Default subject for validation.
    TEST_SUBJ (str): Default subject for test.
    N_FRAMES (int): How many frames long to make each input and output.
    """

    # Taken from siMLPe paper: https://github.com/dulucas/siMLPe/tree/c92c537e833443aa55554e4f7956838746550187
    # Pre-processed from the original dataset to be .txt files
    # TODO: not sure if this is legal, so find a way to go from official website with license
    URLS: ClassVar = {
        "H36MDataset": "https://drive.google.com/file/d/15OAOUrva1S-C_BV8UgPORcwmWG2ul4Rk/view"
    }

    FILE_FORMAT: ClassVar = {
        "H36MDataset": "zip",
    }

    RAW_FILE_NAMES: ClassVar = {}

    SUBJ_NAMES: ClassVar = ["S1", "S5", "S6", "S7", "S8", "S9", "S11", "S11"]
    VAL_SUBJ: ClassVar = "S11"
    TEST_SUBJ: ClassVar = "S5"
    N_FRAMES: ClassVar = 50

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        self.parameters = parameters
        self.force_reload = force_reload

        db_root = osp.join(root, name)
        proc_db = osp.join(db_root, "processed")

        os.makedirs(proc_db, exist_ok=True)

        # Initialize SQLite connection and cursor
        self.db_path = osp.join(proc_db, "metadata.db")
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

        # Create table if it doesn't exist
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS data (
                idx INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL,
                size INTEGER NOT NULL
            )
        """)
        self.connection.commit()

        super().__init__(
            root,
        )

        self.split_idx = self.generate_default_split_idx()
        self.train_idx = set(self.split_idx["train"])
        self.val_idx = set(self.split_idx["valid"])
        self.test_idx = set(self.split_idx["test"])

    def generate_default_split_idx(self):
        """Return the default split index for H3.6M Dataset.

        Returns
        -------
        dict:
            Dictionary containing the train, validation and test indices with keys "train", "valid", and "test".
        """
        print("Generating split index...")

        # Get all filenames and indices from database
        self.cursor.execute("SELECT idx, file_name FROM data")
        rows = self.cursor.fetchall()

        # Initialize empty lists for each split
        train_idx = []
        val_idx = []
        test_idx = []

        # Iterate through rows and assign to splits based on subject
        for idx, filename in rows:
            # Extract subject name from filename (e.g. "S8" from "S8_graph_12229.pt")
            subject = filename.split("_")[0]

            if subject == self.TEST_SUBJ:
                test_idx.append(idx)
            elif subject == self.VAL_SUBJ:
                val_idx.append(idx)
            elif subject in self.SUBJ_NAMES:
                train_idx.append(idx)
            else:
                raise Exception(
                    f"Subject not found! <subject={subject}> <filename={filename}"
                )

        # Convert to numpy arrays
        split_idx = {
            "train": np.array(train_idx),
            "valid": np.array(val_idx),
            "test": np.array(test_idx),
        }

        return split_idx

    def __len__(self) -> int:
        """Return the number of graphs in the dataset.

        Returns
        -------
        int
            The number of graphs in the dataset.
        """
        self.cursor.execute("SELECT COUNT(*) FROM data")
        return self.cursor.fetchone()[0]

    def __repr__(self) -> str:
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        self.processed_root = osp.join(
            self.root,
            self.name,
        )
        return osp.join(self.processed_root, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Return all processed file names.

        Returns
        -------
        list[str]
            List of all processed file names.
        """
        return [
            filename
            for filename in os.listdir(self.processed_dir)
            if filename.endswith(".pt")
        ]

    def get(self, idx: int) -> Data:
        """Load a single graph from disk using the database.

        Parameters
        ----------
        idx : int
            Index of the graph to load.

        Returns
        -------
        Data
            The loaded graph.
        """
        # Query the database to get the filename for this index
        self.cursor.execute("SELECT file_name FROM data WHERE idx = ?", (idx,))
        result = self.cursor.fetchone()

        if result is None:
            raise IndexError(f"No data found for index {idx}")

        filename = result[0]
        filepath = os.path.join(self.processed_dir, filename)

        # Add train/val/test mask to data object.
        # This probably isn't the best place to do it.
        data = torch.load(filepath, weights_only=False)

        n_nodes = data.x.shape[0]
        if idx in self.train_idx:
            # TODO: Compute loss on next 10 frames in training.
            data.train_mask = torch.arange(n_nodes * 2).long()
            data.val_mask = torch.Tensor([0]).long()
            data.test_mask = torch.Tensor([0]).long()

        if idx in self.val_idx:
            data.train_mask = torch.Tensor([0]).long()
            # TODO: Compute loss on next 25 frames in eval.
            data.val_mask = torch.arange(n_nodes * 2).long()
            data.test_mask = torch.Tensor([0]).long()

        if idx in self.test_idx:
            data.train_mask = torch.Tensor([0]).long()
            data.val_mask = torch.Tensor([0]).long()
            #  TODO: Compute loss on next 25 frames in eval.
            data.test_mask = torch.arange(n_nodes * 2).long()

        return data

    def download(self) -> None:
        r"""Download the dataset from a URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """
        # Step 1: download data from the source
        # self.url = self.URLS[self.name]
        # download_file_from_drive(
        #     file_link=self.url,
        #     path_to_save=self.raw_dir,
        #     dataset_name=self.name,
        #     file_format=self.FILE_FORMAT[self.name],
        # )

        # Step 1 isn't working because it is downloading html with the Antivirus Scan Warning instead.
        # Instead:
        #   1) Dowload preprocessed data from here: https://drive.google.com/file/d/15OAOUrva1S-C_BV8UgPORcwmWG2ul4Rk/view
        #       As per siMPLE paper (https://github.com/dulucas/siMLPe/tree/c92c537e833443aa55554e4f7956838746550187)
        #   2) Rename to be H36MDataset.zip
        #   3) Manually plop the zip in the right folder.
        #       scp H36MDataset.zip [your path]/TopoBenchmark/datasets/graph/motion/H36MDataset/raw

        folder = self.raw_dir
        compressed_data_filename = f"{self.name}.{self.FILE_FORMAT[self.name]}"
        compressed_data_path = osp.join(folder, compressed_data_filename)

        if os.path.isfile(compressed_data_path):
            # Step 2: extract file
            print("Zip file exists. Extracting data zip...")
            extract_zip(compressed_data_path, folder)
            os.unlink(compressed_data_path)  # Delete zip file

            # Step 3: move the extracted files to the folder with corresponding name
            # Move files from osp.join(folder, name_download) to folder
            for subject_dir in os.listdir(
                osp.join(
                    folder, "h36m"
                )  # hard coded here because this is the name in the google drive folder; should be self.name
            ):
                for file in os.listdir(osp.join(folder, "h36m", subject_dir)):
                    if file.endswith("ipynb"):
                        continue
                    shutil.move(
                        osp.join(folder, "h36m", subject_dir, file),
                        osp.join(folder, f"{subject_dir}-{file}"),
                    )

            # Delete osp.join(folder, self.name) dir
            shutil.rmtree(osp.join(folder, "h36m"))
            print("Done. Zip file removed.")
        else:
            print("Data already extracted. Skipping extraction.")

    def process(self) -> None:
        r"""Handle the data for the dataset.

        This method loads the Human3.6M dataset,
        loads it into basic graph form, and saves this preprocessed data.

        Will not run if self.force_reload is False and files exist.

        A lot of this is using functions copied from the siMLPe paper.
        """
        print("Loading and processing data...")

        for subj_name in self.SUBJ_NAMES:
            print(f"Processing subject {subj_name}")
            # Step 1: extract the data
            h36m_raw_xyz_motion_matrices = self.load_raw_xyz_motion_matrices(
                subj_name
            )
            motion_inputs, motion_targets = self.process_input_target_pairs(
                h36m_raw_xyz_motion_matrices, debug=False
            )

            # Step 2: define connections and transform into graphs
            motions = self.transform_into_graph_data_objects(
                motion_inputs, motion_targets
            )

            # Step 3: save each graph individually
            print(f"\tSaving {subj_name} graphs...")
            for idx, data in enumerate(motions):
                # Step 3b: save data to file
                torch.save(
                    data,
                    os.path.join(
                        self.processed_dir, f"{subj_name}_graph_{idx}.pt"
                    ),
                )
            print(f"\tDone with {subj_name}.")

        # Step 4: Create database.
        print("Creating database...")
        self._create_database()
        print("Done processing.")

    def _create_database(self) -> None:
        """Create the SQLite database with metadata for all saved .pt files."""
        # Get list of all .pt files
        files = sorted(
            [f for f in os.listdir(self.processed_dir) if f.endswith(".pt")]
        )

        # Create database entries
        data = []
        for idx, filename in enumerate(files):
            filepath = os.path.join(self.processed_dir, filename)
            size = os.path.getsize(filepath)
            data.append((idx, filename, size))

        # Store in database
        self.cursor.executemany(
            "INSERT INTO data(idx, file_name, size) VALUES (?, ?, ?)", data
        )
        self.connection.commit()

    def get_processed_path_for_subj(self, subj_name):
        r"""Get path where processed subject-wise data is stored.

        Parameters
        ----------
        subj_name : str
            Subject name.

        Returns
        -------
        str
            Path where subj_name's graph data is stored.
        """
        path_parts = self.processed_paths[0].split(".")
        return "".join([path_parts[0], subj_name, ".", path_parts[1]])

    def transform_into_graph_data_objects(self, motion_inputs, motion_targets):
        r"""Get path where processed subject-wise data is stored.

        Parameters
        ----------
        motion_inputs : list[torch.tensor]
            List of motion matrix inputs of shape (input_length, n_joints, n_channels).
        motion_targets : list[torch.tensor]
            List of motion matrix labels of shape (target_length, n_joints, n_channels).

        Returns
        -------
        list[Data]
            List of graph objects.
        """
        # Step 1: Create edges. These are the same for all graphs.
        #       We want a superset of all possible edge indices
        #       so the model can choose which to care about.
        # For efficiency, though, perhaps could be good to prune this...
        # TODO: Make which edges are created a parameter to pass into config!
        fully_connected_edges = torch.combinations(
            torch.arange(self.N_FRAMES * 22 * 3), r=2
        ).t()  # Shape: [2, num_edges] where num_edges = (n*(n-1))/2
        empty_edges = torch.zeros((2, 0), dtype=torch.long)  # Shape: [2, 0]

        # Step 2: turn them into torch_geometric.data Data objects
        print("\tConverting to graph objects...")
        motions = []

        for i in range(len(motion_inputs)):
            input_motion_matrix = motion_inputs[i]  # shape = [50, 22, 3]
            target_motion_matrix = motion_targets[i]  # shape = [50, 22, 3]

            # need to flatten so we have: large(time), med(joints), small(channels)
            # so if there are 3 times and 4 joints and 2 channels (ignore the brackets)
            #   [
            #       [j0c0t0, j0c1t0]
            #       [j1c0t0, j1c1t0]
            #       [j2c0t0, j2c1t0]
            #       [j3c0t0, j3c1t0]
            #   ]
            #   [
            #       [j0c0t1, j0c1t1]
            #       [j1c0t1, j1c1t1]
            #       [j2c0t1, j2c1t1]
            #       [j3c0t1, j3c1t1]
            #   ]
            #   [
            #       [j0c0t2, j0c1t2]
            #       [j1c0t2, j1c1t2]
            #       [j2c0t2, j2c1t2]
            #       [j3c0t2, j3c1t2]
            #   ]
            # just read this in order!

            # Step 3: Flatten motion_matrix; node for each time, joint, channel combo.
            flat_input = torch.reshape(
                input_motion_matrix, (-1,)
            )  # shape 50*22*3 = 3300
            flat_target = torch.reshape(
                target_motion_matrix, (-1,)
            )  # shape 50*22*3 = 3300

            # Step 4: Create fully-conencted graph Data objects.
            motion_graph = Data(
                x=flat_input.unsqueeze(1),
                y=flat_target,
                edge_index=empty_edges,
            )
            motions.append(motion_graph)
        return motions

    def load_raw_xyz_motion_matrices(self, subj_name):
        r"""Load raw motion data.

        This method loads the Human3.6M dataset.

        A lot of this is using functions copied from the siMLPe paper.

        Parameters
        ----------
        subj_name : str
            Subject name.

        Returns
        -------
        torch.tensor
            Raw motion matrices from files TODO WHAT IS THIS.
        """
        # Relevant data files for processing. There are -1 and -2 ones
        # and I don't know why. Using the first for fun?
        file_list = [
            osp.join(self.raw_dir, filename)
            for filename in os.listdir(self.raw_dir)
            if filename.startswith(subj_name) and filename.endswith("1.txt")
        ]

        # Read and process each text file; copied from GCNext code.
        raw_motion_matrices = []
        for path in file_list:
            with open(path) as f:
                info = f.readlines()

            pose_info = []
            for line in info:
                line = line.strip().split(",")
                if len(line) > 0:
                    pose_info.append(np.array([float(x) for x in line]))
            pose_info = np.array(pose_info)

            T = pose_info.shape[0]
            pose_info = pose_info.reshape(-1, 33, 3)
            pose_info[:, :2] = 0
            pose_info = pose_info[:, 1:, :].reshape(-1, 3)
            pose_info = expmap2rotmat_torch(
                torch.tensor(pose_info).float()
            ).reshape(T, 32, 3, 3)

            xyz_info = rotmat2xyz_torch(pose_info)
            xyz_info = xyz_info[:, H36MSkeleton.USED_JOINT_INDICES, :]

            raw_motion_matrices.append(xyz_info)

        return raw_motion_matrices

    def process_input_target_pairs(
        self,
        raw_xyz_motion_matrices,
        sample_rate=2,  # space issues
        debug=False,
    ):
        r"""Create input-target pairs from raw motion matrices.

        Parameters
        ----------
        raw_xyz_motion_matrices : list[torch.tensor]
            GA.
        sample_rate : int
            Sample rate to reduce temporal resolution; should be 2 to get 25 frames per second.
        debug : bool
            Debug setting. If True will only return the first 150 input/target pairs for SPEED.

        Returns
        -------
        X : list[torch.tensor]
            List of motion matrix inputs of shape (input_length, n_joints, n_channels).
        Y : list[torch.tensor]
            List of motion matrix labels of shape (target_length, n_joints, n_channels).
        """
        input_length, target_length = self.N_FRAMES, self.N_FRAMES

        X = []
        Y = []

        debug_count = 0
        for h36m_motion_poses in raw_xyz_motion_matrices:
            n_frames = h36m_motion_poses.shape[0]
            frame_span_of_sample = (input_length + target_length) * sample_rate

            # motion doesn't have enough frames
            if n_frames < frame_span_of_sample:
                continue

            for valid_start_frame in range(
                0, n_frames - frame_span_of_sample, sample_rate
            ):
                frame_indexes = np.arange(
                    valid_start_frame,
                    valid_start_frame + frame_span_of_sample,
                    sample_rate,
                )

                motion = h36m_motion_poses[frame_indexes]

                motion_input = (motion[:input_length] / 1000).float()  # meter
                motion_target = (motion[input_length:] / 1000).float()  # meter

                X.append(motion_input)
                Y.append(motion_target)

                # Leave early to verify things and spare my soul.
                debug_count += 1
                if debug and debug_count == 150:
                    return X, Y

        return X, Y


########################
### HELPER FUNCTIONS ###
########################


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


def _some_variables():
    r"""Silly way to store variables but they did it.

    Borrowed from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L100
    We define some variables that are useful to run the kinematic tree

    Returns
    -------
    parent:
        32-long vector with parent-child relationships in the kinematic tree.
    offset:
        96-long vector with bone lenghts.
    rotInd:
        32-long list with indices into angles.
    expmapInd:
        32-long list with indices into expmap angles.
    """

    parent = (
        np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                1,
                7,
                8,
                9,
                10,
                1,
                12,
                13,
                14,
                15,
                13,
                17,
                18,
                19,
                20,
                21,
                20,
                23,
                13,
                25,
                26,
                27,
                28,
                29,
                28,
                31,
            ]
        )
        - 1
    )

    offset = np.array(
        [
            0.000000,
            0.000000,
            0.000000,
            -132.948591,
            0.000000,
            0.000000,
            0.000000,
            -442.894612,
            0.000000,
            0.000000,
            -454.206447,
            0.000000,
            0.000000,
            0.000000,
            162.767078,
            0.000000,
            0.000000,
            74.999437,
            132.948826,
            0.000000,
            0.000000,
            0.000000,
            -442.894413,
            0.000000,
            0.000000,
            -454.206590,
            0.000000,
            0.000000,
            0.000000,
            162.767426,
            0.000000,
            0.000000,
            74.999948,
            0.000000,
            0.100000,
            0.000000,
            0.000000,
            233.383263,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            121.134938,
            0.000000,
            0.000000,
            115.002227,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            151.034226,
            0.000000,
            0.000000,
            278.882773,
            0.000000,
            0.000000,
            251.733451,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            99.999627,
            0.000000,
            100.000188,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            151.031437,
            0.000000,
            0.000000,
            278.892924,
            0.000000,
            0.000000,
            251.728680,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            99.999888,
            0.000000,
            137.499922,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
        ]
    )
    offset = offset.reshape(-1, 3)

    rotInd = [
        [5, 6, 4],
        [8, 9, 7],
        [11, 12, 10],
        [14, 15, 13],
        [17, 18, 16],
        [],
        [20, 21, 19],
        [23, 24, 22],
        [26, 27, 25],
        [29, 30, 28],
        [],
        [32, 33, 31],
        [35, 36, 34],
        [38, 39, 37],
        [41, 42, 40],
        [],
        [44, 45, 43],
        [47, 48, 46],
        [50, 51, 49],
        [53, 54, 52],
        [56, 57, 55],
        [],
        [59, 60, 58],
        [],
        [62, 63, 61],
        [65, 66, 64],
        [68, 69, 67],
        [71, 72, 70],
        [74, 75, 73],
        [],
        [77, 78, 76],
        [],
    ]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmapInd


def fkl_torch(rotmat, parent, offset, rotInd, expmapInd):
    r"""Pytorch version of fkl.

    Convert joint angles to joint locations.
    batch pytorch version of the fkl() method above

    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:

    Parameters
    ----------
    rotmat : idk
        Idk.
    parent : idk
        Idk.
    offset : idk
        Idk.
    rotInd : idk
        Idk.
    expmapInd : idk
        Idk.

    Returns
    -------
    idk
        N*joint_n*3.
    """
    n = rotmat.data.shape[0]
    j_n = offset.shape[0]
    p3d = (
        torch.from_numpy(offset)
        .float()
        .to(rotmat.device)
        .unsqueeze(0)
        .repeat(n, 1, 1)
        .clone()
    )
    R = rotmat.view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            R[:, i, :, :] = torch.matmul(
                R[:, i, :, :], R[:, parent[i], :, :]
            ).clone()
            p3d[:, i, :] = (
                torch.matmul(p3d[0, i, :], R[:, parent[i], :, :])
                + p3d[:, parent[i], :]
            )
    return p3d


def expmap2rotmat_torch(r):
    r"""Convert expmap matrix to rotation.

    batch pytorch version ported from the corresponding method above

    Parameters
    ----------
    r : np.array
        Shape=N*3.

    Returns
    -------
    np.array
        Shape=N*3*3.
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = (
        torch.eye(3, 3).repeat(n, 1, 1).float().to(r.device)
        + torch.mul(
            torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1
        )
        + torch.mul(
            (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)),
            torch.matmul(r1, r1),
        )
    )
    return R


def rotmat2xyz_torch(rotmat):
    r"""Convert expmaps to joint locations.

    Parameters
    ----------
    rotmat : np.array
        Shape=N*32*3*3.

    Returns
    -------
    np.array
        Shape=N*32*3.
    """
    assert rotmat.shape[1] == 32
    parent, offset, rotInd, expmapInd = _some_variables()
    xyz = fkl_torch(rotmat, parent, offset, rotInd, expmapInd)
    return xyz
