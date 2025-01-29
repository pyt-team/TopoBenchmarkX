"""Visualisation callback class for human motion data."""

import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback
from mpl_toolkits.mplot3d import Axes3D


class MotionVisualizationCallback(Callback):
    """Custom Callback function to visualise human motion data.

    Note: you need to conda install imageio for this.

    Parameters
    ----------
    num_samples : int
        Todo.
    """

    def __init__(self, num_samples=1):
        super().__init__()
        self.num_samples = num_samples

        ONE_INDEXED_BONE_LINKS = [
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
        self.BONE_LINKS = [
            (x[0] - 1, x[1] - 1) for x in ONE_INDEXED_BONE_LINKS
        ]

    def plot_skeleton(self, joints, ax):
        """Plot 3D skeleton given joint positions."""
        # H36M skeleton connections (you may need to adjust these based on your joint order)

        # Plot joints
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c="b", marker="o")

        # Plot connections
        for connection in self.BONE_LINKS:
            start = joints[connection[0]]
            end = joints[connection[1]]
            ax.plot(
                [start[0], end[0]],
                [start[2], end[2]],
                [start[1], end[1]],
                "r-",
            )

    def on_train_epoch_end(self, trainer, pl_module):
        """Visualize predictions at the end of each epoch."""
        if not trainer.sanity_checking:
            print("WWOOOOOO")
            # Get a sample from validation set
            batch = next(iter(trainer.train_dataloader))

            # Move to same device as model
            # batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
            #         for k, v in batch.items()}
            # batch["x_0"] = batch["x"]
            batch.x_0 = batch.x.to(pl_module.device)

            # Get predictions
            pl_module.eval()
            with torch.no_grad():
                outputs = pl_module(batch)
            pl_module.train()

            # outputs["labels"].shape = [844800]
            ground_truth = outputs["labels"].reshape(
                256, 50, 22, 3
            )  # [B, T, J, 3]
            # outputs["x_0"].shape = [844800, 1]
            model_output = outputs["x_0"].reshape(
                256, 50, 22, 3
            )  # [B, T, J, 3]

            ground_truth_ex = ground_truth[0]
            model_output_ex = model_output[0]

            # Create directory if it doesn't exist
            os.makedirs("visualisations", exist_ok=True)

            temp_images = []
            for i in range(50):
                ground_truth_frame = ground_truth_ex[i]
                model_output_frame = model_output_ex[i]

                # scale model output to be in the same range as ground truth
                height_in_ground_truth = ground_truth_frame[:, 2]
                height_in_model_output = model_output_frame[:, 2]
                scale_factor = height_in_ground_truth.max() / height_in_model_output.max()
                model_output_frame[:, :] = model_output_frame[:, :] * scale_factor

                # Create visualization
                fig = plt.figure(figsize=(15, 5))

                # Plot ground truth
                ax1 = fig.add_subplot(121, projection="3d")
                
                true_joints = (
                    ground_truth_frame.cpu().numpy()
                )  # Adjust key based on your data structure
                self.plot_skeleton(true_joints, ax1)
                ax1.set_title(f"Ground Truth - Frame {i}")

                # Plot prediction
                ax2 = fig.add_subplot(122, projection="3d")
                pred_joints = (
                    model_output_frame.cpu().numpy()
                )  # Adjust key based on your output structure
                self.plot_skeleton(pred_joints, ax2)
                ax2.set_title(f"Prediction - Frame {i}")

                # set fixed axis ranges
                ax1.set_xlim(-1, 1)
                ax1.set_ylim(-1, 1)
                ax1.set_zlim(-1, 1)
                ax2.set_xlim(-1, 1)
                ax2.set_ylim(-1, 1)
                ax2.set_zlim(-1, 1)

                # Set common properties
                for ax in [ax1, ax2]:
                    ax.set_xlabel("X")
                    ax.set_ylabel("Z")
                    ax.set_zlabel("Y")
                    ax.view_init(elev=20, azim=45)

                temp_path = f"visualisations/temp_{i}.png"
                plt.savefig(temp_path)
                temp_images.append(temp_path)
                plt.close(fig)

            # Create GIF
            images = [imageio.imread(filename) for filename in temp_images]
            imageio.mimsave(
                f"visualisations/epoch_{trainer.current_epoch}_animation.gif",
                images,
                fps=10,
            )  # Adjust fps as needed

            # Clean up temporary files
            for temp_file in temp_images:
                os.remove(temp_file)