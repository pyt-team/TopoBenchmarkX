"""Loss module for the topobenchmark package."""

import torch
import torch_geometric

from topobenchmark.loss.base import AbstractLoss
from topobenchmark.loss.dataset import DatasetLoss


class GCNextLoss(AbstractLoss):
    r"""Defines the default model loss for the given task.

    Parameters
    ----------
    dataset_loss : dict
        Dictionary containing the dataset loss information.
    modules_losses : AbstractLoss, optional
        Custom modules' losses to be used.
    """

    def __init__(self, frames_considered=50):  # noqa: B006
        super().__init__()
        self.frames_considered = frames_considered
        self.criterion = torch.nn.MSELoss()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(frames_considered={self.frames_considered})"

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass of the loss function.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output with the loss.
        """
        pred = model_out["x_0"]
        target = model_out["labels"]

        # target = target.unsqueeze(1)
        pred = pred.view(-1, 50, 22, 3)
        target = target.view(-1, 50, 22, 3)

        # here we want to plot the predicted frames and the target frames to compare them
        visualize_skeleton_static(pred)
        visualize_skeleton_static(target)

        # Compute loss for the frames considered
        pred_considered = pred[:, :self.frames_considered, :, :]
        target_considered = target[:, :self.frames_considered, :, :]


        loss = self.criterion(pred_considered, target_considered)

        model_out["loss"] = loss

        return model_out


def visualize_skeleton_static(skeleton):
    """Plot the frames of th eskeleton, in a sequence
    
    Parameters
    ----------
    skeleton : torch.Tensor
        The skeleton to visualize.
        shape: [batch_size, n_frames, n_joints, 3]
    """
    # # pick the first batch size
    # skeleton = skeleton[0] # shape: [n_frames, n_joints, 3]
    
    # for frame in skeleton:
    #     visualize_skeleton(frame)
    pass