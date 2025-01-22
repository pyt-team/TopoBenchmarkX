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

    def __init__(self, frames_considered=10):  # noqa: B006
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
        pred = pred.view(-1, 50, 22, 3, 64)
        target = target.view(-1, 50, 22, 3, 1)

        # Compute loss for the frames considered
        pred_considered = pred[:, :self.frames_considered, :, :]
        target_considered = target[:, :self.frames_considered, :, :]


        loss = self.criterion(pred_considered, target_considered)

        model_out["loss"] = loss

        return model_out
