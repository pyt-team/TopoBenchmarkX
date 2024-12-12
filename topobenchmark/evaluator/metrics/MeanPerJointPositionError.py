"""Loss module for human motion prediction in the topobenchmark package."""

from typing import Any

import torch
from torchmetrics import Metric
from torchmetrics.functional.regression.mse import (
    _mean_squared_error_compute,
    _mean_squared_error_update,
)


class MeanPerJointPositionError(Metric):
    r"""Mean per joint position error (MPJPE), used with H3.6MDataaset.

    Parameters
    ----------
    num_outputs : int
        The number of outputs.
    step_size : int
        Step size over which to calculate MPJPE.
    **kwargs : Any
        Additional keyword arguments.
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    sum_squared_error: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        num_outputs: int = 1,
        step_size: int = 25,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(
                f"Expected num_outputs to be a positive integer but got {num_outputs}"
            )
        self.num_outputs = num_outputs

        if not (isinstance(step_size, int) and step_size > 0):
            raise ValueError(
                f"Expected argument `step_size` to be a positive integer but got {step_size}"
            )
        self.step_size = step_size

        # Initialize states for accumulating errors
        self.add_state(
            "sum_errors",
            default=torch.zeros(
                num_outputs * (50 // step_size)
            ),  # To store multiple steps
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Parameters
        ----------
        preds : torch.Tensor
            Predictions from model shape=(bs*frames*joints*channels, 1).
        target : torch.Tensor
            Ground truth values shape=(bs*frames*joints*channels, 1).
        """
        # Reshape inputs from (bs*f*j*c, 1) to (bs, f, j, c)
        batch_size = preds.shape[0] // (
            50 * 22 * 3
        )  # 50 frames, 22 joints, 3 channels
        preds = preds.view(batch_size, 50, 22, 3)
        target = target.view(batch_size, 50, 22, 3)

        # Step 1: Convert to millimetres and get difference
        error_mm = 1000 * (preds - target)

        # Step 2: Calculate euclidean distance across xyz coordinates -> (bs, f, j)
        euc_dist = torch.norm(error_mm, dim=3)

        # Step 3: Average error across joints -> (bs, f)
        avg_across_joints = torch.mean(euc_dist, dim=2)

        # Step 4: Sum across samples in batch -> (f, )
        mpjpe = torch.sum(avg_across_joints, dim=0)

        # Step 5: Calculate MPJPE at step_size intervals
        step_indices = torch.arange(0, 50, self.step_size)
        mpjpe_steps = mpjpe[step_indices]  # Shape: (num_steps,)
        self.sum_errors += mpjpe_steps  # Shape: (num_steps,)

        self.total += preds.shape[0]  # Add batch size

    def compute(self) -> torch.Tensor:
        """Summary.

        Returns
        -------
        torch.Tensor
            Mean per joint position error in millimeters per frame.
        """
        return self.sum_errors / self.total
