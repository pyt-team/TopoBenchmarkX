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
    per_frame : bool
        Whether to calculate MPJPE per frame (default: True),
        otherwise average up to that frame.
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
        per_frame: bool = True,
        step_size: int = 25,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(
                f"Expected num_outputs to be a positive integer but got {num_outputs}"
            )
        self.num_outputs = num_outputs

        if not isinstance(per_frame, bool):
            raise ValueError(
                f"Expected argument `per_frame` to be a boolean but got {per_frame}"
            )
        self.per_frame = per_frame

        if not (isinstance(step_size, int) and step_size > 0):
            raise ValueError(
                f"Expected argument `step_size` to be a positive integer but got {step_size}"
            )
        self.step_size = step_size

        # Initialize joint indices
        self.joint_to_ignore = torch.tensor([16, 20, 23, 24, 28, 31])
        self.joint_equal = torch.tensor([13, 19, 22, 13, 27, 30])
        self.joint_used_xyz = torch.tensor(
            [
                2,
                3,
                4,
                5,
                7,
                8,
                9,
                10,
                12,
                13,
                14,
                15,
                17,
                18,
                19,
                21,
                22,
                25,
                26,
                27,
                29,
                30,
            ]
        )

        # Initialize states for accumulating errors
        self.add_state(
            "sum_errors",
            default=torch.zeros(num_outputs),
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

        # Predict "autoregressively"
        num_steps = (
            25 // self.step_size
        )  # predictions only for first 1000ms = 25 frames
        # for step in range(num_steps):

        # # Handle joint substitutions
        # preds = preds.clone()
        # preds[..., self.joint_to_ignore, :] = preds[..., self.joint_equal, :]

        # Calculate MPJPE (in millimeters)
        errors = torch.norm(
            preds * 1000 - target * 1000, dim=-1
        )  # L2 norm across xyz
        mean_errors = torch.mean(errors, dim=2)  # Mean across joints

        if self.per_frame:
            # Sum errors per frame
            self.sum_errors += torch.sum(mean_errors, dim=0)
        else:
            # Average up to each frame
            cumulative_means = (
                torch.cumsum(mean_errors, dim=1)
                / torch.arange(
                    1, mean_errors.shape[1] + 1, device=mean_errors.device
                )[None, :]
            )
            self.sum_errors += torch.sum(cumulative_means, dim=0)

        self.total += preds.shape[0]  # Add batch size

    def compute(self) -> torch.Tensor:
        """Summary.

        Returns
        -------
        torch.Tensor
            Mean per joint position error in millimeters per frame.
        """
        return self.sum_errors / self.total
