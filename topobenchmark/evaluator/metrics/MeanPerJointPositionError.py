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
                f"Expected argument `squared` to be a boolean but got {per_frame}"
            )
        self.per_frame = per_frame

        self.add_state(
            "mean_per_joint_position_error",
            default=torch.zeros(num_outputs),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Parameters
        ----------
        preds : torch.Tensor
            Predictions from model.
        target : torch.Tensor
            Ground truth values.
        """
        sum_squared_error, num_obs = _mean_squared_error_update(
            preds, target, num_outputs=self.num_outputs
        )

        self.sum_squared_error += sum_squared_error
        self.total += num_obs

    def compute(self) -> torch.Tensor:
        """Compute mean squared error over state.

        Returns
        -------
        torch.Tensor
            Mean squared error.
        """
        return _mean_squared_error_compute(
            self.sum_squared_error, self.total, squared=self.squared
        )
