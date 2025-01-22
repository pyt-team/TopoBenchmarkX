"""This module contains the Evaluator class that is responsible for computing the metrics."""

import torch
from topobenchmark.evaluator import AbstractEvaluator

class PoseEvaluator(AbstractEvaluator):
    def __init__(self, task, **kwargs):
        self.task = task
        if self.task != "regression":
            raise ValueError(f"Invalid task {task}. Only 'regression' is supported for MPJPE.")

        # Define the specific frames for MPJPE calculation
        self.frames_to_evaluate = [2, 4, 8, 10] # 10, 14, 18, 22, 25]
        self.metrics = {f"mpjpe_{k}_frames": 0.0 for k in self.frames_to_evaluate}
        self.num_samples = {f"mpjpe_{k}_frames": 0 for k in self.frames_to_evaluate}  # For averaging
        self.best_metric = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, metrics={list(self.metrics.keys())})"

    def update(self, model_out: dict):
        preds = model_out["logits"].cpu()
        target = model_out["labels"].cpu()

        # Reshape to [batch_size, n_frames, n_joints, 3]
        n_frames = 50
        preds = preds.view(-1, n_frames, 22, 3)
        target = target.view(-1, n_frames, 22, 3)
        batch_size = preds.size(0)

        for k in self.frames_to_evaluate:
            if n_frames < k:
                continue  # Skip if there aren't enough frames

            # Extract first k frames
            preds_k = preds[:, :k, :, :]  # [batch_size, k, n_joints, 3]
            target_k = target[:, :k, :, :]  # [batch_size, k, n_joints, 3]

            # Compute MPJPE for k frames
            error = torch.norm(preds_k - target_k, dim=-1)  # [batch_size, k, n_joints]
            mpjpe_k = error.mean()  # Average over batch, frames, and joints

            # Update the running metric
            metric_name = f"mpjpe_{k}_frames"
            self.metrics[metric_name] += mpjpe_k.item() * batch_size  # Sum for averaging
            self.num_samples[metric_name] += batch_size

    def compute(self):
        """Compute the final metrics."""
        return {
            metric_name: total_error / self.num_samples[metric_name]
            for metric_name, total_error in self.metrics.items()
        }

    def reset(self):
        """Reset the metrics for a new evaluation."""
        self.metrics = {f"mpjpe_{k}_frames": 0.0 for k in self.frames_to_evaluate}
        self.num_samples = {f"mpjpe_{k}_frames": 0 for k in self.frames_to_evaluate}
