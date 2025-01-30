import torch

class H36MCriterion(torch.nn.Module):
    """Custom loss criterion for H36M dataset
    that includes preprocessing steps for including only first 10 frames.
    """
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, logits, target):
        """Forward pass for the loss criterion.
        
        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits.
        target : torch.Tensor
            Target labels.
        
        Returns
        -------
        torch.Tensor
            Loss value.
        """
        # Preprocessing steps
        logits = logits.reshape(-1, 50, 22, 3)
        target = target.reshape(-1, 50, 22, 3)
        
        logits = logits[:, :10, :, :]
        target = target[:, :10, :, :]
                
        # Calculate loss
        return self.mse(logits, target)