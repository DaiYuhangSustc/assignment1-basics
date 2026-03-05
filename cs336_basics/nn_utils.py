import torch
from torch import Tensor
from typing import Iterable


def softmax(in_features: Tensor, dim: int) -> Tensor:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Tensor): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Tensor: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    # Subtract max for numerical stability
    max_val = in_features.max(dim=dim, keepdim=True).values
    exp_features = torch.exp(in_features - max_val)
    sum_exp = exp_features.sum(dim=dim, keepdim=True)
    return exp_features / sum_exp


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Tensor): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
            Shape: (batch_size, vocab_size)
        targets (Tensor): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Tensor: The average cross-entropy loss across examples.
    """
    # Compute softmax with numerical stability
    max_val = inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = max_val.squeeze(-1) + torch.log(torch.exp(inputs - max_val).sum(dim=-1))
    
    # Get the log probability of the correct class for each example
    batch_size = inputs.shape[0]
    correct_logits = inputs[torch.arange(batch_size), targets]
    
    # Cross entropy: -log(softmax(inputs)[correct_class])
    loss = log_sum_exp - correct_logits
    
    return loss.mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # Collect all gradients into a single list
    grads = [p.grad for p in parameters if p.grad is not None]
    
    if len(grads) == 0:
        return
    
    # Compute total L2 norm of all gradients
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    
    # Clip if necessary
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(clip_coef)
