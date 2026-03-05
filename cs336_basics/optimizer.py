from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None) -> float | None:
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            The loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                
                state["step"] += 1
                t = state["step"]
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** t
                exp_avg_hat = exp_avg / bias_correction1
                
                # Compute bias-corrected second raw moment estimate
                bias_correction2 = 1 - beta2 ** t
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                # Apply weight decay (decoupled from gradient-based update)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                
                # Update parameters
                denom = exp_avg_sq_hat.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg_hat, denom, value=-lr)
        
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Compute the learning rate at a given iteration using cosine learning rate 
    decay schedule with linear warmup.
    
    Args:
        it: Iteration number to get learning rate for.
        max_learning_rate: The maximum learning rate for cosine learning rate schedule.
        min_learning_rate: The minimum / final learning rate for the cosine learning rate schedule.
        warmup_iters: The number of iterations to linearly warm-up the learning rate.
        cosine_cycle_iters: The number of cosine annealing iterations.
    
    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        # Linear warmup phase
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        # Cosine annealing phase
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress))
    else:
        # Post-annealing phase: return minimum learning rate
        return min_learning_rate
