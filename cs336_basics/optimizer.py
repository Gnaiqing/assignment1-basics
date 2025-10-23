from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


def cross_entropy(inputs: Float[Tensor, " ... vocab_size"], targets: Int[Tensor, " ..."]):
    """
    Return the cross entropy loss given input logits and targets
    """
    max_logits, _ = torch.max(inputs, dim=-1, keepdim=True)
    inputs = inputs - max_logits
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = torch.logsumexp(inputs, dim=-1) - target_logits
    return torch.mean(loss)


def calc_lr_cosine_schedule(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int) -> float:
    """
    Return the cosine annealing learning rate
    """
    if t < t_w:
        return lr_max * t / t_w
    elif t <= t_c:
        return lr_min + 0.5 * (1+math.cos((t-t_w)*math.pi/(t_c-t_w)))* (lr_max-lr_min)
    else:
        return lr_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    l2_norm = 0
    for param in parameters:
        if param.grad is not None:
            l2_norm += param.grad.data.square().sum()

    l2_norm = l2_norm ** 0.5
    if l2_norm > max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad.data = param.grad.data * max_l2_norm / (l2_norm + 1e-6)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 weight_decay=0.01,
                 betas=(0.9, 0.999),
                 eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                m = state.get("m", torch.zeros_like(p.data, device=p.device))  # get first moment estimate
                v = state.get("v", torch.zeros_like(p.data, device=p.device))  # get second moment estimate
                t = state.get("t", 1)  # get current iteration
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                m = betas[0] * m + (1-betas[0]) * grad
                v = betas[1] * v + (1-betas[1]) * grad * grad
                lr_t = lr * (1-betas[1]**t)**0.5 / (1 - betas[0]**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m  # Update first momentum
                state["v"] = v   # Update second momentum

        return loss



