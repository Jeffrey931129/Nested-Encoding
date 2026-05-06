import torch
from typing import Iterable

class NestedSGD(torch.optim.Optimizer):
    """
    A custom SGD optimizer equipped only with a slow (macroscopic) momentum mechanism.
    
    The local updates strictly follow the raw stochastic gradients (Vanilla SGD), 
    while a low-frequency momentum (m2) is updated every `chunk_size` steps to 
    provide a macroscopic directional correction based on accumulated gradients.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        alpha: float = 1.0,
        chunk_size: int = 4,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid slow momentum value: {momentum}")
        if chunk_size < 1:
            raise ValueError(f"Invalid chunk_size: {chunk_size}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            chunk_size=chunk_size,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            alpha = group["alpha"]
            chunk_size = group["chunk_size"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if not state:
                    state["step"] = 0
                    state["m2"] = torch.zeros_like(p)
                    state["slow_buffer"] = torch.zeros_like(p)

                state["step"] += 1
                
                m2 = state["m2"]
                slow_buffer = state["slow_buffer"]

                # 1. Accumulate raw gradients into the slow buffer for macroscopic tracking
                slow_buffer.add_(grad)

                # 2. Parameter update: Vanilla SGD gradient + Slow Momentum correction
                # Update rule: p = p - lr * (grad + alpha * m2)
                update = grad + alpha * m2
                p.add_(update, alpha=-lr)

                # 3. Update slow momentum strictly at chunk intervals
                if chunk_size > 0 and state["step"] % chunk_size == 0:
                    # Integrate the buffered gradients into the slow momentum
                    m2.mul_(beta).add_(slow_buffer)
                    
                    # Reset the buffer for the next chunk interval
                    slow_buffer.zero_()

        return loss