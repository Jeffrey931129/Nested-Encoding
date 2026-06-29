import math
import torch
from typing import Iterable, Tuple


class NestedAdam(torch.optim.Optimizer):
    """
    A custom Adam optimizer with low-frequency, macroscopic moment updates.

    The first and second moments (m, v) are only updated every `chunk_size` steps
    using the average of the accumulated gradients within the chunk.
    Local parameter updates use the raw stochastic gradients combined with the
    bias-corrected macroscopic moments.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        alpha: float = 1.0,
        beta: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        chunk_size: int = 4,
        weight_decay: float = 0.0,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta[0]}")
        if not 0.0 <= beta[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta[1]}")
        if chunk_size < 1:
            raise ValueError(f"Invalid chunk_size: {chunk_size}")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            beta=beta,
            eps=eps,
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
            alpha = group["alpha"]
            beta1, beta2 = group["beta"]
            eps = group["eps"]
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
                    state["inner_step"] = 0
                    state["chunk_step"] = 0

                    # Macroscopic moments (updated low-frequency)
                    state["m"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["v"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    # High-frequency accumulation buffers
                    state["m_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["v_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                state["inner_step"] += 1

                m = state["m"]
                v = state["v"]
                m_buffer = state["m_buffer"]
                v_buffer = state["v_buffer"]

                # 1. Accumulate raw gradients and squared gradients into buffers
                m_buffer.mul_(alpha).add_(grad, alpha=(1.0 - alpha))
                v_buffer.addcmul_(grad, grad)

                # 2. Update macroscopic moments strictly at chunk boundaries
                # Note: We also trigger an update on step 1 to prevent division by zero (v=0)
                if state["step"] == 1 or state["inner_step"] == chunk_size:
                    state["chunk_step"] += 1

                    # Update Rule: m = beta1 * m + (1 - beta1) * (m_buffer * scale)
                    m.mul_(beta1).add_(m_buffer, alpha=(1.0 - beta1))

                    # Update Rule: v = beta2 * v + (1 - beta2) * (v_buffer * scale)
                    # v.mul_(beta2).add_(v_buffer, alpha=(1.0 - beta2) * scale)

                    # Reset buffers for the next chunk interval
                    m_buffer.zero_()
                    # v_buffer.zero_()
                    state["inner_step"] = 0

                v.mul_(beta2).add_(v_buffer, alpha=(1.0 - beta2))
                v_buffer.zero_()

                # 3. Bias Correction based on the number of macroscopic updates (chunk_step)
                T = state["chunk_step"]
                bias_correction1 = 1.0 - beta1**T
                bias_correction2 = 1.0 - beta2**T

                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # 4. Local Parameter Update
                # Combines the immediate local gradient with the macroscopic momentum (m_hat),
                # scaled by the macroscopic variance (v_hat) to maintain Adam's adaptive properties.
                denom = v_hat.sqrt().add_(eps)

                # update_direction = (grad + alpha * m_hat) / denom
                update_direction = (
                    grad.mul(1.0 - alpha).add_(m_hat, alpha=alpha).div_(denom)
                )

                p.add_(update_direction, alpha=-lr)

        return loss
