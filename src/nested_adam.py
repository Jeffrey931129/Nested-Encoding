from collections.abc import Iterable

import torch


class NestedAdam(torch.optim.Optimizer):
    """
    A custom Adam optimizer with low-frequency, macroscopic moment updates.

    The standard first and second moments (m_1, v) are updated at every step,
    while an additional macroscopic first moment (m_2) is updated every `chunk_size`
    steps using the average of the accumulated gradients within the chunk.
    Local parameter updates use standard bias-corrected moments combined with the
    bias-corrected macroscopic moment.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-5,
        alpha: float = 1.0,
        beta: tuple[float, float, float] = (0.9, 0.9, 0.999),
        eps: float = 1e-8,
        freq: int = 1,
        chunk_size: int = 4,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta[0] < 0.0 or beta[0] >= 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta[0]}")
        if beta[1] < 0.0 or beta[1] >= 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta[1]}")
        if beta[2] < 0.0 or beta[2] >= 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {beta[2]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if freq < 1:
            raise ValueError(f"Invalid freq: {freq}")
        if chunk_size < 1:
            raise ValueError(f"Invalid chunk_size: {chunk_size}")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            beta=beta,
            eps=eps,
            freq=freq,
            chunk_size=chunk_size,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            beta1, beta2, beta3 = group["beta"]
            eps = group["eps"]
            freq = group["freq"]
            chunk_size = group["chunk_size"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad / freq
                state = self.state[p]

                if not state:
                    state["step"] = 0
                    state["m_1"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["m_2"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["m_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                m_1 = state["m_1"]
                m_2 = state["m_2"]
                v = state["v"]
                m_buffer = state["m_buffer"]

                m_1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta3).addcmul_(grad, grad, value=1.0 - beta3)
                m_buffer.add_(grad)

                if state["step"] % chunk_size == 0:
                    m_2.mul_(beta2).add_(m_buffer, alpha=(1.0 - beta2) / chunk_size)
                    m_buffer.zero_()

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** max(1, state["step"] // chunk_size)
                bias_correction3 = 1.0 - beta3 ** state["step"]

                m_1_hat = m_1 / bias_correction1
                m_2_hat = m_2 / bias_correction2
                v_hat = v / bias_correction3

                denom = v_hat.sqrt().add_(eps)
                update_direction = m_1_hat.add(m_2_hat, alpha=alpha).div_(denom)
                p.mul_(1.0 - lr * weight_decay)
                p.add_(update_direction, alpha=-lr)

        return loss
