import torch
from torch.optim import Optimizer

class DeepMomentum(Optimizer):
    """
    移植版 DeepMomentum，適配標準 PyTorch Optimizer 接口。
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        beta=0.9,
        beta2=0.999,
        eps=1e-8,
        variant="preconditioned",
        weight_decay=0.0
    ):
        defaults = dict(
            lr=lr, 
            beta=beta, 
            beta2=beta2, 
            eps=eps, 
            variant=variant, 
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            beta2 = group['beta2']
            eps = group['eps']
            variant = group['variant']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['sq_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                sq_avg = state['sq_avg']
                grad_avg = state['grad_avg']
                
                # Core DeepMomentum Logic from original 'deep.py'
                update = grad
                
                if variant in ["preconditioned", "muon"]:
                    # RMSProp-like preconditioning
                    sq_avg.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = sq_avg.sqrt().add_(eps)
                    update = grad / denom

                if variant == "l2_objective":
                    # Custom logic from original code
                    update = grad + 0.1 * torch.mean(grad, dim=-1, keepdim=True)
                
                if variant in ["dmgd", "muon"]:
                    # Non-linearity
                    update = torch.tanh(update)

                # Perform step
                grad_avg.mul_(beta).add_(update, alpha=1 - beta)
                p.add_(grad_avg, alpha=-lr)

        return loss
