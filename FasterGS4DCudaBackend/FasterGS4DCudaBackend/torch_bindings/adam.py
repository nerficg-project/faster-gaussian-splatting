import torch

from FasterGS4DCudaBackend import _C


class FusedAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps) -> None:
        super().__init__(params=params, lr=lr, eps=eps)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            assert len(group['params']) == 1, 'more than one tensor in group'
            param = group['params'][0]

            if param.grad is None or param.numel() == 0:
                continue

            state = self.state[param]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(param)
                state['exp_avg_sq'] = torch.zeros_like(param)

            state['step'] += 1

            _C.adam_step(
                param.grad,
                param,
                state['exp_avg'],
                state['exp_avg_sq'],
                state['step'],
                group['lr'],
                *group['betas'],
                group['eps'],
            )
