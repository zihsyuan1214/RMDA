import torch
from typing import Iterable, Callable

class RMDA(torch.optim.Optimizer):
    '''
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            Learning rate (default: 1e-1).
        momentum (float): 
            Momentum value in  the range (0,1] (default: 1e-2).
        lambda_ (float): 
            regularization weight (default: 0.0).
        prox_fn (Callable):
            proximal function with corresponding regularization (default: None).
    '''
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter], 
                 lr: float = 1e-1, momentum: float = 1e-2, 
                 lambda_: float = 0.0, prox_fn: Callable = None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum <= 0.0 or momentum > 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if lambda_ < 0.0:
            raise ValueError("Invalid lambda_ value: {}".format(lambda_))

        defaults = dict(lr=lr, momentum=momentum, 
                        lambda_=lambda_, prox_fn=prox_fn,
                        iteration=0.0, accum=0.0)

        super(RMDA, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):  
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            lambda_ = group['lambda_']
            prox_fn = group['prox_fn']
            iteration = group['iteration']
            accum = group['accum']

            scaling_coeffcient = (iteration+1)**0.5
            stepsize = lr*scaling_coeffcient
            accum += stepsize

            alpha = accum/scaling_coeffcient

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                param_state = self.state[p]
                if 'initial_point' not in param_state:
                    p0 = param_state['initial_point'] = p.clone().detach()
                else:
                    p0 = param_state['initial_point']
                    
                if 'gradient_buffer' not in param_state:
                    gradient_buffer = param_state['gradient_buffer'] = torch.zeros_like(d_p)
                else:
                    gradient_buffer = param_state['gradient_buffer']
                
                # update gradient_buffer 
                gradient_buffer.add_(d_p, alpha=stepsize)
                
                # compute p_tilde 
                p_tilde = p0.sub(gradient_buffer, alpha=(1/scaling_coeffcient))
                
                # do proximal operation if needed
                if lambda_ != 0 and prox_fn is not None:
                    prox_fn(p=p_tilde, lambda_=lambda_, alpha=alpha)
                
                # introduce momentum and update parameters
                if momentum != 1.0:
                    p.mul_(1-momentum).add_(p_tilde, alpha=momentum)
                else:
                    p.copy_(p_tilde)

        group['accum'] = accum
        group['iteration'] += 1
