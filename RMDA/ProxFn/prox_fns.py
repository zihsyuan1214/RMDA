import torch

from torch import Tensor

# Group LASSO regularization
def prox_glasso(p: Tensor, lambda_: float, alpha: float):
    # reweight for group lasso
    if p.ndim == 4 or p.ndim == 2:
        lambda_alpha = lambda_*alpha 
        lambda_alpha *= (p.numel()/p.shape[1])**0.5
    # channel-wise
    if p.ndim == 4:
        temp = torch.nn.functional.relu(torch.linalg.norm(p, dim=(0,2,3), keepdim=True).sub(lambda_alpha))
        p.mul_(temp.div(temp.add(lambda_alpha)))  
    # column-wise
    elif p.ndim == 2:
        temp = torch.nn.functional.relu(torch.linalg.norm(p, dim=(0), keepdim=True).sub(lambda_alpha))
        p.mul_(temp.div(temp.add(lambda_alpha)))
        
# L1 regularization         
def prox_l1(p: Tensor, lambda_: float, alpha: float):
    if p.ndim == 4 or p.ndim == 2: 
        p_abs = p.abs()
        p.sign_().mul_(torch.nn.functional.relu(p_abs.sub(lambda_*alpha))) 
