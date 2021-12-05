#This file provides a wrapper as an example on how to use RMDA.
#In this example, we train sparse neural networks using either L1-regularization
#(for unstructured sparsity considered in pruning) or group-LASSO regularization
#(for structured sparsity that groups outgoing weights of each neuron
#separately, and treats each channel in a convolutional layer as a group).

import torch

from RMDA.Optimizer.rmda import RMDA
from RMDA.ProxFn.prox_fns import prox_glasso, prox_l1
from RMDA.ParamScheduler.param_scheduler import MultiStepParam

def train(training_dataloader, 
          model, 
          criterion, 
          epochs: int, 
          lr : float,
          momentum: float,
          lambda_: float,
          regularization: str,
          milestones: list,
          gamma: float,
          gpu : bool = True):
    if gpu:
        model.cuda()

    if regularization == "Group LASSO":
        prox_fn = prox_glasso
    elif regularization == "L1":
        prox_fn = prox_l1    
    else:
        raise ValueError('Unknown regularization '+regularization)
        
    optimizer = RMDA(params=model.parameters(),                       
                     lr=lr,
                     momentum=momentum,
                     lambda_=lambda_,
                     prox_fn=prox_fn) 
    
    scheduler = MultiStepParam(optimizer=optimizer, 
                               milestones=milestones, 
                               gamma=gamma) 
    
    for epoch in range(epochs):
        model.train()
        for X, y in training_dataloader:
            if gpu:
                X, y = X.cuda(), y.cuda()
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()     
            
        scheduler.step()

    return optimizer
