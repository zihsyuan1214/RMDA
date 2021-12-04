# Training Structured Neural Networks Through Manifold Identification and Variance Reduction

This is the repository for the paper
> Zih-Syuan Huang, Ching-pei Lee, *Training Structured Neural Networks Through Manifold Identification and Variance Reduction*

This repository contains:
 - Regularized modernized dual averaging ([RMDA](https://github.com/zihsyuan1214/rmda/blob/master/RMDA/Optimizer/rmda.py)) algorithm.
 - [Scheduler](https://github.com/zihsyuan1214/rmda/blob/master/RMDA/ParamScheduler/param_scheduler.py) for learning rate, momentum and doing restart.  
 - [Proximal operators](https://github.com/zihsyuan1214/rmda/blob/master/RMDA/ProxFn/prox_fns.py) for the group LASSO and L1.
 - [Training file](https://github.com/zihsyuan1214/rmda/blob/master/RMDA/Train.py).

## Getting started
To compile the code, you will need to install torch and torchvision.

## Examples
### Logistic Regression on MNIST
To run an experiment of logistic regression on MNIST, run:

    python LogisticRegression_on_MNIST.py

in the [Experiments directory](https://github.com/zihsyuan1214/rmda/tree/master/Experiments).
