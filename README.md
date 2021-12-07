# Training Structured Neural Networks Through Manifold Identification and Variance Reduction

This repository is a pytorch implementation of the Regularized Modernized Dual Averaging (RMDA) algorithm for training structred neural network models.
Details of the algorithm can be found in the following paper:
> Zih-Syuan Huang, Ching-pei Lee, *Training Structured Neural Networks Through Manifold Identification and Variance Reduction*[[arXiv](https://arxiv.org/abs/2112.02612)]

When provided with a regularizer and the corresponding proximal operator, this algorithm trains a neural network model that conforms the structure induced by the regularizer.
In this repository, we include the proximal operator of the L1 norm and the group-LASSO norm as illustrating examples, but users can replace them with any other proximal operators.

This repository contains:
 - Regularized modernized dual averaging ([RMDA](https://github.com/zihsyuan1214/rmda/blob/master/RMDA/Optimizer/rmda.py)) algorithm.
 - [Scheduler](https://github.com/zihsyuan1214/rmda/blob/master/RMDA/ParamScheduler/param_scheduler.py) for learning rate, momentum scheduling and restart.
 - [Proximal operators](https://github.com/zihsyuan1214/rmda/blob/master/RMDA/ProxFn/prox_fns.py) for the group-LASSO and L1 norms.
 - [Training file](https://github.com/zihsyuan1214/rmda/blob/master/RMDA/Train.py). An exemplary wrapper for using our algorithm to train a structured neural network.

## Getting started
To compile the code, you will need to install torch and torchvision.

## Examples
### Logistic Regression on MNIST
To run an experiment of logistic regression on MNIST, run:

    python LogisticRegression_on_MNIST.py

in the [Experiments directory](https://github.com/zihsyuan1214/rmda/tree/master/Experiments).
