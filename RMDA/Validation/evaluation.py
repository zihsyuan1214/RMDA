import torch

from RMDA.ProxFn.prox_fns import prox_glasso, prox_l1
  
def Evaluation(training_dataloader, 
               testing_dataloader,
               len_training_dataset,
               len_testing_dataset,
               model,
               criterion_sum,
               lambda_,
               regularization,
               gpu):
    model.eval()  
    
    training_objective = 0.0
    training_accuracy = 0.0
    with torch.no_grad():
        for X, y in training_dataloader:
            if gpu:
                X, y = X.cuda(), y.cuda()
            y_hat = model(X)
            loss = criterion_sum(y_hat, y)
            y_hat = y_hat.argmax(dim=1)
            training_objective += loss.item()
            training_accuracy += y_hat.eq(y.view_as(y_hat)).float().sum().item() 
            
    training_objective /= len_training_dataset
    training_accuracy /= len_training_dataset
        
    
    if regularization == "Group LASSO":
        prox_fn = prox_glasso
    elif regularization == "L1":
        prox_fn = prox_l1    
    else:
        raise ValueError('Unknown regularization '+regularization)
        
    if lambda_ != 0.0:
        for p in model.parameters():
            if regularization == "Group LASSO":
                if p.ndim == 4 or p.ndim == 2:
                    reg_scaling = (p.numel()/p.shape[1])**0.5
                if p.ndim == 4:
                    training_objective += reg_scaling*lambda_*(torch.linalg.norm(p, dim=(0,2,3)).sum().item())
                elif p.ndim == 2:
                    training_objective += reg_scaling*lambda_*(torch.linalg.norm(p, dim=(0)).sum().item())
                    
            elif regularization == "L1":
                if p.ndim == 4 or p.ndim == 2:
                    training_objective += lambda_*(p.abs().sum().item())
                
    validation_accuracy = 0.0
    with torch.no_grad():
        for X, y in testing_dataloader:
            if gpu:
                X, y = X.cuda(), y.cuda()
            output = model(X)
            y_hat = output.argmax(dim=1)
            validation_accuracy += y_hat.eq(y.view_as(y_hat)).float().sum().item() 
    validation_accuracy /= len_testing_dataset    
        
    nonzero = 0.0
    num_el = 0.0 
    for p in model.parameters():
        nonzero += p.clone().detach().count_nonzero().item()
        num_el += p.numel()
    sparsity = 1.0-(nonzero/num_el)
    
    num_nonsparse_group = 0.0
    num_group = 0.0
    if lambda_ != 0.0:
        for p in model.parameters():
            if p.ndim == 4: 
                num_nonsparse_group += p.clone().detach().count_nonzero(dim=(2,3)).count_nonzero().item()
                num_group += (p.shape[0]*p.shape[1])
            elif p.ndim == 2:
                num_nonsparse_group += p.clone().detach().count_nonzero(dim=(0)).count_nonzero().item()
                num_group += p.shape[1]            
        group_sparsity = 1.0-(num_nonsparse_group/num_group)
    else:
        group_sparsity = 0.0
    return training_objective, validation_accuracy, training_accuracy, sparsity, group_sparsity
