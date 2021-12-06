import torch
import torchvision
import argparse
import sys 
sys.path.append("..") 

from RMDA.Train import train
from RMDA.Model.logistic_regression import Lin
from RMDA.Validation.evaluation import Evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--download', action='store_false', default=True)
parser.add_argument('--training-batch-size', type=int, default=128)
parser.add_argument('--testing-batch-size', type=int, default=128)
parser.add_argument('--gpu', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--momentum', type=float, default=1e-2)
parser.add_argument('--lambda_', type=float, default=1e-3)
parser.add_argument('--regularization', type=str, default="Group LASSO")
parser.add_argument('--milestones', type=int, nargs='+', default=[50, 100])
parser.add_argument('--gamma', type=float, default=1e-1)
args = parser.parse_args()

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                             torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])
training_dataset = torchvision.datasets.MNIST(root='./Data',
                                              train=True,
                                              download=args.download,
                                              transform=transforms)
training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset, 
                                                  batch_size=args.training_batch_size, 
                                                  shuffle=True,
                                                  pin_memory=True)
testing_dataset = torchvision.datasets.MNIST(root='./Data', 
                                             train=False, 
                                             download=args.download,
                                             transform=transforms)
testing_dataloader = torch.utils.data.DataLoader(dataset=testing_dataset, 
                                                 batch_size=args.testing_batch_size,
                                                 pin_memory=True)

criterion_mean = torch.nn.NLLLoss(reduction='mean')
criterion_sum = torch.nn.NLLLoss(reduction='sum')
model = Lin()
with torch.no_grad():
    for p in model.parameters():
        p.zero_()

train(training_dataloader=training_dataloader, 
      model=model, 
      criterion=criterion_mean, 
      epochs=args.epochs, 
      lr=args.lr,
      momentum=args.momentum,
      lambda_=args.lambda_,
      regularization=args.regularization,
      milestones=args.milestones,
      gamma=args.gamma,
      gpu=args.gpu) 

(training_objective, validation_accuracy, training_accuracy,
sparsity, group_sparsity) = Evaluation(training_dataloader=training_dataloader, 
                                       testing_dataloader=testing_dataloader, 
                                       len_training_dataset=len(training_dataset),
                                       len_testing_dataset=len(testing_dataset), 
                                       model=model, 
                                       criterion_sum=criterion_sum, 
                                       lambda_=args.lambda_, 
                                       regularization=args.regularization,
                                       gpu=args.gpu)

print("----------Results----------")
print("training objective: {}".format(training_objective))
print("validation accuracy: {}".format(validation_accuracy))
print("training accuracy: {}".format(training_accuracy))
print("sparsity: {}".format(sparsity))
print("group sparsity: {}".format(group_sparsity))
