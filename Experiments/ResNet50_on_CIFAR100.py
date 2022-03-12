import torch
import torchvision
import argparse
import sys 
sys.path.append("..")
 
from Models.resnet50 import ResNet50
from RMDA.Train import train
from RMDA.Validation.evaluation import Evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--download', action='store_false', default=True)
parser.add_argument('--training-batch-size', type=int, default=128)
parser.add_argument('--testing-batch-size', type=int, default=128)
parser.add_argument('--gpu', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--lr', type=float, default=1e0)
parser.add_argument('--momentum', type=float, default=1e-2)
parser.add_argument('--lambda_', type=float, default=1e-5)
parser.add_argument('--regularization', type=str, default="Group_LASSO")
parser.add_argument('--milestones', type=int, nargs='+', default=[i for i in range(150, 750, 150)])
parser.add_argument('--gamma', type=float, default=1e-1)
args = parser.parse_args()

training_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])
testing_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])
training_dataset = torchvision.datasets.CIFAR100(root='./Data',
                                                 train=True,
                                                 download=args.download,
                                                 transform=training_transforms)
training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset, 
                                                  batch_size=args.training_batch_size, 
                                                  shuffle=True,
                                                  pin_memory=True)
testing_dataset = torchvision.datasets.CIFAR100(root='./Data', 
                                                train=False, 
                                                download=args.download,
                                                transform=testing_transforms)
testing_dataloader = torch.utils.data.DataLoader(dataset=testing_dataset, 
                                                 batch_size=args.testing_batch_size, 
                                                 pin_memory=True)

criterion_mean = torch.nn.NLLLoss(reduction='mean')
criterion_sum = torch.nn.NLLLoss(reduction='sum')
model = ResNet50(num_classes=100)

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

(training_objective, training_accuracy, validation_accuracy, 
unstructured_sparsity, structured_sparsity) = Evaluation(training_dataloader=training_dataloader, 
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
print("training accuracy: {}".format(training_accuracy))
print("validation accuracy: {}".format(validation_accuracy))
print("unstructured sparsity: {}".format(unstructured_sparsity))
print("structured sparsity: {}".format(structured_sparsity))
