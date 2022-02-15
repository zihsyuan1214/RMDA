import torch
import torchvision

def VGG19(num_classes):
    model = torchvision.models.vgg19_bn(num_classes=num_classes)
    model.avgpool = torch.nn.Identity()
    model.classifier = torch.nn.Sequential(torch.nn.Linear(512, num_classes), torch.nn.LogSoftmax(dim=1))
    model._initialize_weights()
    return model

'''
vgg19_cifar10 = VGG19(num_classes=10)
torch.save(vgg19_cifar10.state_dict(), 'VGG19_CIFAR10.pt')

vgg19_cifar10_1 = VGG19(num_classes=10)
torch.save(vgg19_cifar10_1.state_dict(), 'VGG19_CIFAR10_1.pt')

vgg19_cifar10_2 = VGG19(num_classes=10)
torch.save(vgg19_cifar10_2.state_dict(), 'VGG19_CIFAR10_2.pt')

vgg19_cifar10_3 = VGG19(num_classes=10)
torch.save(vgg19_cifar10_3.state_dict(), 'VGG19_CIFAR10_3.pt')

vgg19_cifar100 = VGG19(num_classes=100)
torch.save(vgg19_cifar100.state_dict(), 'VGG19_CIFAR100.pt')

vgg19_cifar100_1 = VGG19(num_classes=100)
torch.save(vgg19_cifar100_1.state_dict(), 'VGG19_CIFAR100_1.pt')

vgg19_cifar100_2 = VGG19(num_classes=100)
torch.save(vgg19_cifar100_2.state_dict(), 'VGG19_CIFAR100_2.pt')

vgg19_cifar100_3 = VGG19(num_classes=100)
torch.save(vgg19_cifar100_3.state_dict(), 'VGG19_CIFAR100_3.pt')
'''
