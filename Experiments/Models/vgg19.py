import torch
import torchvision

def VGG19(num_classes):
    model = torchvision.models.vgg19_bn(num_classes=num_classes)
    model.avgpool = torch.nn.Identity()
    model.classifier = torch.nn.Sequential(torch.nn.Linear(512, num_classes), torch.nn.LogSoftmax(dim=1))
    model._initialize_weights()
    return model
