import torch

class BasicBlock(torch.nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = torch.nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return torch.nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(torch.nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = torch.nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return torch.nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(torch.nn.Module):
    def __init__(self, block, num_block, num_classes):
        super().__init__()

        self.in_channels = 64

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = self.log_softmax(output)

        return output

def ResNet18(num_classes):
    return ResNet(block=BasicBlock, num_block=[2, 2, 2, 2], num_classes=num_classes)

def ResNet34(num_classes):
    return ResNet(block=BasicBlock, num_block=[3, 4, 6, 3], num_classes=num_classes)

def ResNet50(num_classes):
    return ResNet(block=BottleNeck, num_block=[3, 4, 6, 3], num_classes=num_classes)

def ResNet101(num_classes):
    return ResNet(block=BottleNeck, num_block=[3, 4, 23, 3], num_classes=num_classes)

def ResNet152(num_classes):
    return ResNet(block=BottleNeck, num_block=[3, 8, 36, 3], num_classes=num_classes)
