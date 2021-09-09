from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchmetrics.functional import accuracy

seed_everything(7)

PATH_DATASETS = os.environ.get('PATH_DATASETS', '.')
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 128 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# CIFAR10 Data Module
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LitResNet(LightningModule):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        err = 1 - acc 

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage} classificaiton error', err, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        # Set milestones=[80, 160] like torch blog 
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160], last_epoch=-1)
        return {
            'optimizer' : optimizer,
            'lr_scheduler': lr_scheduler
        }
    

def resnet20():
    return LitResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return LitResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return LitResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return LitResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return LitResNet(BasicBlock, [18, 18, 18])

def resnet1202():
    return LitResNet(BasicBlock, [200, 200, 200])

def test():
    model = resnet20()

    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=200,
        gpus=AVAIL_GPUS,
        logger=WandbLogger(name="ResNet-20",project="Lightning-Cifar10", log_model="all", reinit=True),
        callbacks=[LearningRateMonitor(logging_interval='step')],
    )

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    wandb.finish()

if __name__ == "__main__":
    test()