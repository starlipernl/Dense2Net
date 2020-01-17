from __future__ import print_function

# -*- coding: utf-8 -*-
"""
# Dense2Net
Implementation incorporating the Res2Net architecture into dense net.
DenseNet Paper: https://arxiv.org/abs/1608.06993
Res2Net Paper: https://arxiv.org/abs/1904.01169
DenseNet Cifar10 code from
https://github.com/kuangliu/pytorch-cifar
Res2Net code from https://github.com/lxtGH/OctaveConv_pytorch/blob/master/nn/res2net.py

USAGE: python3 dense2net.py --args
-lr: Specify learning rate (default=0.1)
--r: Resume training from checkpoint (default=False)
--a: Run without data augmentation 
--se: run without SE layer 
--c: Run without cutout 
--scale: Specify scale parameters (default=4)
--groups: Specify cardinality (default=1)

"""
# The below cell is only necessary when running in a Colaboratory notebook

# Load the Drive helper and mount 
# from google.colab import drive
# # This will prompt for authorization.
# drive.mount('/content/drive/')
# #Navigate to the directory containing this notebook
# %cd "/content/drive/My Drive/StarliperSongkakul-Project3/Dense2Net"

# !ls


import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from cutout import Cutout
import os
import argparse
import time
import numpy as np
import pickle as pkl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Use to limit gpu's
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def conv3x3(in_planes, out_planes, stride=1, groups=1):  
    # returns a 3x3 2d convolution, used in Res2Net sub-convolutions
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


class Res2Net_block(nn.Module):
    # Res2net bottleneck block
    def __init__(self, planes, scale=1, stride=1, groups=1, norm_layer=None):
        super(Res2Net_block, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.scale = scale
        ch_per_sub = planes // self.scale
        ch_res = planes % self.scale    
        self.chunks  = [ch_per_sub * i + ch_res for i in range(1, scale + 1)]
        self.conv_blocks = self.get_sub_convs(ch_per_sub, norm_layer, stride, groups)
        
    def forward(self, x):
        sub_convs = []
        sub_convs.append(x[:, :self.chunks[0]])
        sub_convs.append(self.conv_blocks[0](x[:, self.chunks[0]: self.chunks[1]]))
        for s in range(2, self.scale):
            sub_x = x[:, self.chunks[s-1]: self.chunks[s]]
            sub_x += sub_convs[-1]
            sub_convs.append(self.conv_blocks[s-1](sub_x))

        return torch.cat(sub_convs, dim=1)
    
    def get_sub_convs(self, ch_per_sub, norm_layer, stride, groups):
        layers = []
        for _ in range(1, self.scale):
            layers.append(nn.Sequential(
                conv3x3(ch_per_sub, ch_per_sub, stride, groups),
                norm_layer(ch_per_sub), self.relu))
        
        return nn.Sequential(*layers)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Bottleneck(nn.Module):
  #Densenet bottleneck block
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        #Replaced 3x3 convolution with Res2Net block with scale=4
        #Use this for standard Densenet
        #self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        #Use this for Dense2net
        self.conv2 = Res2Net_block(4*growth_rate, scale=args.scale, stride=1, groups=args.groups)
        self.conv3 =nn.Conv2d(4*growth_rate, growth_rate, kernel_size=1, bias=False)
        self.se = SELayer(growth_rate, reduction=16)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(out)
        # out = F.dropout(out, p=0.2, training=self.training)
        if args.se:
            out = self.se(out)
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
  #Densenet transition layer
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=100):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    overfit = 0
    overfit_limit = 99.98
    train_loss = 0
    correct = 0
    total = 0
    avg_loss = 0
    acc = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        avg_loss = train_loss/(batch_idx+1)
        if(batch_idx%25==0) or batch_idx==len(trainloader)-1:
          print(batch_idx+1, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (avg_loss, acc, correct, total))
    if acc > overfit_limit:
        overfit = 1

    return avg_loss, acc, overfit


def test(epoch, acc_count):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    avg_loss = 0
    acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            avg_loss = test_loss / (batch_idx + 1)
            if(batch_idx%25==0) or batch_idx==len(testloader)-1:
                print(batch_idx+1, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
        acc_count = 0
    else:
        acc_count += 1
    return avg_loss, acc, acc_count


def plot_curves(train, test, stop, mode):
    if mode == 1:
        title = "Accuracy Curves"
        label = "Accuracy"
        filename = "acc.png"
    else:
        title = "Loss Curves"
        label = "Loss"
        filename = "loss.png"
    num_epochs_plot = range(0, stop+1)  # x axis range
    plt.figure()
    plt.plot(num_epochs_plot, train[:stop+1], "b", label="Training")
    plt.plot(num_epochs_plot, test[:stop+1], "r", label="Validation")
    plt.title(title)
    plt.xlabel("Number of Epochs")
    plt.ylabel(label)
    plt.legend()
    plt.savefig(filename)
    plt.close()


# Main training script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--a', action='store_false', default=True, help='Remove Data Augmentation')
    parser.add_argument('--se', action='store_false', default=True, help='Remove SE block')
    parser.add_argument('--c', action='store_false', default=True, help='Remove cutout')
    parser.add_argument('--scale', type=int, default=4, help='Specify scale param')
    parser.add_argument('--groups', type=int, default=1, help='Specify group param')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    end_epoch = 100 # number of epochs to run
    augment = 1 # set to 1 to augment data
    resume = 0 # resume from checkpoint
    acc_count = 0
    # Data
    print('==> Preparing data..')
    if args.a: #TANNER: this was backwards before, switched in v2
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
      ])
    else:
      transform_train = transforms.Compose([
          # transforms.RandomRotation(15),
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
      ])
    if args.c:
        transform_train.transforms.append(Cutout(n_holes=1, length=8))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    # CIFAR10 normalization for reference
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = DenseNet121()
    # net = DenseNet161()
    net = net.to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)
        cudnn.benchmark = True
    
    learning_rate = args.lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer,  mode='min', patience=10, verbose=True)

    train_loss = np.zeros((end_epoch, 1))
    train_acc = np.zeros((end_epoch, 1))
    test_acc = np.zeros((end_epoch, 1))
    test_loss = np.zeros((end_epoch, 1))

    if args.r:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        with open('results.pkl', 'rb') as file:
            train_loss, train_acc, test_loss, test_acc = pkl.load(file)


    #train and test
    track_overfit = 0
    start_time = time.time()
    stop_epoch = 0
    for epoch in range(start_epoch, end_epoch):
        # if (epoch == 30) or (epoch == 60):#((epoch+1)%30)==0:
        #     for g in optimizer.param_groups:
        #       learning_rate /= 10
        #       g['lr'] = learning_rate #reduce learning rate by a factor of 10 every 30 epochs
        #       print('learning rate reduced to '+str(learning_rate))
        train_loss[epoch], train_acc[epoch], overfit_check = train(epoch)
        test_loss[epoch], test_acc[epoch], acc_count = test(epoch, acc_count)
        scheduler.step(test_loss[epoch])
        results = [train_loss, train_acc, test_loss, test_acc]
        with open('results.pkl', 'wb') as file:
          pkl.dump(results, file)
        stop_epoch = epoch
        track_overfit += overfit_check
        if acc_count > 20:
          print('overfit, terminating run')
          break


    #plot training and testing curves
    plot_curves(train_loss, test_loss, stop_epoch, 0)
    plot_curves(train_acc, test_acc, stop_epoch, 1)
    end_time = time.time()
    print('Total Time: ', (end_time - start_time))
    print('Best Validation Accuracy:', best_acc)
