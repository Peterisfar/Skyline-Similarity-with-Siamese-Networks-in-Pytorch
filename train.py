from __future__ import print_function
import argparse
import os
import sys
import cv2
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"]="3"

import models.curve_compare_mode as ccm
from skyline_dataloader import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False,
                    default='./data/train/', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
                    default='./data/val/', help='path to val dataset')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
                    default=320, help='the lenght of the original input image')
parser.add_argument('--inputChannelSize', type=int,
                    default=2, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
                    default=1, help='size of the output channels')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--lrG', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--exp', default='./checkpoint/new', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=10, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=426,
                    help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
print(opt)

# Detect if we have a GPU available
opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# get train dataloader
train_dataset = SkylineDataset(root=opt.dataroot,
                                 seed=opt.manualSeed,
                                transform=transforms.Compose([ToTensor()]))
train_dataloader = DataLoader(train_dataset,
                       shuffle=True,
                       batch_size = opt.batchSize,
                       num_workers = opt.workers
                       )

# get valid dataloader
val_dataset = SkylineDataset(root=opt.valDataroot,
                                 seed=opt.manualSeed,
                                transform=transforms.Compose([ToTensor()]))
val_dataloader = DataLoader(val_dataset,
                       shuffle=False,
                       batch_size = opt.valBatchSize,
                       num_workers = opt.workers
                       )


# get logger
trainLogger = open('%s/train.log' % opt.exp, 'a+')
validLogger = open('%s/valid.log' % opt.exp, 'a+')

# get models
netG = ccm.Curve_Compare()
netG = netG.to(opt.device)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG, map_location={'cuda': 'cuda'}))
print(netG)

# get criterion
criterionLoss = nn.CrossEntropyLoss()

# get optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999), weight_decay=0)

# NOTE training loop
ganIterations = 0
best_val_acc = 0.
disp_loss = np.zeros(1)
time_str = time.strftime("%Y-%m-%d", time.localtime())
for epoch in range(opt.niter):
    for i, data in enumerate(train_dataloader, 0):
        x1 = data["line"][0].view(opt.batchSize, 1, -1)
        x2 = data["line"][1].view(opt.batchSize, 1, -1)
        label = data["label"].view(-1, )

        x1 = x1.to(opt.device)
        x2 = x2.to(opt.device)
        label = label.long().to(opt.device)

        optimizerG.zero_grad()  # start to update G

        netG.train()
        with torch.enable_grad():
            pred = netG(x1, x2)
            loss = criterionLoss(pred, label)
            loss.backward()
            optimizerG.step()

        # every opt.display write loss
        ganIterations += 1
        disp_loss += [loss.item()]
        if ganIterations % opt.display == 0:
            disp_loss = (disp_loss / opt.display)
            disp_str = '[%d/%d][%d/%d] loss:%f\t' % (
            epoch, opt.niter, i, len(train_dataset) / opt.batchSize, disp_loss[0])
            print(disp_str)
            disp_loss -= disp_loss
            sys.stdout.flush()
            trainLogger.write(disp_str + '\n')
            trainLogger.flush()

    # every 10 epoch to eval val_set
    if epoch % 1 == 0:
        netG.eval()
        print('*' * 5 + ' Validating ' + '*' * 5)
        acc_valid = 0.
        with torch.no_grad():
            for idx, data in enumerate(val_dataloader, 0):
                x1 = data["line"][0].view(opt.valBatchSize, 1, -1)
                x2 = data["line"][1].view(opt.valBatchSize, 1, -1)
                label = data["label"].view(-1, )

                x1 = x1.to(opt.device)
                x2 = x2.to(opt.device)
                label = label.long().to(opt.device)

                pred = netG(x1, x2)
                acc_valid += (pred.argmax(1) == label).sum().item()

            acc_valid /= len(val_dataset)
            print('Valid Accuracy = {}'.format(acc_valid))
            validLogger.write('epoch:{}\t {}\n'.format(epoch, acc_valid))
            validLogger.flush()

        # save model
        if acc_valid > best_val_acc:
            best_val_acc = acc_valid
            torch.save(netG.state_dict(), '%s/%s_best.pth' % (opt.exp, time_str))
            print('%s/%s_best.pth has beed saved.' % (opt.exp, time_str))

    # every 100 epoch to eval train_set
    if epoch % 1 == 0:
        netG.eval()
        print('*' * 5 + ' Validating the train_dataset ' + '*' * 5)
        acc_train = 0.
        with torch.no_grad():
            for idx, data in enumerate(train_dataloader, 0):
                x1 = data["line"][0].view(opt.batchSize, 1, -1)
                x2 = data["line"][1].view(opt.batchSize, 1, -1)
                label = data["label"].view(-1, )

                x1 = x1.to(opt.device)
                x2 = x2.to(opt.device)
                label = label.long().to(opt.device)

                pred = netG(x1, x2)
                acc_train += (pred.argmax(1) == label).sum().item()

            acc_train /= len(train_dataset)
            print('Train Accuracy = {}'.format(acc_train))
        print('*' * 5 + ' Valid End. ' + '*' * 5)

    # save model
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), '%s/%s_netG_epoch_%d.pth' % (opt.exp, time_str, epoch))
        print('%s/%s_netG_epoch_%d.pth has beed saved.' % (opt.exp, time_str, epoch))

    torch.save(netG.state_dict(), '%s/%s_lastest.pth' % (opt.exp, time_str))
    print('%s/%s_latest.pth has beed saved.' % (opt.exp, time_str))

trainLogger.close()
validLogger.close()