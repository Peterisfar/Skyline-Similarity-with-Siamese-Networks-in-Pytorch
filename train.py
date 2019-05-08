import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import linecache
import pandas as pd
from sklearn import preprocessing
from tqdm import trange
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# torch.cuda.set_device(2)
import time
import math
from model import *
from skyline_dataloader import *


class Config():
    training_dir = "./data/train/"
    testing_dir = "./data/test/"
    train_val_dir = "./data/train_val/"
    train_batch_size = 32
    train_number_epochs = 30
    train_lr = 0.001
    train_m = 10.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

skyline_dataset = SkylineDataset(root=Config.training_dir,
                                transform=transforms.Compose([ToTensor()]))
train_dataloader = DataLoader(skyline_dataset,
                       shuffle=True,
                       batch_size = Config.train_batch_size,
                       num_workers = 4
                       )

net = SiameseNetwork().to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(),lr = Config.train_lr)


counter = []
loss_history = []
iteration_number= 0
best_acc = 0


def train(epoch):
    net.train()
    for i, data in enumerate(train_dataloader, 0):
        line1 = data["line"][0].view(Config.train_batch_size, 1, -1)
        line2 = data["line"][1].view(Config.train_batch_size, 1, -1)
        label = data["label"]
        line1, line2, label = line1.to(device), line2.to(device), label.to(device)
        optimizer.zero_grad()
        output = net(line1, line2)
        loss_contrastive = criterion(output, label)
        loss_contrastive.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Epoch {} | {} | Current loss {} ".format(epoch, i, loss_contrastive.item()))


def test(epoch, path, save=True):
    net.eval()
    with torch.no_grad():
        filenames = os.listdir(path)
        acc_top20 = 0
        acc_top50 = 0
        acc_top100 = 0
        acc_top200 = 0

        for _ in range(5):
            for i in trange(100):
                index_target = random.randint(0, len(filenames) - 1)
                file0 = read_data_row(os.path.join(path, filenames[index_target]), 1).strip().split(" ")
                line0 = np.array(list(map(int, file0[0].split(','))))
                dist = []
                for f in filenames:
                    file1 = read_data_row(os.path.join(path, f), 1).strip().split(" ")
                    line1 = np.array(list(map(int, file1[1].split(','))))
                    line = np.hstack((line0, line1))
                    line_min, line_max = line.min(), line.max()
                    line = (line - line_min) / (line_max - line_min)
                    line1 = torch.from_numpy(line[:320].reshape(1, 1, -1))
                    line2 = torch.from_numpy(line[320:].reshape(1, 1, -1))

                    output = net(Variable(line1).float().to(device), Variable(line2).float().to(device))
                    dist.append(output.item())

                dist = pd.Series(np.array(dist)).sort_values(ascending=False)
                if index_target in dist[:20].index.tolist():
                    acc_top20 += 1
                if index_target in dist[:50].index.tolist():
                    acc_top50 += 1
                if index_target in dist[:100].index.tolist():
                    acc_top100 += 1
                if index_target in dist[:200].index.tolist():
                    acc_top200 += 1

        acc_top20 /= 500.0
        acc_top50 /= 500.0
        acc_top100 /= 500.0
        acc_top200 /= 500.0
        acc_mean = (acc_top20 + acc_top50 + acc_top100 + acc_top200) / 4
        print('epoch %d | acc_top20: %.3f | acc_top50: %.3f | acc_top100: %.3f | acc_top200: %.3f | acc_mean: %.3f' % (
        epoch,
        acc_top20,
        acc_top50,
        acc_top100,
        acc_top200,
        acc_mean))

        # saving
        if save:
            global best_acc
            if acc_mean > best_acc:
                print("saving...")
                if not os.path.isdir('./checkpoint'):
                    os.mkdir('./checkpoint')
                time_str = time.strftime("%Y-%m-%d %X", time.localtime())
                torch.save(net.state_dict(), './checkpoint/ckpt.{}.{:.2f}.pth'.format(time_str, acc_mean))
                best_acc = acc_mean

for epoch in range(0, 15):
    train(epoch)
    test(epoch, Config.train_val_dir, save=False)
    test(epoch, Config.testing_dir)
