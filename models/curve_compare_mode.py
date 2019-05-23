import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo



class Curve_Compare(nn.Module):
    def __init__(self):
        super(Curve_Compare, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace = True),
            
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace = True),
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace = True),
            
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace = True),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            )

        self.conv6 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            )

        self.cla = nn.Sequential(
            nn.Linear(512*10, 64),  # Note : size need to adjust!
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(32, 2),
            )

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        
    def forward_sub(self, x):
        conv1 = self.maxpool(self.conv1(x))
        conv2 = self.maxpool(self.conv2(conv1))
        conv3 = self.maxpool(self.conv3(conv2))
        conv4 = self.maxpool(self.conv4(conv3))
        conv5 = self.maxpool(self.conv5(conv4))

        return conv5

    def forward(self, x1, x2):
        out1 = self.forward_sub(x1)
        out2 = self.forward_sub(x2)
        out = torch.cat((out1, out2), 1)

        conv6 = self.conv6(out)
        conv6 = conv6.view(conv6.shape[0], -1)
        
        cla = self.cla(conv6)
        
        return cla
    
        
if __name__ == '__main__':
    
    x1 = torch.rand(4, 1024)
    x2 = torch.rand(4, 1024)
    
    cure_mode = Curve_Compare()
    
    y = cure_mode(x1, x2)
    
    print(cure_mode)
    print("y = {}".format(y.shape))

