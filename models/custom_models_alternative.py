#custom_models_alternative.py
#Copyright (c) 2020 Rachel Lea Ballantyne Draelos

#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE

import copy
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms, models, utils

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class BodyConv(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC.
    3d conv used here follows recommendation to reshape to
    [batch_size, 512, 134, 14, 14] before performing 3d conv so that the
    convolution occurs over 'mini-bodies.'
    The 134 represents the height of the body, and the 14 x 14 represents the
    result of applying a particular ResNet filter (one of the 512 different
    filters.)"""
    def __init__(self, n_outputs):
        super(BodyConv, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size([1,512,134,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU())
        
        self.classifier = nn.Sequential(
            nn.Linear(16*4*5*5, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs))
        
    def forward(self, x):
        shape = list(x.size())
        batch_size = int(shape[0])
        x = x.view(batch_size*134,3,420,420)
        x = self.features(x)
        
        #the following code is correct only for batch_size==1 which occurs
        #when we have 1 CT scan per GPU
        assert batch_size == 1
        x = x.transpose(0,1).unsqueeze(0)
        x = x.contiguous()
        x = self.reducingconvs(x)
        
        #output is shape [batch_size, 16, 4, 5, 5]
        x = x.view(batch_size, 16*4*5*5)
        x = self.classifier(x)
        return x

class ThreeDConv(nn.Module):
    """3D conv then FC. No pretrained model"""
    def __init__(self, n_outputs):
        super(ThreeDConv, self).__init__()        
        #conv input [1,1,402,420,420]
        # torch.nn.Conv2d or Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size = (3,3,3), stride=(3,3,3), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(64, 128, kernel_size = (3,3,3), stride=(3,3,3), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(128, 256, kernel_size = (3,3,3), stride=(3,3,3), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(256, 512, kernel_size = (3,3,3), stride=(3,3,3), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(512, 128, kernel_size = (1,1,1), stride=(1,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(128, 64, kernel_size = (1,1,1), stride=(1,1,1), padding=0),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64*4*5*5, 128), #1024
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 96), 
            nn.ReLU(),
            nn.Dropout(0.1),
                        
            nn.Linear(96, n_outputs))
        
    def forward(self, x):
        #input x has shape [1,402,420,420]
        shape = list(x.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        x = x.unsqueeze(dim=0) #get shape [1,1,402,420,420] = [N,C,D,H,W]
        x = self.reducingconvs(x)
        x = x.view(1, 64*4*5*5)
        x = self.classifier(x)
        return x
