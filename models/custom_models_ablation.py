#custom_models_ablation.py
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

class CTNetModel_Ablate_RandomInitResNet(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC.
    Initialize the ResNet randomly instead of pretrained."""
    def __init__(self, n_outputs):
        super(CTNetModel_Ablate_RandomInitResNet, self).__init__()
        print('ResNet18_Batch_Ablate_RandomInitResNet: pretrained=False')
        resnet = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(134, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU())
        
        self.classifier = nn.Sequential(
            nn.Linear(16*18*5*5, 128), #7200
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
        x = x.view(batch_size,134,512,14,14)
        x = self.reducingconvs(x)
        x = x.view(batch_size, 16*18*5*5)
        x = self.classifier(x)
        return x


class CTNetModel_Ablate_PoolInsteadOf3D(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC.
    Perform pooling instead of 3D convolution."""
    def __init__(self, n_outputs):
        super(CTNetModel_Ablate_PoolInsteadOf3D, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size([1,134,512,14,14])
        self.reducingpools = nn.Sequential(
            nn.MaxPool3d(kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.MaxPool3d(kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.MaxPool3d(kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU())
        
        self.reducingpools2 = nn.Sequential(
            nn.MaxPool3d(kernel_size = (8,1,1), stride=(8,1,1), padding=0),
            nn.ReLU()
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(16*18*5*5, 128), #7200
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
        x = x.view(batch_size,134,512,14,14)
        x = self.reducingpools(x)
        #Output of reducingpools is [1, 134, 18, 5, 5]
        #We need it to be [1, 16, 18, 5, 5]
        #134 / 16 is 8.3. Do more pooling.
        assert batch_size == 1
        x = torch.squeeze(x) #size [134, 18, 5, 5]
        x = x.transpose(0,1) #size [18, 134, 5, 5]
        x = self.reducingpools2(x)
        #Output is [18, 16, 5, 5]
        x = x.transpose(0,1) #size [16, 18, 5, 5]
        x = x.unsqueeze(0) #size [1, 16, 18, 5, 5]
        x = x.contiguous()
        x = x.view(1, 16*18*5*5)
        x = self.classifier(x)
        return x