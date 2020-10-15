#unit_tests.py
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

import os
import copy
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import evaluate
from load_dataset import custom_datasets
from load_dataset import utils

#################
# Test utils.py #---------------------------------------------------------------
#################
x = np.array([[[1,2],[3,4]],
    [[-1,-2],[-3,-4]],
    [[6,2],[1,9]],
    [[5,3],[1,1]],
    [[8,8],[4,7]]])  #shape (5, 2, 2)

def test_pad_slices_and_sides():
    global x
    out = utils.pad_slices(x, 6)
    cor = np.array([[[1,2],[3,4]],
    [[-1,-2],[-3,-4]],
    [[6,2],[1,9]],
    [[5,3],[1,1]],
    [[8,8],[4,7]],
    [[-4,-4],[-4,-4]]])
    assert arrays_equal(out, cor)
    
    z = copy.deepcopy(x)
    z[0,0,0] = -10
    out2 =  utils.pad_sides(z, 3)
    cor2 = np.array([[[-10,2,-10],[3,4,-10],[-10,-10,-10]],
    [[-1,-2,-10],[-3,-4,-10],[-10,-10,-10]],
    [[6,2,-10],[1,9,-10],[-10,-10,-10]],
    [[5,3,-10],[1,1,-10],[-10,-10,-10]],
    [[8,8,-10],[4,7,-10],[-10,-10,-10]]])
    assert arrays_equal(out2, cor2)
    print('Passed test_pad_slices_and_sides()')

def test_sliceify():
    global x
    out = utils.sliceify(utils.pad_slices(x, 6))
    cor = np.array([[[[1,2],[3,4]],
    [[-1,-2],[-3,-4]],
    [[6,2],[1,9]]],

    [[[5,3],[1,1]],
    [[8,8],[4,7]],
    [[-4,-4],[-4,-4]]]])
    assert arrays_equal(out, cor)
    assert cor.shape == (2, 3, 2, 2)
    print('Passed test_sliceify()')

def test_normalize():
    lower_bound = -1000
    upper_bound = 200
    test = np.reshape(np.array([500,3000,-150,-1000,-1500,130]), (6,1,1))
    out = utils.normalize(torch.Tensor(test), lower_bound = lower_bound, upper_bound = upper_bound)
    cor = np.reshape(np.array([1,1,0.70833333,0,0,0.941666666]), (6,1,1))
    assert arrays_equal(out.numpy(), cor)
    print('Passed test_normalize()')

def test_crop_specified_axis():
    ctvol = np.array([[[1],[2],[3]],
                      [[4],[5],[6]],
                      [[7],[8],[9]]])
    out = utils.crop_specified_axis(copy.deepcopy(ctvol), max_dim=2, axis=0)
    cor = np.array([[[1],[2],[3]],
                    [[4],[5],[6]]])
    assert arrays_equal(out, cor)
    out2 = utils.crop_specified_axis(copy.deepcopy(ctvol), max_dim=1, axis=1)
    cor2 = np.array([[[2]],[[5]],[[8]]])
    assert arrays_equal(out2,cor2)
    
    ctvol = np.array([[  [0,1,2], [3,4,5]  ],
                      [  [6,7,8], [9,10,11]  ],
                      [  [12,13,14], [15,16,17]  ]])
    out3 = utils.crop_specified_axis(copy.deepcopy(ctvol), max_dim=2, axis=2)
    cor3 = np.array([[  [0,1], [3,4]  ],
                      [  [6,7], [9,10]  ],
                      [  [12,13], [15,16]  ]])
    assert arrays_equal(out3,cor3)
    print('Passed test_crop_specified_axis()')

###################
# Test evalute.py #-------------------------------------------------------------
###################
def test_calculate_top_k_accuracy():
    tl1 = np.array([0,0,0,0,1])
    pp1 = np.array([0,0,0,0,0.99])
    assert (1 - evaluate.calculate_top_k_accuracy(tl1, pp1, 1)) < 1e-6
    assert (1 - evaluate.calculate_top_k_accuracy(tl1, pp1, 15)) < 1e-6
    
    tl2 = np.array([1,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1])
    pp2 = np.array([0.91,0.22,0.11,0.33,0.98,0.75,0.36,0.28,0.02,0.995,
                    0.55,0.87,0.03,0.40,0.41,0.67,0.25,0.39,0.08,0.833,0.74,0.765])
    assert (1 - evaluate.calculate_top_k_accuracy(tl2, pp2, 1)) < 1e-6
    assert (0.75 - evaluate.calculate_top_k_accuracy(tl2, pp2, 4)) < 1e-6
    assert (8.0/9.0 - evaluate.calculate_top_k_accuracy(tl2, pp2, 14)) < 1e-6
    assert (1 - evaluate.calculate_top_k_accuracy(tl2, pp2, 20)) < 1e-6
    print('Passed test_calculate_top_k_accuracy()')

###########################
# Test model-related code #-----------------------------------------------------
###########################
def test_reshape_and_view():
    """Ensure that reshaping and viewing using numpy and torch has the desired
    effects. Needed in order to use pretrained 2D feature extractor on a 3D
    network"""
    device = torch.device('cuda:0')
    x_orig = np.random.rand(6,140,3,50,50) #batch size, height, channels, squareside, squareside
    x = torch.from_numpy(x_orig).squeeze().type(torch.float)
    
    #does torch view reverse numpy reshape?:
    s = x.shape
    x = np.reshape(x, (s[0]*s[1],s[2],s[3],s[4]))
    x = x.to(device)
    x = x.view(6,140,3,50,50)
    xout1 = x.cpu().numpy()
    assert arrays_equal(xout1, x_orig)
    
    #Does torch reverse itself?
    x = x.view(6,140*3*50*50)
    x = x.view(6,140,3,50,50)
    xout2 = x.cpu().numpy()
    assert arrays_equal(xout2, x_orig)
    print('Passed test_reshape_and_view()')

def test_reshape_and_view_resnet18_batch():
    #batch size 2
    x = torch.Tensor(np.random.rand(2,13,3,42,42))
    shape = list(x.size())
    batch_size = int(shape[0])
    y = x.view(batch_size*13, 3, 42, 42)
    y = y.view(batch_size, 13, 3, 42, 42)
    assert int((x == y).all())==1
    z = x.view(batch_size,13*3*42*42)
    z = z.view(batch_size,13,3,42,42)
    assert int((z == x).all())==1
    #batch size 1
    x = torch.Tensor(np.random.rand(1,13,3,42,42))
    shape = list(x.size())
    batch_size = int(shape[0])
    y = x.view(batch_size*13,3,42,42)
    y = y.view(batch_size, 13, 3, 42, 42)
    assert int((x==y).all())==1
    z = x.view(batch_size,13*3*42*42)
    z = z.view(batch_size,13,3,42,42)
    assert int((z==x).all())==1
    print('Passed test_reshape_and_view_resnet18_batch()')

def test_reshape_and_view_bodyconv():
    #Basic test
    newx = torch.Tensor(np.random.rand(13,51,14,14))
    newx2 = newx.transpose(0,1).unsqueeze(0)
    newx2 = newx2.squeeze(0).transpose(0,1)
    assert int((newx==newx2).all())==1
    
    mini = torch.Tensor(np.array([[[[5, 8],[3, 7]],
         [[2, 2],[8, 3]],
         [[1, 9],[4, 5]],
         [[3, 9],[7, 9]],
         [[3, 2],[6, 6]]],

        [[[5, 8],[4, 8]],
         [[7, 7],[8, 6]],
         [[6, 1],[9, 9]],
         [[4, 9],[6, 6]],
         [[2, 2],[6, 4]]],

        [[[4, 8],[2, 5]],
         [[5, 7],[9, 8]],
         [[1, 9],[4, 4]],
         [[2, 8],[1, 3]],
         [[8, 8],[1, 3]]]]))
    output = mini.transpose(0,1).unsqueeze(0)
    correct = torch.Tensor(np.array([[[[[5, 8],[3, 7]],
          [[5, 8],[4, 8]],
          [[4, 8],[2, 5]]],

         [[[2, 2],[8, 3]],
          [[7, 7],[8, 6]],
          [[5, 7],[9, 8]]],

         [[[1, 9],[4, 5]],
          [[6, 1],[9, 9]],
          [[1, 9],[4, 4]]],

         [[[3, 9],[7, 9]],
          [[4, 9],[6, 6]],
          [[2, 8],[1, 3]]],

         [[[3, 2],[6, 6]],
          [[2, 2],[6, 4]],
          [[8, 8],[1, 3]]]]]))
    assert int((output==correct).all())==1
    print('Passed test_reshape_and_view_bodyconv()')

def test_reshape_and_view_pool():
    #The right way to think about it is the effective kernel size across
    #the 512 dimension, which is in fact 3*3*3=27 (because the kernel size
    #is 3 in the 512 direction and its stride in that direction is 3.) 
    #Don't get distracted by reducingpools2 which are across the 134 dimension.
    #So, we actually have (for the 512 direction):
    #0-27, 27-54, 54-81, 81-108, 108-135, 135-162, 162-189, 189-216, 216-243,
    #243-270, 270-297, 297-324, 324-351, 351-378, 378-405, 405-432, 432-459,
    #459-486, 486-513
    x = np.zeros([1,134,512,14,14],dtype='int')
    x[:,:,14,:,:] = 0; x[:,:,42,:,:] = 1
    x[:,:,70,:,:] = 2; x[:,:,98,:,:] = 3
    x[:,:,126,:,:] = 4; x[:,:,154,:,:] = 5
    x[:,:,182,:,:] = 6; x[:,:,210,:,:] = 7
    x[:,:,238,:,:] = 8; x[:,:,266,:,:] = 9
    x[:,:,294,:,:] = 10; x[:,:,322,:,:] = 11
    x[:,:,348,:,:] = 12; x[:,:,372,:,:] = 13
    x[:,:,400,:,:] = 14; x[:,:,420,:,:] = 15
    x[:,:,440,:,:] = 16; x[:,:,465,:,:] = 17
    #27*18 = 486. If we put any value into 486 or higher, it won't be found in
    #the final result, which honestly disturbs me.
    #But if we put anything huge into 485 or lower, it will show up in the 17 slot
    #in the final output. 
    x[:,:,486,:,:] = 99999 #doesn't show up in final output which is disturbing
    x = torch.Tensor(x)
    
    reducingpools = nn.Sequential(
        nn.MaxPool3d(kernel_size = (3,3,3), stride=(3,1,1), padding=0),
        nn.ReLU(),
    
        nn.MaxPool3d(kernel_size = (3,3,3), stride=(3,1,1), padding=0),
        nn.ReLU(),
    
        nn.MaxPool3d(kernel_size = (3,2,2), stride=(3,2,2), padding=0),
        nn.ReLU())
        
    reducingpools2 = nn.Sequential(
        nn.MaxPool3d(kernel_size = (8,1,1), stride=(8,1,1), padding=0),
        nn.ReLU())
    
    shape = list(x.size())
    batch_size = int(shape[0])
    x = reducingpools(x)
    assert batch_size == 1
    x = torch.squeeze(x) #size [134, 18, 5, 5]
    x = x.transpose(0,1) #size [18, 134, 5, 5]
    x = reducingpools2(x) #Output is [18, 16, 5, 5]
    x = x.transpose(0,1) #size [16, 18, 5, 5]
    x = x.unsqueeze(0) #size [1, 16, 18, 5, 5]
    x = x.contiguous().numpy()
    
    for number in range(0,18):
        print(number)
        selected = list(set(x[:,:,number,:,:].flatten().tolist()))
        print(selected)
        assert len(selected)==1, 'len='+str(len(selected))
        assert int(selected[0]) == (number), 'int(selected[0])='+str(int(selected[0]))+' number+1='+str(number)
    print('Passed test_reshape_and_view_pool()')

########
# Meta #------------------------------------------------------------------------
########
#Function for testing equality of dataframes
def dfs_equal(df1, df2):
    assert arrays_equal(df1.values, df2.values, tol = 0)
    assert df1.columns.values.tolist()==df2.columns.values.tolist()
    assert df1.index.values.tolist()==df2.index.values.tolist()
    return True

#Function for testing equality of arrays
def arrays_equal(output, correct, tol =  1e-6):
    """Check if arrays are equal within tolerance <tol>
    Note that if <tol>==0 then check that arrays are identical."""
    assert output.shape == correct.shape
    max_difference = np.amax(np.absolute(output - correct))
    if tol == 0:
        assert max_difference == 0
    else:
        assert max_difference < tol
    return True
    
if __name__ == '__main__':
    test_reshape_and_view_bodyconv()
    test_reshape_and_view_resnet18_batch()
    test_reshape_and_view_pool()
    test_calculate_top_k_accuracy()
    test_normalize()
    test_pad_slices_and_sides()
    test_sliceify()
    test_reshape_and_view()
    test_crop_specified_axis()
