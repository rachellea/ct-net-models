#utils.py
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

import torch
import numpy as np

"""CT volume preprocessing functions"""

#############################################
# Pixel Values (on torch Tensors for speed) #-----------------------------------
#############################################
def normalize(ctvol, lower_bound, upper_bound): #Done testing
    """Clip images and normalize"""
    #formula https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    ctvol = torch.clamp(ctvol, lower_bound, upper_bound)
    ctvol = (ctvol - lower_bound) / (upper_bound - lower_bound)
    return ctvol

def torchify_pixelnorm_pixelcenter(ctvol, pixel_bounds):
    """Normalize using specified pixel_bounds and then center on the ImageNet
    mean. Used in 2019_10 dataset preparation"""
    #Cast to torch Tensor
    #use torch Tensor instead of numpy array because addition, subtraction,
    #multiplication, and division are faster in torch Tensors than np arrays
    ctvol = torch.from_numpy(ctvol).type(torch.float)
    
    #Clip Hounsfield units and normalize pixel values
    ctvol = normalize(ctvol, pixel_bounds[0], pixel_bounds[1])
    
    #Center on the ImageNet mean since you are using an ImageNet pretrained
    #feature extractor:
    ctvol = ctvol - 0.449
    return ctvol

###########
# Padding #---------------------------------------------------------------------
###########
def pad_slices(ctvol, max_slices): #Done testing
    """For <ctvol> of shape (slices, side, side) pad the slices to shape
    max_slices for output of shape (max_slices, side, side)"""
    padding_needed = max_slices - ctvol.shape[0]
    assert (padding_needed >= 0), 'Image slices exceed max_slices by'+str(-1*padding_needed)
    if padding_needed > 0:
        before_padding = int(padding_needed/2.0)
        after_padding = padding_needed - before_padding
        ctvol = np.pad(ctvol, pad_width = ((before_padding, after_padding), (0,0), (0,0)),
                     mode = 'constant', constant_values = np.amin(ctvol))
        assert ctvol.shape[0]==max_slices
    return ctvol

def pad_sides(ctvol, max_side_length): #Done testing
    """For <ctvol> of shape (slices, side, side) pad the sides to shape
    max_side_length for output of shape (slices, max_side_length,
    max_side_length)"""
    needed_padding = 0
    for side in [1,2]:
        padding_needed = max_side_length - ctvol.shape[side]
        if padding_needed > 0:
            before_padding = int(padding_needed/2.0)
            after_padding = padding_needed - before_padding
            if side == 1:
                ctvol = np.pad(ctvol, pad_width = ((0,0), (before_padding, after_padding), (0,0)),
                         mode = 'constant', constant_values = np.amin(ctvol))
                needed_padding += 1
            elif side == 2:
                ctvol = np.pad(ctvol, pad_width = ((0,0), (0,0), (before_padding, after_padding)),
                         mode = 'constant', constant_values = np.amin(ctvol))
                needed_padding += 1
    if needed_padding == 2: #if both sides needed to be padded, then they
        #should be equal (but it's possible one side or both were too large
        #in which case we wouldn't expect them to be equal)
        assert ctvol.shape[1]==ctvol.shape[2]==max_side_length
    return ctvol

def pad_volume(ctvol, max_slices, max_side_length):
    """Pad <ctvol> to a minimum size of
    [max_slices, max_side_length, max_side_length], e.g. [402, 308, 308]
    Used in 2019_10 dataset preparation"""
    if ctvol.shape[0] < max_slices:
        ctvol = pad_slices(ctvol, max_slices)
    if ctvol.shape[1] < max_side_length:
        ctvol = pad_sides(ctvol, max_side_length)
    return ctvol

###########################
# Reshaping to 3 Channels #-----------------------------------------------------
###########################
def sliceify(ctvol): #Done testing
    """Given a numpy array <ctvol> with shape [slices, square, square]
    reshape to 'RGB' [max_slices/3, 3, square, square]"""
    return np.reshape(ctvol, newshape=[int(ctvol.shape[0]/3), 3, ctvol.shape[1], ctvol.shape[2]])

def reshape_3_channels(ctvol):
    """Reshape grayscale <ctvol> to a 3-channel image
    Used in 2019_10 dataset preparation"""
    if ctvol.shape[0]%3 == 0:
        ctvol = sliceify(ctvol)
    else:
        if (ctvol.shape[0]-1)%3 == 0:
            ctvol = sliceify(ctvol[:-1,:,:])
        elif (ctvol.shape[0]-2)%3 == 0:
            ctvol = sliceify(ctvol[:-2,:,:])
    return ctvol

##################################
# Cropping and Data Augmentation #----------------------------------------------
##################################
def crop_specified_axis(ctvol, max_dim, axis): #Done testing
    """Crop 3D volume <ctvol> to <max_dim> along <axis>"""
    dim = ctvol.shape[axis]
    if dim > max_dim:
        amount_to_crop = dim - max_dim
        part_one = int(amount_to_crop/2.0)
        part_two = dim - (amount_to_crop - part_one)
        if axis == 0:
            return ctvol[part_one:part_two, :, :]
        elif axis == 1:
            return ctvol[:, part_one:part_two, :]
        elif axis == 2:
            return ctvol[:, :, part_one:part_two]
    else:
        return ctvol

def single_crop_3d_fixed(ctvol, max_slices, max_side_length):
    """Crop a single 3D volume to shape [max_slices, max_side_length,
    max_side_length]"""
    ctvol = crop_specified_axis(ctvol, max_slices, 0)
    ctvol = crop_specified_axis(ctvol, max_side_length, 1)
    ctvol = crop_specified_axis(ctvol, max_side_length, 2)
    return ctvol

def single_crop_3d_augment(ctvol, max_slices, max_side_length):
    """Crop a single 3D volume to shape [max_slices, max_side_length,
    max_side_length] with randomness in the centering and random
    flips or rotations"""
    #Introduce random padding so that the centered crop will be slightly random
    ctvol = rand_pad(ctvol)
    
    #Obtain the center crop
    ctvol = single_crop_3d_fixed(ctvol, max_slices, max_side_length)
    
    #Flip and rotate
    ctvol = rand_flip(ctvol)
    ctvol = rand_rotate(ctvol)
    
    #Make contiguous array to avoid Pytorch error
    return np.ascontiguousarray(ctvol)

def rand_pad(ctvol):
    """Introduce random padding between 0 and 15 pixels on each of the 6 sides
    of the <ctvol>"""
    randpad = np.random.randint(low=0,high=15,size=(6))
    ctvol = np.pad(ctvol, pad_width = ((randpad[0],randpad[1]), (randpad[2],randpad[3]), (randpad[4], randpad[5])),
                         mode = 'constant', constant_values = np.amin(ctvol))
    return ctvol
    
def rand_flip(ctvol):
    """Flip <ctvol> along a random axis with 50% probability"""
    if np.random.randint(low=0,high=100) < 50:
        chosen_axis = np.random.randint(low=0,high=3) #0, 1, and 2 are axis options
        ctvol =  np.flip(ctvol, axis=chosen_axis)
    return ctvol

def rand_rotate(ctvol):
    """Rotate <ctvol> some random amount axially with 50% probability"""
    if np.random.randint(low=0,high=100) < 50:
        chosen_k = np.random.randint(low=0,high=4)
        ctvol = np.rot90(ctvol, k=chosen_k, axes=(1,2))
    return ctvol

###########################################
# 2019_10 Dataset Preprocessing Sequences #-------------------------------------
###########################################
def prepare_ctvol_2019_10_dataset(ctvol, pixel_bounds, data_augment, num_channels,
                                  crop_type):
    """Pad, crop, possibly augment, reshape to correct
    number of channels, cast to torch tensor (to speed up subsequent operations),
    Clip Hounsfield units, normalize pixel values, center on the
    ImageNet mean, and return as a torch tensor (for crop_type='single')
    
    <pixel_bounds> is a list of ints e.g. [-1000,200] Hounsfield units. Used for
        pixel value clipping and normalization.
    <data_augment> is True to employ data augmentation, and False otherwise
    <num_channels> is an int, e.g. 3 to reshape the grayscale volume into
        a volume of 3-channel images
    <crop_type>: if 'single' then return the volume as one 3D numpy array."""
    max_slices = 402
    max_side_length = 420
    assert num_channels == 3 or num_channels == 1
    assert crop_type == 'single'
    
    #Padding to minimum size [max_slices, max_side_length, max_side_length]
    ctvol = pad_volume(ctvol, max_slices, max_side_length)
    
    #Cropping, and data augmentation if indicated
    if crop_type == 'single':
        if data_augment is True:
            ctvol = single_crop_3d_augment(ctvol, max_slices, max_side_length)
        else:
            ctvol = single_crop_3d_fixed(ctvol, max_slices, max_side_length)
        #Reshape to 3 channels if indicated
        if num_channels == 3:
            ctvol = reshape_3_channels(ctvol)
        #Cast to torch tensor and deal with pixel values
        output = torchify_pixelnorm_pixelcenter(ctvol, pixel_bounds)
    
    return output
