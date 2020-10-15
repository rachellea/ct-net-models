#custom_datasets.py
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
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from . import utils

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

###################################################
# PACE Dataset for Data Stored in 2019-10-BigData #-----------------------------
###################################################
class CTDataset_2019_10(Dataset):    
    def __init__(self, setname, label_type_ld,
                 label_meanings, num_channels, pixel_bounds,
                 data_augment, crop_type,
                 selected_note_acc_files):
        """CT Dataset class that works for preprocessed data in 2019-10-BigData.
        A single example (for crop_type == 'single') is a 4D CT volume:
            if num_channels == 3, shape [134,3,420,420]
            if num_channels == 1, shape [402,420,420]
        
        Variables:
        <setname> is either 'train' or 'valid' or 'test'
        <label_type_ld> is 'disease_new'
        <label_meanings>: list of strings indicating which labels should
            be kept. Alternatively, can be the string 'all' in which case
            all labels are kept.
        <num_channels>: number of channels to reshape the image to.
            == 3 if the model uses a pretrained feature extractor.
            == 1 if the model uses only 3D convolutions.
        <pixel_bounds>: list of ints e.g. [-1000,200]
            Determines the lower bound, upper bound of pixel value clipping
            and normalization.
        <data_augment>: if True, perform data augmentation.
        <crop_type>: is 'single' for an example consisting of one 4D numpy array
        <selected_note_acc_files>: This should be a dictionary
            with key equal to setname and value that is a string. If the value
            is a path to a file, the file must be a CSV. Only note accessions
            in this file will be used. If the value is not a valid file path,
            all available note accs will be used, i.e. the model will be
            trained on the whole dataset."""
        self.setname = setname
        self.define_subsets_list()
        self.label_type_ld = label_type_ld
        self.label_meanings = label_meanings
        self.num_channels = num_channels
        self.pixel_bounds = pixel_bounds
        if self.setname == 'train':
            self.data_augment = data_augment
        else:
            self.data_augment = False
        print('For dataset',self.setname,'data_augment is',self.data_augment)
        self.crop_type = crop_type
        assert self.crop_type == 'single'
        self.selected_note_acc_files = selected_note_acc_files
        
        #Define location of the CT volumes
        self.main_clean_path = './load_dataset/fakedata'
        self.volume_log_df = pd.read_csv('./load_dataset/fakedata/CT_Scan_Preprocessing_Log_File_FINAL_SMALL.csv',header=0,index_col=0)
        
        #Get the example ids
        self.volume_accessions = self.get_volume_accessions()
                        
        #Get the ground truth labels
        self.labels_df = self.get_labels_df()
    
    # Pytorch Required Methods #------------------------------------------------
    def __len__(self):
        return len(self.volume_accessions)
        
    def __getitem__(self, idx):
        """Return a single sample at index <idx>. The sample is a Python
        dictionary with keys 'data' and 'gr_truth' for the image and label,
        respectively"""
        return self._get_pace(self.volume_accessions[idx])
    
    # Volume Accession Methods #------------------------------------------------
    def get_note_accessions(self):
        setname_file = self.selected_note_acc_files[self.setname]
        if os.path.isfile(setname_file):
            print('\tObtaining note accessions from',setname_file)
            sel_accs = pd.read_csv(setname_file,header=0)            
            assert sorted(list(set(sel_accs['Subset_Assigned'].values.tolist())))==sorted(self.subsets_list)
            note_accs = sel_accs.loc[:,'Accession'].values.tolist()
            print('\tTotal theoretical note accessions in subsets:',len(note_accs))
            return note_accs
        else: 
            print('\tObtaining note accessions from complete identifiers file')
            #Read in identifiers file, which contains note_accessions
            #Columns are MRN, Accession, Set_Assigned, Set_Should_Be, Subset_Assigned
            all_ids = pd.read_csv('./load_dataset/fakedata/all_identifiers.csv',header=0)
           
            #Extract the note_accessions
            note_accs = []
            for subset in self.subsets_list: #e.g. ['imgvalid_a','imgvalid_b']
                subset_note_accs = all_ids[all_ids['Subset_Assigned']==subset].loc[:,'Accession'].values.tolist()
                note_accs += subset_note_accs
            print('\tTotal theoretical note accessions in subsets:',len(note_accs))
            return note_accs
    
    def get_volume_accessions(self):
        note_accs = self.get_note_accessions()
        #Translate note_accessions to volume_accessions based on what data has been
        #preprocessed successfully. volume_log_df has note accessions as the
        #index, and the column 'full_filename_npz' for the volume accession.
        #The column 'status' should equal 'success' if the volume has been
        #preprocessed correctly.
        print('\tTotal theoretical volumes in whole dataset:',self.volume_log_df.shape[0])
        self.volume_log_df = self.volume_log_df[self.volume_log_df['status']=='success']
        print('\tTotal successfully preprocessed volumes in whole dataset:',self.volume_log_df.shape[0])
        volume_accs = []
        for note_acc in note_accs:
            if note_acc in self.volume_log_df.index.values.tolist():
                volume_accs.append(self.volume_log_df.at[note_acc,'full_filename_npz'])
        print('\tFinal total successfully preprocessed volumes in requested subsets:',len(volume_accs))
        #According to this thread: https://github.com/pytorch/pytorch/issues/13246
        #it is better to use a numpy array than a list to reduce memory leaks.
        return np.array(volume_accs)
    
    # Ground Truth Label Methods #----------------------------------------------
    def get_labels_df(self):
        #Get the ground truth labels based on requested label type.
        labels_df = read_in_labels(self.label_type_ld, self.setname)
        
        #Now filter the ground truth labels based on the desired label meanings:
        if self.label_meanings != 'all': #i.e. if you want to filter
            labels_df = labels_df[self.label_meanings]
        return labels_df
    
    # Fetch a CT Volume (__getitem__ implementation) #--------------------------
    def _get_pace(self, volume_acc):
        """<volume_acc> is for example RHAA12345_6.npz"""
        #Load compressed npz file: [slices, square, square]
        ctvol = np.load(os.path.join(self.main_clean_path, volume_acc))['ct']
        
        #Prepare the CT volume data (already torch Tensors)
        data = utils.prepare_ctvol_2019_10_dataset(ctvol, self.pixel_bounds, self.data_augment, self.num_channels, self.crop_type)
        
        #Get the ground truth:
        note_acc = self.volume_log_df[self.volume_log_df['full_filename_npz']==volume_acc].index.values.tolist()[0]
        gr_truth = self.labels_df.loc[note_acc, :].values
        gr_truth = torch.from_numpy(gr_truth).squeeze().type(torch.float)
        
        #When training on only one abnormality you must unsqueeze to prevent
        #a dimensions error when training the model:
        if len(self.label_meanings)==1:
            gr_truth = gr_truth.unsqueeze(0)
        
        #Create the sample
        sample = {'data': data, 'gr_truth': gr_truth, 'volume_acc': volume_acc}
        return sample
    
    # Sanity Check #------------------------------------------------------------
    def define_subsets_list(self):
        assert self.setname in ['train','valid','test']
        if self.setname == 'train':
            self.subsets_list = ['imgtrain']
        elif self.setname == 'valid':
            self.subsets_list = ['imgvalid_a']
        elif self.setname == 'test':
            self.subsets_list = ['imgtest_a','imgtest_b','imgtest_c','imgtest_d']
        print('Creating',self.setname,'dataset with subsets',self.subsets_list)

#######################
# Ground Truth Labels #---------------------------------------------------------
#######################

def read_in_labels(label_type_ld, setname):
    """Return a pandas dataframe with the dataset labels.
    Accession numbers are the index and labels (e.g. "pneumonia") are the columns.
    <setname> can be 'train', 'valid', or 'test'."""
    assert label_type_ld == 'disease_new'
    labels_file = './load_dataset/fakedata/2019-12-18_duke_disease/img'+setname+'_BinaryLabels.csv'
    return pd.read_csv(labels_file, header=0, index_col = 0)
    