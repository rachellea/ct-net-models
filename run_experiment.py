#run_experiment.py
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
import timeit
import datetime
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms, models, utils

import evaluate
from load_dataset import custom_datasets

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class DukeCTModel(object):
    def __init__(self, descriptor, custom_net, custom_net_args,
                 loss, loss_args, num_epochs, patience, batch_size, device, data_parallel,
                 use_test_set, task, old_params_dir, dataset_class, dataset_args):
        """Variables:
        <descriptor>: string describing the experiment
        <custom_net>: class defining a model
        <custom_net_args>: dictionary where keys correspond to custom net
            input arguments, and values are the desired values    
        <loss>: 'bce' for binary cross entropy
        <loss_args>: arguments to pass to the loss function if any
        <num_epochs>: int for the maximum number of epochs to train
        <patience>: number of epochs for which loss must fail to improve to
            cause early stopping
        <batch_size>: int for number of examples per batch
        <device>: int specifying which device to use, or 'all' for all devices
        <data_parallel>: if True then parallelize across available GPUs.
        <use_test_set>: if True, then run model on the test set. If False, use
            only the training and validation sets.
        <task>:
            'train_eval': train and evaluate a new model. 'evaluate' will
                always imply use of the validation set. if <use_test_set> is
                True, then 'evaluate' also includes calculation of test set
                performance for the best validation epoch.
            'predict_on_test': load a trained model and make predictions on
                the test set using that model.
        <old_params_dir>: this is only needed if <task>=='predict_on_test'. This
            is the path to the parameters that will be loaded in to the model.
        <dataset_class>: CT Dataset class for preprocessing the data
        <dataset_args>: arguments for the dataset class specifying how
            the data should be prepared."""
        self.descriptor = descriptor
        self.set_up_results_dirs()
        self.custom_net = custom_net
        self.custom_net_args = custom_net_args
        self.loss = loss
        self.loss_args = loss_args
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        print('self.batch_size=',self.batch_size)
        #num_workers is number of threads to use for data loading
        self.num_workers = int(batch_size*4) #batch_size 1 = num_workers 4. batch_size 2 = num workers 8. batch_size 4 = num_workers 16.
        print('self.num_workers=',self.num_workers)
        if self.num_workers == 1:
            print('Warning: Using only one worker will slow down data loading')
        
        #Set Device and Data Parallelism
        if device in [0,1,2,3]: #i.e. if a GPU number was specified:
            self.device = torch.device('cuda:'+str(device))
            print('using device:',str(self.device),'\ndescriptor: ',self.descriptor)
        elif device == 'all':
            self.device = torch.device('cuda')
        self.data_parallel = data_parallel
        if self.data_parallel:
            assert device == 'all' #use all devices when running data parallel
        
        #Set Task
        self.use_test_set = use_test_set
        self.task = task
        assert self.task in ['train_eval','predict_on_test']
        if self.task == 'predict_on_test':
            #overwrite the params dir that was created in the call to
            #set_up_results_dirs() with the dir you want to load from
            self.params_dir = old_params_dir
        
        #Data and Labels
        self.CTDatasetClass = dataset_class
        self.dataset_args = dataset_args
        #Get label meanings, a list of descriptive strings (list elements must
        #be strings found in the column headers of the labels file)
        self.set_up_label_meanings(self.dataset_args['label_meanings'])
        if self.task == 'train_eval':
            self.dataset_train = self.CTDatasetClass(setname = 'train', **self.dataset_args)
            self.dataset_valid = self.CTDatasetClass(setname = 'valid', **self.dataset_args)
        if self.use_test_set:
            self.dataset_test = self.CTDatasetClass(setname = 'test', **self.dataset_args)
        
        #Tracking losses and evaluation results
        self.train_loss = np.zeros((self.num_epochs))
        self.valid_loss = np.zeros((self.num_epochs))
        self.eval_results_valid, self.eval_results_test = evaluate.initialize_evaluation_dfs(self.label_meanings, self.num_epochs)
        
        #For early stopping
        self.initial_patience = patience
        self.patience_remaining = patience
        self.best_valid_epoch = 0
        self.min_val_loss = np.inf
        
        #Run everything
        self.run_model()
    
    ### Methods ###
    def set_up_label_meanings(self,label_meanings):
        if label_meanings == 'all': #get full list of all available labels
            temp = custom_datasets.read_in_labels(self.dataset_args['label_type_ld'], 'valid')
            self.label_meanings = temp.columns.values.tolist()
        else: #use the label meanings that were passed in
            self.label_meanings = label_meanings
        print('label meanings ('+str(len(self.label_meanings))+' labels total):',self.label_meanings)
        
    def set_up_results_dirs(self):
        if not os.path.isdir('results'):
            os.mkdir('results')
        self.results_dir = os.path.join('results',datetime.datetime.today().strftime('%Y-%m-%d')+'_'+self.descriptor)
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
        self.params_dir = os.path.join(self.results_dir,'params')
        if not os.path.isdir(self.params_dir):
            os.mkdir(self.params_dir)
        self.backup_dir = os.path.join(self.results_dir,'backup')
        if not os.path.isdir(self.backup_dir):
            os.mkdir(self.backup_dir)
        
    def run_model(self):
        if self.data_parallel:
            self.model = nn.DataParallel(self.custom_net(**self.custom_net_args)).to(self.device)
        else:
            self.model = self.custom_net(**self.custom_net_args).to(self.device)
        self.sigmoid = torch.nn.Sigmoid()
        self.set_up_loss_function()
        
        momentum = 0.99
        print('Running with optimizer lr=1e-3, momentum='+str(round(momentum,2))+' and weight_decay=1e-7')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 1e-3, momentum=momentum, weight_decay=1e-7)
        
        train_dataloader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers)
        valid_dataloader = DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)
        
        if self.task == 'train_eval':
            for epoch in range(self.num_epochs):
                t0 = timeit.default_timer()
                self.train(train_dataloader, epoch)
                self.valid(valid_dataloader, epoch)
                self.save_evals(epoch)
                if self.patience_remaining <= 0:
                    print('No more patience (',self.initial_patience,') left at epoch',epoch)
                    print('--> Implementing early stopping. Best epoch was:',self.best_valid_epoch)
                    break
                t1 = timeit.default_timer()
                self.back_up_model_every_ten(epoch)
                print('Epoch',epoch,'time:',round((t1 - t0)/60.0,2),'minutes')  
        if self.use_test_set: self.test(DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers))
        self.save_final_summary()
    
    def set_up_loss_function(self):
        if self.loss == 'bce': 
            self.loss_func = nn.BCEWithLogitsLoss() #includes application of sigmoid for numerical stability
    
    def train(self, dataloader, epoch):
        model = self.model.train()
        epoch_loss, pred_epoch, gr_truth_epoch, volume_accs_epoch = self.iterate_through_batches(model, dataloader, epoch, training=True)
        self.train_loss[epoch] = epoch_loss
        self.plot_roc_and_pr_curves('train', epoch, pred_epoch, gr_truth_epoch)
        print("{:5s} {:<3d} {:11s} {:.3f}".format('Epoch', epoch, 'Train Loss', epoch_loss))
        
    def valid(self, dataloader, epoch):
        model = self.model.eval()
        with torch.no_grad():
            epoch_loss, pred_epoch, gr_truth_epoch, volume_accs_epoch = self.iterate_through_batches(model, dataloader, epoch, training=False)
        self.valid_loss[epoch] = epoch_loss
        self.eval_results_valid = evaluate.evaluate_all(self.eval_results_valid, epoch,
            self.label_meanings, gr_truth_epoch, pred_epoch)
        self.early_stopping_check(epoch, pred_epoch, gr_truth_epoch, volume_accs_epoch)
        print("{:5s} {:<3d} {:11s} {:.3f}".format('Epoch', epoch, 'Valid Loss', epoch_loss))
    
    def early_stopping_check(self, epoch, val_pred_epoch, val_gr_truth_epoch, val_volume_accs_epoch):
        """Check whether criteria for early stopping are met and update
        counters accordingly"""
        val_loss = self.valid_loss[epoch]
        if (val_loss < self.min_val_loss) or epoch==0: #then save parameters
            self.min_val_loss = val_loss
            check_point = {'params': self.model.state_dict(),                            
                           'optimizer': self.optimizer.state_dict()}
            torch.save(check_point, os.path.join(self.params_dir, self.descriptor))                                 
            self.best_valid_epoch = epoch
            self.patience_remaining = self.initial_patience
            print('model saved, val loss',val_loss)
            self.plot_roc_and_pr_curves('valid', epoch, val_pred_epoch, val_gr_truth_epoch)
            self.save_all_pred_probs('valid', epoch, val_pred_epoch, val_gr_truth_epoch, val_volume_accs_epoch)
        else:
            self.patience_remaining -= 1
    
    def back_up_model_every_ten(self, epoch):
        """Back up the model parameters every 10 epochs"""
        if epoch % 10 == 0:
            check_point = {'params': self.model.state_dict(),                            
                           'optimizer': self.optimizer.state_dict()}
            torch.save(check_point, os.path.join(self.backup_dir, self.descriptor+'_ep_'+str(epoch)))   
    
    def test(self, dataloader):
        epoch = self.best_valid_epoch
        if self.data_parallel:
            model = nn.DataParallel(self.custom_net(**self.custom_net_args)).to(self.device).eval()
        else:
            model = self.custom_net(**self.custom_net_args).to(self.device).eval()
        params_path = os.path.join(self.params_dir,self.descriptor)
        print('For test set predictions, loading model params from params_path=',params_path)
        check_point = torch.load(params_path)
        model.load_state_dict(check_point['params'])
        with torch.no_grad():
            epoch_loss, pred_epoch, gr_truth_epoch, volume_accs_epoch = self.iterate_through_batches(model, dataloader, epoch, training=False)
        self.eval_results_test = evaluate.evaluate_all(self.eval_results_test, epoch,
            self.label_meanings, gr_truth_epoch, pred_epoch)
        self.plot_roc_and_pr_curves('test', epoch, pred_epoch, gr_truth_epoch)
        self.save_all_pred_probs('test', epoch, pred_epoch, gr_truth_epoch, volume_accs_epoch)
        print("{:5s} {:<3d} {:11s} {:.3f}".format('Epoch', epoch, 'Test Loss', epoch_loss))
    
    def iterate_through_batches(self, model, dataloader, epoch, training):
        epoch_loss = 0
        
        #Initialize numpy arrays for storing results. examples x labels
        #Do NOT use concatenation, or else you will have memory fragmentation.
        num_examples = len(dataloader.dataset)
        num_labels = len(self.label_meanings)
        pred_epoch = np.zeros([num_examples,num_labels])
        gr_truth_epoch = np.zeros([num_examples,num_labels])
        volume_accs_epoch = np.empty(num_examples,dtype='U32') #need to use U32 to allow string of length 32
        
        for batch_idx, batch in enumerate(dataloader):
            data, gr_truth = self.move_data_to_device(batch)
            self.optimizer.zero_grad()
            if training:
                out = model(data)
            else:
                with torch.set_grad_enabled(False):
                   out = model(data)
            loss = self.loss_func(out, gr_truth)
            if training:
                loss.backward()
                self.optimizer.step()   
            
            epoch_loss += loss.item()
            torch.cuda.empty_cache()
            
            #Save predictions and ground truth across batches
            pred = self.sigmoid(out.data).detach().cpu().numpy()
            gr_truth = gr_truth.detach().cpu().numpy()
            
            start_row = batch_idx*self.batch_size
            stop_row = min(start_row + self.batch_size, num_examples)
            pred_epoch[start_row:stop_row,:] = pred #pred_epoch is e.g. [25355,80] and pred is e.g. [1,80] for a batch size of 1
            gr_truth_epoch[start_row:stop_row,:] = gr_truth #gr_truth_epoch has same shape as pred_epoch
            volume_accs_epoch[start_row:stop_row] = batch['volume_acc'] #volume_accs_epoch stores the volume accessions in the order they were used
            
            #the following line to empty the cache is necessary in order to
            #reduce memory usage and avoid OOM error:
            torch.cuda.empty_cache() 
        return epoch_loss, pred_epoch, gr_truth_epoch, volume_accs_epoch
    
    def move_data_to_device(self, batch):
        """Move data and ground truth to device."""
        assert self.dataset_args['crop_type'] == 'single'
        if self.dataset_args['crop_type'] == 'single':
            data = batch['data'].to(self.device)
        
        #Ground truth to device
        gr_truth = batch['gr_truth'].to(self.device)
        return data, gr_truth
    
    def plot_roc_and_pr_curves(self, setname, epoch, pred_epoch, gr_truth_epoch):
        outdir = os.path.join(self.results_dir,'curves')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        evaluate.plot_roc_curve_multi_class(label_meanings=self.label_meanings,
                    y_test=gr_truth_epoch, y_score=pred_epoch,
                    outdir = outdir, setname = setname, epoch = epoch)
        evaluate.plot_pr_curve_multi_class(label_meanings=self.label_meanings,
                    y_test=gr_truth_epoch, y_score=pred_epoch,
                    outdir = outdir, setname = setname, epoch = epoch)
    
    def save_all_pred_probs(self, setname, epoch, pred_epoch, gr_truth_epoch, volume_accs_epoch):
        outdir = os.path.join(self.results_dir,'pred_probs')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        (pd.DataFrame(pred_epoch,columns=self.label_meanings,index=volume_accs_epoch.tolist())).to_csv(os.path.join(outdir, setname+'_predprob_ep'+str(epoch)+'.csv'))
        (pd.DataFrame(gr_truth_epoch,columns=self.label_meanings,index=volume_accs_epoch.tolist())).to_csv(os.path.join(outdir, setname+'_grtruth_ep'+str(epoch)+'.csv'))
        
    def save_evals(self, epoch):
        evaluate.save(self.eval_results_valid, self.results_dir, self.descriptor+'_valid')
        if self.use_test_set: evaluate.save(self.eval_results_test, self.results_dir, self.descriptor+'_test')
        evaluate.plot_learning_curves(self.train_loss, self.valid_loss, self.results_dir, self.descriptor)
               
    def save_final_summary(self):
        evaluate.save_final_summary(self.eval_results_valid, self.best_valid_epoch, 'valid', self.results_dir)
        if self.use_test_set: evaluate.save_final_summary(self.eval_results_test, self.best_valid_epoch, 'test', self.results_dir)
        evaluate.clean_up_output_files(self.best_valid_epoch, self.results_dir)
        