#evaluate.py
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

#Imports
import os
import copy
import time
import bisect
import shutil
import operator
import itertools
import numpy as np
import pandas as pd
import sklearn.metrics
from scipy import interp
from itertools import cycle

import matplotlib
matplotlib.use('agg') #so that it does not attempt to display via SSH
import seaborn
import matplotlib.pyplot as plt
plt.ioff() #turn interactive plotting off

#suppress numpy warnings
import warnings
warnings.filterwarnings('ignore')

#######################
# Reporting Functions #---------------------------------------------------------
#######################
def initialize_evaluation_dfs(all_labels, num_epochs):
    """Create empty "eval_dfs_dict"
    Variables
    <all_labels>: a list of strings describing the labels in order
    <num_epochs>: int for total number of epochs"""
    if len(all_labels)==2:
        index = [all_labels[1]]
        numrows = 1
    else:
        index = all_labels
        numrows = len(all_labels)
    #Initialize empty pandas dataframe to store evaluation results across epochs
    #for accuracy, AUROC, and AP
    result_df = pd.DataFrame(data=np.zeros((numrows, num_epochs)),
                            index = index,
                            columns = ['epoch_'+str(n) for n in range(0,num_epochs)])
    #Initialize empty pandas dataframe to store evaluation results for top k
    top_k_result_df = pd.DataFrame(np.zeros((len(all_labels), num_epochs)),
                                   index=[x for x in range(1,len(all_labels)+1)], #e.g. 1,...,64 for len(all_labels)=64
                                   columns = ['epoch_'+str(n) for n in range(0,num_epochs)])
    
    #Make eval results dictionaries
    eval_results_valid = {'accuracy':copy.deepcopy(result_df),
        'auroc':copy.deepcopy(result_df),
        'avg_precision':copy.deepcopy(result_df),
        'top_k':top_k_result_df}
    eval_results_test = copy.deepcopy(eval_results_valid)
    return eval_results_valid, eval_results_test

def save(eval_dfs_dict, results_dir, descriptor):
    """Variables
    <eval_dfs_dict> is a dict of pandas dataframes
    <descriptor> is a string"""
    for k in eval_dfs_dict.keys():
        eval_dfs_dict[k].to_csv(os.path.join(results_dir, descriptor+'_'+k+'_Table.csv'))
    
def save_final_summary(eval_dfs_dict, best_valid_epoch, setname, results_dir):
    """Save to overall df and print summary of best epoch."""
    #final_descriptor is e.g. '2019-11-15-awesome-model_epoch15
    final_descriptor = results_dir.replace('results/','')+'_epoch'+str(best_valid_epoch)
    if setname=='valid': print('***Summary for',setname,results_dir,'***')
    for metricname in list(eval_dfs_dict.keys()):
        #metricnames are accuracy, auroc, avg_precision, and top_k.
        #df holds a particular metric for the particular model we just ran.
        #for accuracy, auroc, and avg_precision, df index is diseases, columns are epochs.
        #for top_k, df index is the k value (an int) and columns are epochs.
        df = eval_dfs_dict[metricname]
        #all_df tracks results of all models in one giant table.
        #all_df has index of diseases or k value, and columns which are particular models.
        all_df_path = os.path.join('results',setname+'_'+metricname+'_all.csv') #e.g. valid_accuracy_all.csv
        if os.path.isfile(all_df_path):
            all_df = pd.read_csv(all_df_path,header=0,index_col=0)
            all_df[final_descriptor] = np.nan
        else: #all_df doesn't exist yet - create it.
            all_df = pd.DataFrame(np.empty((df.shape[0],1)),
                                  index = df.index.values.tolist(),
                                  columns = [final_descriptor])
        #Print off and save results for best_valid_epoch
        if setname=='valid': print('\tEpoch',best_valid_epoch,metricname)
        for label in df.index.values:
            #print off to console
            value = df.at[label,'epoch_'+str(best_valid_epoch)]
            if setname=='valid': print('\t\t',label,':',str( round(value, 3) ))
            #save in all_df
            all_df.at[label,final_descriptor] = value
        all_df.to_csv(all_df_path,header=True,index=True)

def clean_up_output_files(best_valid_epoch, results_dir):
    """Delete output files that aren't from the best epoch"""
    #Delete all the backup parameters (they take a lot of space and you do not
    #need to have them)
    shutil.rmtree(os.path.join(results_dir,'backup'))
    #Delete all the extra output files:
    for subdir in ['heatmaps','curves','pred_probs']:
        #Clean up saved ROC and PR curves
        fullpath = os.path.join(results_dir,subdir)
        if os.path.exists(fullpath): #e.g. there may not be a heatmaps dir for a non-bottleneck model
            allfiles = os.listdir(fullpath)
            for filename in allfiles:
                if str(best_valid_epoch) not in filename:
                    os.remove(os.path.join(fullpath,filename))
    print('Output files all clean')
    
#########################
# Calculation Functions #-------------------------------------------------------
#########################        
def evaluate_all(eval_dfs_dict, epoch, label_meanings,
                 true_labels_array, pred_probs_array):
    """Fill out the pandas dataframes in the dictionary <eval_dfs_dict>
    which is created in cnn.py. <epoch> and <which_label> are used to index into
    the dataframe for the metric. Metrics calculated for the provided vectors
    are: accuracy, AUC, partial AUC (threshold 0.2), and average precision.
    If <subjective> is set to True, additional metrics will be calculated
    (confusion matrix, sensitivity, specificity, PPV, NPV.)
    
    Variables:
    <all_eval_results> is a dictionary of pandas dataframes created in cnn.py
    <epoch> is an integer indicating which epoch it is, starting from epoch 1
    <true_labels_array>: array of true labels. examples x labels
    <pred_probs_array>: array of predicted probabilities. examples x labels"""
    #Accuracy, AUROC, and AP (iter over labels)
    for label_number in range(len(label_meanings)):
        which_label = label_meanings[label_number] #descriptive string for the label
        true_labels = true_labels_array[:,label_number]
        pred_probs = pred_probs_array[:,label_number]
        pred_labels = (pred_probs>=0.5).astype(dtype='int') #decision threshold of 0.5
        
        #Accuracy and confusion matrix (dependent on decision threshold)
        (eval_dfs_dict['accuracy']).at[which_label, 'epoch_'+str(epoch)] = compute_accuracy(true_labels, pred_labels)
        #confusion_matrix, sensitivity, specificity, ppv, npv = compute_confusion_matrix(true_labels, pred_labels)
        
        #AUROC and AP (sliding across multiple decision thresholds)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true = true_labels,
                                         y_score = pred_probs,
                                         pos_label = 1)
        (eval_dfs_dict['auroc']).at[which_label, 'epoch_'+str(epoch)] = sklearn.metrics.auc(fpr, tpr)
        (eval_dfs_dict['avg_precision']).at[which_label, 'epoch_'+str(epoch)] = sklearn.metrics.average_precision_score(true_labels, pred_probs)
    
    #Top k eval metrics (iter over examples)
    eval_dfs_dict['top_k'] = evaluate_top_k(eval_dfs_dict['top_k'],
                                    epoch, true_labels_array, pred_probs_array)
    return eval_dfs_dict

#################
# Top K Metrics #---------------------------------------------------------------
#################
def evaluate_top_k(eval_top_k_df, epoch, true_labels_array,
                   pred_probs_array):
    """<eval_top_k_df> is a pandas dataframe with epoch number as columns and
        k values as rows, where k is an integer"""
    num_labels = true_labels_array.shape[1] #e.g. 64
    total_examples = true_labels_array.shape[0]
    vals = [0 for x in range(1,num_labels+2)] #e.g. length 65 list but the index of the last element is 64 for num_labels=64
    for example_number in range(total_examples):
        #iterate through individual examples (predictions for an individual CT)
        #rather than iterating through predicted labels
        true_labels = true_labels_array[example_number,:]
        pred_probs = pred_probs_array[example_number,:]
        for k in range(1,num_labels+1): #e.g. 1,...,64
            previous_value = vals[k]
            incremental_update = calculate_top_k_accuracy(true_labels, pred_probs, k)
            new_value = previous_value + incremental_update
            vals[k] = new_value
    #Now update the dataframe. Should reach 100% performance by the end.
    for k in range(1,num_labels+1):
        eval_top_k_df.at[k,'epoch_'+str(epoch)] = vals[k]/total_examples
    
    ##Now average over all the examples
    #eval_top_k_df.loc[:,'epoch_'+str(epoch)] = eval_top_k_df.loc[:,'epoch_'+str(epoch)] / total_examples
    return eval_top_k_df

def calculate_top_k_accuracy(true_labels, pred_probs, k):
    k = min(k, len(true_labels)) #avoid accessing array elements that don't exist
    #argpartition described here: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    #get the indices of the largest k probabilities
    ind = np.argpartition(pred_probs, -1*k)[-1*k:]
    #now figure out what percent of these top predictions were equal to 1 in the
    #true_labels.
    #Note that the denominator should not exceed the number of true labels, to
    #avoid penalizing the model inappropriately:
    denom = min(k, np.sum(true_labels))
    if denom == 0: #because np.sum(true_labels) is 0
        #super important! must return 1 to avoid dividing by 0 and producing nan
        #we don't return 0 because then the model can never get perfect performance
        #even at k=num_labels because it'll get 0 for anything that has no labels
        return 1 
    else:
        return float(np.sum(true_labels[ind]))/denom

######################
# Accuracy and AUROC #----------------------------------------------------------
######################
def compute_accuracy(true_labels, labels_pred):
    """Print and save the accuracy of the model on the dataset"""    
    correct = (true_labels == labels_pred)
    correct_sum = correct.sum()
    return (float(correct_sum)/len(true_labels))

def compute_confusion_matrix(true_labels, labels_pred):
    """Return the confusion matrix"""
    cm = sklearn.metrics.confusion_matrix(y_true=true_labels,
                          y_pred=labels_pred)
    if cm.size < 4: #cm is too small to calculate anything
        return np.nan, np.nan, np.nan, np.nan, np.nan
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    sensitivity = float(true_pos)/(true_pos + false_neg)
    specificity = float(true_neg)/(true_neg + false_pos)
    ppv = float(true_pos)/(true_pos + false_pos)
    npv = float(true_neg)/(true_neg + false_neg)
    
    return((str(cm).replace("\n","_")), sensitivity, specificity, ppv, npv)

def compute_partial_auroc(fpr, tpr, thresh = 0.2, trapezoid = False, verbose=False):
    fpr_thresh, tpr_thresh = get_fpr_tpr_for_thresh(fpr, tpr, thresh)
    if len(fpr_thresh) < 2:#can't calculate an AUC with only 1 data point
        return np.nan 
    if verbose:
        print('fpr: '+str(fpr))
        print('fpr_thresh: '+str(fpr_thresh))
        print('tpr: '+str(tpr))
        print('tpr_thresh: '+str(tpr_thresh))
    return sklearn.metrics.auc(fpr_thresh, tpr_thresh)

def get_fpr_tpr_for_thresh(fpr, tpr, thresh):
    """The <fpr> and <tpr> are already sorted according to threshold (which is
    sorted from highest to lowest, and is NOT the same as <thresh>; threshold
    is the third output of sklearn.metrics.roc_curve and is a vector of the
    thresholds used to calculate FPR and TPR). This function figures out where
    to bisect the FPR so that the remaining elements are no greater than
    <thresh>. It bisects the TPR in the same place."""
    p = (bisect.bisect_left(fpr, thresh)-1) #subtract one so that the FPR
    #of the remaining elements is NO GREATER THAN <thresh>
    return fpr[: p + 1], tpr[: p + 1]

######################
# Plotting Functions #----------------------------------------------------------
######################
def plot_pr_and_roc_curves(results_dir, label_meanings, true_labels, pred_probs,
                           epoch):
    #Plot Precision Recall Curve
    
    #Plot ROC Curve
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true = true_labels,
                                         y_score = pred_probs,
                                         pos_label = 1)
    plot_roc_curve(fpr, tpr, epoch, outfilepath)


def plot_roc_curve_multi_class(label_meanings, y_test, y_score, 
                               outdir, setname, epoch):
    """<label_meanings>: list of strings, one for each label
    <y_test>: matrix of ground truth
    <y_score>: matrix of predicted probabilities
    <outdir>: directory to save output file
    <setname>: string e.g. 'train' 'valid' or 'test'
    <epoch>: int for epoch"""
    #Modified from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    n_classes = len(label_meanings)
    lw = 2
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
    
    #make order df. (note that roc_auc is a dictionary with ints as keys
    #and AUCs as values. 
    order = pd.DataFrame(np.zeros((n_classes,1)), index = [x for x in range(n_classes)],
                         columns = ['roc_auc'])
    for i in range(n_classes):
        order.at[i,'roc_auc'] = roc_auc[i]
    order = order.sort_values(by='roc_auc',ascending=False)
    
    #Plot all ROC curves
    #Plot in order of the rainbow colors, from highest AUC to lowest AUC
    plt.figure()
    colors_list = ['palevioletred','darkorange','yellowgreen','olive','deepskyblue','royalblue','navy']
    curves_plotted = 0
    for i in order.index.values.tolist()[0:10]: #only plot the top ten so the plot is readable
        color_idx = curves_plotted%len(colors_list) #cycle through the colors list in order of colors
        color = colors_list[color_idx]
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{:5s} (area {:0.2f})'.format(label_meanings[i], roc_auc[i]))
        curves_plotted+=1
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(setname.lower().capitalize()+' ROC Epoch '+str(epoch))
    plt.legend(loc="lower right",prop={'size':6})
    outfilepath = os.path.join(outdir,setname+'_ROC_ep'+str(epoch)+'.pdf')
    plt.savefig(outfilepath)
    plt.close()

def plot_pr_curve_multi_class(label_meanings, y_test, y_score, 
                              outdir, setname, epoch):
    """<label_meanings>: list of strings, one for each label
    <y_test>: matrix of ground truth
    <y_score>: matrix of predicted probabilities
    <outdir>: directory to save output file
    <setname>: string e.g. 'train' 'valid' or 'test'
    <epoch>: int for epoch"""
    #Modified from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    #https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
    n_classes = len(label_meanings)
    lw = 2
    
    #make order df.
    order = pd.DataFrame(np.zeros((n_classes,1)), index = [x for x in range(n_classes)],
                         columns = ['prc'])
    for i in range(n_classes):
        order.at[i,'prc'] = sklearn.metrics.average_precision_score(y_test[:,i], y_score[:,i])
    order = order.sort_values(by='prc',ascending=False)
    
    #Plot
    plt.figure()
    colors_list = ['palevioletred','darkorange','yellowgreen','olive','deepskyblue','royalblue','navy']
    curves_plotted = 0
    for i in order.index.values.tolist()[0:10]: #only plot the top ten so the plot is readable
        color_idx = curves_plotted%len(colors_list) #cycle through the colors list in order of colors
        color = colors_list[color_idx]
        average_precision = sklearn.metrics.average_precision_score(y_test[:,i], y_score[:,i])
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test[:,i], y_score[:,i])
        plt.step(recall, precision, color=color, where='post',
                 label='{:5s} (area {:0.2f})'.format(label_meanings[i], average_precision))
        curves_plotted+=1
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(setname.lower().capitalize()+' PRC Epoch '+str(epoch))
    plt.legend(loc="lower right",prop={'size':6})
    outfilepath = os.path.join(outdir,setname+'_PR_ep'+str(epoch)+'.pdf')
    plt.savefig(outfilepath)
    plt.close()

def plot_pr_curve_single_class(true_labels, pred_probs, outfilepath):
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    average_precision = sklearn.metrics.average_precision_score(true_labels, pred_probs)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(true_labels, pred_probs)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    plt.savefig(outfilepath)
    plt.close()

def plot_roc_curve_single_class(fpr, tpr, epoch, outfilepath):
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(outputfilepath)
    plt.close()

def plot_learning_curves(train_loss, valid_loss, results_dir, descriptor):
    """Variables
    <train_loss> and <valid_loss> are numpy arrays with one numerical entry
    for each epoch quanitfying the loss for that epoch."""
    x = np.arange(0,len(train_loss))
    plt.plot(x, train_loss, color='blue', lw=2, label='train')
    plt.plot(x, valid_loss, color='green',lw = 2, label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_dir, descriptor+'_Learning_Curves.png'))
    plt.close()
    #save numpy arrays of the losses
    np.save(os.path.join(results_dir,'train_loss.npy'),train_loss)
    np.save(os.path.join(results_dir,'valid_loss.npy'),valid_loss)

def plot_heatmap(outprefix, numeric_array, center, xticklabels, yticklabels):
    """Save a heatmap based on numeric_array"""
    seaborn.set(font_scale=0.6)
    seaplt = (seaborn.heatmap(numeric_array,
                           center=center,
                           xticklabels=xticklabels,
                           yticklabels=yticklabels)).get_figure()
    seaplt.savefig(outprefix+'.png')
    seaplt.clf()
