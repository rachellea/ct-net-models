#main.py

import timeit

from run_experiment import DukeCTModel
from models import custom_models_ctnet, custom_models_alternative, custom_models_ablation
from load_dataset import custom_datasets

#Note that here NUM_EPOCHS is set to 2 for the purposes of quickly demonstrating
#the code on the fake data. In all of the experiments reported in the paper,
#NUM_EPOCHS was set to 100. No model actually trained all the way to 100 epochs
#due to use of early stopping.
NUM_EPOCHS = 2

if __name__=='__main__':
    ####################################
    # CTNet-83 Model on Whole Data Set #----------------------------------------
    ####################################
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor = 'CTNet83',
                custom_net = custom_models_ctnet.CTNetModel,
                custom_net_args = {'n_outputs':83},
                loss = 'bce', loss_args = {},
                num_epochs=NUM_EPOCHS, patience = 15,
                batch_size = 2, device = 'all', data_parallel = True,
                use_test_set = False, task = 'train_eval',
                old_params_dir = '',
                dataset_class = custom_datasets.CTDataset_2019_10,
                dataset_args = {'label_type_ld':'disease_new',
                                    'label_meanings':'all',
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'train':'','valid':''}})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
    
    ###################################
    # CTNet-9 Model on Whole Data Set #-----------------------------------------
    ###################################
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor = 'CTNet9',
                custom_net = custom_models_ctnet.CTNetModel,
                custom_net_args = {'n_outputs':9},
                loss = 'bce', loss_args = {},
                num_epochs=NUM_EPOCHS, patience = 15,
                batch_size = 2, device = 'all', data_parallel = True,
                use_test_set = False, task = 'train_eval',
                old_params_dir = '',
                dataset_class = custom_datasets.CTDataset_2019_10,
                dataset_args = {'label_type_ld':'disease_new',
                                    'label_meanings':['nodule','opacity','atelectasis','pleural_effusion','consolidation','mass','pericardial_effusion','cardiomegaly','pneumothorax'],
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'train':'','valid':''}})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
    
    ####################################################
    # CTNet-83 Model on 2000 Train and 1000 Val Subset #------------------------
    ####################################################
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor = 'CTNet83_SmallData',
                custom_net = custom_models_ctnet.CTNetModel,
                custom_net_args = {'n_outputs':83},
                loss = 'bce', loss_args = {},
                num_epochs=NUM_EPOCHS, patience = 15,
                batch_size = 2, device = 'all', data_parallel = True,
                use_test_set = False, task = 'train_eval',
                old_params_dir = '',
                dataset_class = custom_datasets.CTDataset_2019_10,
                dataset_args = {'label_type_ld':'disease_new',
                                    'label_meanings':'all',
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'train':'/load_dataset/fakedata/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                        'valid':'/load_dataset/fakedata/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
    
    ######################################################################
    # Alternative Arch: BodyConv Model on 2000 Train and 1000 Val Subset #------
    ######################################################################
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor = 'BodyConv_SmallData',
                custom_net = custom_models_alternative.BodyConv,
                custom_net_args = {'n_outputs':83},
                loss = 'bce', loss_args = {},
                num_epochs=NUM_EPOCHS, patience = 15,
                batch_size = 2, device = 'all', data_parallel = True,
                use_test_set = False, task = 'train_eval',
                old_params_dir = '',
                dataset_class = custom_datasets.CTDataset_2019_10,
                dataset_args = {'label_type_ld':'disease_new',
                                    'label_meanings':'all',
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'train':'/load_dataset/fakedata/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                        'valid':'/load_dataset/fakedata/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
    
    ####################################################################
    # Alternative Arch: 3DConv Model on 2000 Train and 1000 Val Subset #--------
    ####################################################################
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor = 'ThreeDConv_SmallData',
                custom_net = custom_models_alternative.ThreeDConv,
                custom_net_args = {'n_outputs':83},
                loss = 'bce', loss_args = {},
                num_epochs=NUM_EPOCHS, patience = 15,
                batch_size = 4, device = 'all', data_parallel = True,
                use_test_set = False, task = 'train_eval',
                old_params_dir = '',
                dataset_class = custom_datasets.CTDataset_2019_10,
                dataset_args = {'label_type_ld':'disease_new',
                                    'label_meanings':'all',
                                    'num_channels':1,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'train':'/load_dataset/fakedata/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                        'valid':'/load_dataset/fakedata/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
    
    ####################################################################
    # Ablation Study: CTNet-83 (Pool) on 2000Train and 1000 Val Subset #--------
    ####################################################################
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor = 'CTNet83AblatePool_SmallData',
                custom_net = custom_models_ablation.CTNetModel_Ablate_PoolInsteadOf3D,
                custom_net_args = {'n_outputs':83},
                loss = 'bce', loss_args = {},
                num_epochs=NUM_EPOCHS, patience = 15,
                batch_size = 2, device = 'all', data_parallel = True,
                use_test_set = False, task = 'train_eval',
                old_params_dir = '',
                dataset_class = custom_datasets.CTDataset_2019_10,
                dataset_args = {'label_type_ld':'disease_new',
                                    'label_meanings':'all',
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'train':'/load_dataset/fakedata/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                        'valid':'/load_dataset/fakedata/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
    
    #####################################################################
    # Ablation Study: CTNet-83 (Rand) on 2000 Train and 1000 Val Subset #-------
    #####################################################################
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor = 'CTNet83AblateRand_SmallData',
                custom_net = custom_models_ablation.CTNetModel_Ablate_RandomInitResNet,
                custom_net_args = {'n_outputs':83},
                loss = 'bce', loss_args = {},
                num_epochs=NUM_EPOCHS, patience = 15,
                batch_size = 2, device = 'all', data_parallel = True,
                use_test_set = False, task = 'train_eval',
                old_params_dir = '',
                dataset_class = custom_datasets.CTDataset_2019_10,
                dataset_args = {'label_type_ld':'disease_new',
                                    'label_meanings':'all',
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'train':'/load_dataset/fakedata/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                        'valid':'/load_dataset/fakedata/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
    
    ###################################################
    # CTNet-1 Model on 2000 Train and 1000 Val Subset #-------------------------
    ###################################################
    for abnormality in ['nodule', 'opacity', 'atelectasis', 'pleural_effusion',
                        'consolidation', 'mass', 'pericardial_effusion',
                        'cardiomegaly', 'pneumothorax']:
        print('\n\n\n\n********** Working on',abnormality,'**********')
        tot0 = timeit.default_timer()
        DukeCTModel(descriptor = 'CTNet-'+abnormality,
                    custom_net =  custom_models_ctnet.CTNetModel,
                    custom_net_args = {'n_outputs':1},
                    loss = 'bce', loss_args = {},
                    num_epochs=NUM_EPOCHS, patience = 15,
                    batch_size = 2, device = 'all', data_parallel = True,
                    use_test_set = False, task = 'train_eval',
                    old_params_dir = '',
                    dataset_class = custom_datasets.CTDataset_2019_10,
                    dataset_args = {'label_type_ld':'disease_new',
                                        'label_meanings':[abnormality], #can be 'all' or a list of strings
                                        'num_channels':3,
                                        'pixel_bounds':[-1000,200],
                                        'data_augment':True,
                                        'crop_type':'single',
                                        'selected_note_acc_files':{'train':'/load_dataset/fakedata/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                            'valid':'/load_dataset/fakedata/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}}
                    )
        tot1 = timeit.default_timer()
        print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
        