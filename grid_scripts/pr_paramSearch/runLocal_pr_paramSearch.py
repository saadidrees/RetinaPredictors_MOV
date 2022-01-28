#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:38:04 2021

@author: saad
"""
from pr_paramSearch import run_pr_paramSearch

# expDate = 'retina3'
# samps_shift=0.0
# path_mdl = '/home/saad/data/analyses/data_kiersten/'+expDate+'/8ms/photopic-10000_preproc-added_norm-1_rfac-2/CNN_2D/U-0.00_T-120_C1-13-03_C2-26-02_C3-24-01_BN-1_MP-0_TR-01/'
# trainingDataset = '/home/saad/data/analyses/data_kiersten/retina2/8ms/datasets/photopic-10000_preproc-cones_norm-1_rfac-2.h5'
# testingDataset = '/home/saad/data/analyses/data_kiersten/retina2/8ms/datasets/retina2_dataset_train_val_test_scotopic.h5'
# path_excel = '/home/saad/data/analyses/data_kiersten/retina2/8ms/pr_paramSearch'
# path_perFiles = '/home/saad/data/analyses/data_kiersten/retina2/8ms/pr_paramSearch'
# mdl_name='CNN_2D'

expDate = 'retina1'
samps_shift = 4
lightLevel = 'scotopic'
pr_type = 'rods'
pr_mdl_name = 'clark'
path_mdl = '/home/saad/data/analyses/data_kiersten/'+expDate+'/8ms_clark/photopic-10000_mdl-clark_b-0.36_g-0.448_y-4.48_z-166_preproc-added_norm-1_rfac-2/CNN_2D/U-0.00_T-120_C1-13-03_C2-26-02_C3-24-01_BN-1_MP-0_TR-01/'
trainingDataset = '/home/saad/data/analyses/data_kiersten/'+expDate+'/8ms_clark/datasets/'+expDate+'_dataset_train_val_test_photopic-10000_mdl-clark_b-0.36_g-0.448_y-4.48_z-166_preproc-added_norm-1_rfac-2.h5'
testingDataset = '/home/saad/data/analyses/data_kiersten/'+expDate+'/8ms/datasets/'+expDate+'_dataset_train_val_test_'+lightLevel+'.h5'
mdl_name = 'CNN_2D'

path_excel = '/home/saad/data/analyses/data_kiersten/'+expDate+'/8ms_clark/pr_diff'
path_perFiles = '/home/saad/data/analyses/data_kiersten/'+expDate+'/8ms_clark/pr_diff'


# r_sigma = 8.5
# r_phi = 11.5
# r_eta = 4.5
# r_k=0.01
# r_h=3
# r_beta=10.0
# r_hillcoef=4
# r_gamma=800.0

c_beta = 0.36
c_gamma = 0.448
c_tau_y = 22#4.48
c_n_y = 4.33
c_tau_z = 166
c_n_z = 1

# run_pr_paramSearch(expDate,path_mdl,trainingDataset,testingDataset,path_excel,path_perFiles,lightLevel,pr_type,samps_shift=samps_shift,r_sigma=r_sigma,r_phi=r_phi,r_eta=r_eta,r_k=r_k,r_h=r_h,r_beta=r_beta,r_hillcoef=r_hillcoef,r_gamma=r_gamma,mdl_name=mdl_name,num_cores=28)
run_pr_paramSearch(expDate,path_mdl,trainingDataset,testingDataset,path_excel,path_perFiles,lightLevel,pr_type,pr_mdl_name,samps_shift=samps_shift,c_beta=c_beta,c_gamma=c_gamma,c_tau_y=c_tau_y,c_n_y=c_n_y,c_tau_z=c_tau_z,c_n_z=c_n_z,mdl_name=mdl_name,num_cores=28)