#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:42:54 2021

@author: saad
"""


import numpy as np
import matplotlib.pyplot as plt
import os

from run_models import run_model

lr = 1e-3 #0.01/5

expDate = '20180502_s3'
subFold = '' #'8ms_clark'
dataset_subFold = 'lr-%0.4f' %lr
mdl_name = 'PR_CNN2D' #'PR_CNN2D_fixed' #'PR_CNN2D'#'CNN_2D'
dataset = 'SACC_mesopic-2026'
path_existing_mdl = ''
info = ''
idx_CNN_start=1
CONTINUE_TRAINING=0

USE_CHUNKER=1
pr_temporal_width = 130
temporal_width=70
thresh_rr=0
chan1_n=15#20#13
filt1_size=11
filt1_3rdDim=0
chan2_n=25#24#26
filt2_size=7
filt2_3rdDim=0
chan3_n=0 #25#22#24
filt3_size=0 #3
filt3_3rdDim=0
trainingSamps_dur = 5 # minutesfn
validationSamps_dur=0.5   # minutes
nb_epochs=10
bz_ms=3000#20000 #10000
BatchNorm=1
MaxPool=1
saveToCSV=1
runOnCluster=0
num_trials=1

BatchNorm_train = 0

name_datasetFile = expDate+'_dataset_train_val_test_'+dataset+'.h5'
path_model_save_base = os.path.join('/home/saad/data/analyses/data_saad/',expDate,subFold,dataset,dataset_subFold)
path_dataset_base = os.path.join('/home/saad/data/analyses/data_saad/',expDate,subFold)
# path_dataset_base = os.path.join('/home/saad/postdoc_db/analyses/data_saad/',expDate,subFold)
fname_data_train_val_test = os.path.join(path_dataset_base,'datasets',name_datasetFile)

c_trial = 1

# %%
for c_trial in range(1,num_trials+1):
    model_performance,mdl = run_model(expDate,mdl_name,path_model_save_base,fname_data_train_val_test,path_existing_mdl = path_existing_mdl,path_dataset_base=path_dataset_base,saveToCSV=saveToCSV,runOnCluster=0,
                            temporal_width=temporal_width, pr_temporal_width=pr_temporal_width, thresh_rr=thresh_rr,
                            chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim,
                            chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim,
                            chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim,
                            nb_epochs=nb_epochs,bz_ms=bz_ms,trainingSamps_dur=trainingSamps_dur,
                            BatchNorm=BatchNorm,BatchNorm_train = BatchNorm_train,MaxPool=MaxPool,c_trial=c_trial,
                            lr=lr, USE_CHUNKER=USE_CHUNKER,idx_CNN_start=idx_CNN_start,CONTINUE_TRAINING=CONTINUE_TRAINING,info=info)
    
plt.plot(model_performance['fev_medianUnits_allEpochs'])

# %% for reading from params array
path_model_save_base = os.path.join('/home/saad/data/analyses/data_kiersten',expDate)
path_dataset_base = os.path.join('/home/saad/data/Dropbox/postdoc/analyses/data_kiersten')

params_array = params_array.astype(int)
select_range = np.arange(237+5,params_array.shape[0])
for i in select_range:
    for c_trial in range(1,num_trials+1):
        model_performance = run_model(expDate,mdl_name,path_model_save_base,path_dataset_base=path_dataset_base,saveToCSV=saveToCSV,runOnCluster=0,
                            temporal_width=temporal_width, thresh_rr=thresh_rr,
                            chan1_n=params_array[i,0], filt1_size=params_array[i,1], filt1_3rdDim=params_array[i,2],
                            chan2_n=params_array[i,3], filt2_size=params_array[i,4], filt2_3rdDim=params_array[i,5],
                            chan3_n=params_array[i,6], filt3_size=params_array[i,7], filt3_3rdDim=params_array[i,8],
                            nb_epochs=nb_epochs,bz_ms=bz_ms,BatchNorm=BatchNorm,MaxPool=MaxPool,c_trial=c_trial)
