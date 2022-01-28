#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 22:38:14 2021

@author: saad
"""
import numpy as np
import csv
import os
APPEND_TO_EXISTING = 0
expDate = 'retina1'
subfold = '8ms'
datasetName = 'photopic'    # ['scotopic','photopic','scotopic_photopic','photopic_shiftedPhotopic']
path_model_save_base = os.path.join('/home/sidrees/scratch/RetinaPredictors/data',expDate,subfold,datasetName)
path_dataset = os.path.join('/home/sidrees/scratch/RetinaPredictors/data',expDate,subfold,'datasets')
name_datasetFile = expDate+'_dataset_train_val_test_'+datasetName+'.h5'       # _dataset_train_val_test_scotopic_photopic
fname_data_train_val_test = os.path.join(path_dataset,name_datasetFile)
path_existing_mdl = 0

mdl_name = 'CNN_2D'
thresh_rr=0
temporal_width=120
pr_temporal_width=0
bz_ms=2000
nb_epochs=300
TRSAMPS = 0

USE_CHUNKER=0
BatchNorm=0
MaxPool=0
num_trials=1

chan1_n = np.atleast_1d((20)) #np.array((8,9,10,11,12,13,14,15,16,18,20)) #np.atleast_1d((18)) #np.atleast_1d((18))
filt1_size = np.atleast_1d((3)) #((1,2,3,4,5,6,7,8,9))
filt1_3rdDim = np.atleast_1d((0)) #np.atleast_1d((1,10,20,30,40,50,60))

chan2_n = np.atleast_1d((24)) #np.array((0,15,18,20,22,24,25,26,28,30)) #np.atleast_1d((25))     #np.array((8,10,13,15,18,20,22,24,25,26,28,30))
filt2_size = np.atleast_1d((2))   #((1,2,3,4,5,6,7,8,9))
filt2_3rdDim = np.atleast_1d((0)) #np.atleast_1d((1,8,10,12,14,18,20,30,40,50))

chan3_n = np.atleast_1d((22)) #np.array((0,15,18,20,22,24,25,26,28,30)) #np.atleast_1d((18))     #np.array((13,15,18,20,22,24,25,26,28,30))
filt3_size = np.atleast_1d((1))   # ((1,2,3,4,5,6,7,8,9))
filt3_3rdDim = np.atleast_1d((0))#np.atleast_1d((1,8,10,12,14,18,20,30,40,50))


image_dim = 10
image_tempDim = temporal_width


csv_header = ['expDate','mdl_name','path_model_save_base','name_datasetFile','path_existing_mdl','thresh_rr','temp_width','pr_temporal_width','bz_ms','nc_epochs','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','BatchNorm','MaxPool','num_trials','USE_CHUNKER','TRSAMPS']
params_array = np.zeros((100000,3*3))
counter = -1
for cc1 in chan1_n:
    for cc2 in chan2_n:
        for cc3 in chan3_n:
            for ff1 in filt1_size:
                for ff2 in filt2_size:
                    for ff3 in filt3_size:
                        for dd1 in filt1_3rdDim:
                            for dd2 in filt2_3rdDim:
                                for dd3 in filt3_3rdDim:
                        
                                    
                                    c1 = cc1
                                    c2 = cc2
                                    c3 = cc3
                                    
                                    f1 = ff1
                                    f2 = ff2
                                    f3 = ff3
                                    
                                    d1 = dd1
                                    d2 = dd2
                                    d3 = dd3
                                    
                                    
                                    l1_out = image_dim - f1 + 1
                                    l2_out = l1_out - f2 + 1
                                    l3_out = l2_out - f3 + 1
                                    
                                    d1_out = image_tempDim - d1 + 1
                                    d2_out = d1_out - d2 + 1
                                    d3_out = d2_out - d3 + 1
                                    
                                    if l2_out < 1:
                                        c2 = 0
                                        f2 = 0
                                        c3 = 0
                                        f3 = 0
                                        
                                    if l3_out < 1:
                                        c3 = 0
                                        f3 = 0
                                        
                                    if d2_out < 1:
                                        c2 = 0
                                        f2 = 0
                                        d2 = 0
                                        
                                        c3 = 0
                                        f3 = 0
                                        d3 = 0
                                        
                                    if d3_out < 1:
                                       c3 = 0
                                       f3 = 0
                                       d3 = 0
                                       
                                    if c2==0:
                                        f2 = 0
                                        d2 = 0
                                        c3 = 0
                                        f3 = 0
                                        d3 = 0
                                        
                                    if (mdl_name=='CNN_3D' or mdl_name[-5:]=='CNN3D') and d2==0:
                                        c2 = 0
                                        f2 = 0
                                        d3 = 0
                                        f3 = 0
                                        d3 = 0
                                        
                                    if (mdl_name=='CNN_3D' or mdl_name[-5:]=='CNN3D') and d3==0:
                                        c3 = 0
                                        f3 = 0
                                        
                                        
                                    if c2>0 and f2==0:
                                        raise ValueError('Post f2 is 0')
                                    
                                    if (mdl_name=='CNN_3D' or mdl_name[-5:]=='CNN3D'):
                                        conds = np.atleast_1d((d3_out > 1,np.logical_and(c3==0,d2_out>1),np.logical_and(d2_out<1,d1_out>1)))
                                    
                                        if np.any(conds)!=True:   # we want temporal dimension to be flattedned in the last layer
                                            counter +=1
                                            params_array[counter] = [c1, f1, d1, c2, f2, d2, c3, f3, d3]
                                        
                                    elif mdl_name=='CNN_2D':
                                        counter +=1
                                        params_array[counter] = [c1, f1, d1, c2, f2, d2, c3, f3, d3]
                                    
                                        
                                        
                        
params_array = params_array[:counter+1]
params_array = np.unique(params_array,axis=0)

# %%
fname_csv_file = 'model_params.csv'
if APPEND_TO_EXISTING == 0:
    if os.path.exists(fname_csv_file):
        raise ValueError('Paramter file already exists')
        
else:
    write_mode='a'

fname_model = ([])
for i in range(params_array.shape[0]):
                        
    rgb = params_array[i,:].astype('int').tolist()
    csv_data = [expDate,mdl_name,path_model_save_base,fname_data_train_val_test,path_existing_mdl,thresh_rr,temporal_width,pr_temporal_width,bz_ms,nb_epochs]
    csv_data.extend(rgb)
    csv_data.extend([BatchNorm,MaxPool,num_trials,USE_CHUNKER,TRSAMPS])
               
    # fname_model.append('U-%0.2f_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(csv_data[3],csv_data[4],csv_data[7],csv_data[8],
    #                                                                                                  csv_data[10],csv_data[11], 
    #                                                                                                  csv_data[13],csv_data[14], 
    #                                                                                                  csv_data[16],csv_data[17],csv_data[18]))
    
    if not os.path.exists(fname_csv_file):
        with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_header) 
            
    with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(csv_data) 
    
















