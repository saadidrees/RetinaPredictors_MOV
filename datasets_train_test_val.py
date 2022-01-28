#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:34:08 2021

@author: saad
"""

# Save training, testing and validation datasets to be read by jobs on cluster

import os
import h5py
import numpy as np
from model.data_handler import load_data, load_data_kr_allLightLevels, save_h5Dataset, check_trainVal_contamination
from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])

whos_data = 'saad'
lightLevel = 'allLightLevels'     # ['scotopic', 'photopic','scotopic_photopic']
datasetsToLoad = 'CB_sig-4_tf-8' #['scotopic','photopic','scotopic_photopic']
convertToRstar = True

def rgbToRstar(meanIntensity,zeroIntensity,data):

    X = data.X
    rgb = X[0]
    pix_rgbVals = np.unique(rgb)
    pix_RstarVals = (pix_rgbVals/60)*meanIntensity
    for i in range(pix_rgbVals.shape[0]):
        X[X==pix_rgbVals[i]] = pix_RstarVals[i]
        
    X[X==0] = zeroIntensity
        
    X = X * 1e-3 * t_frame  # photons per time bin 

        
    data = Exptdata(X,data.y)
    return data

def rgbToNormalize(data):
    X = data.X
    X = np.round(X - X.mean())
    X = X/X.max()       
        
        
expDate = '20180502_s3'     # ('20180502_s3', '20180919_s3','20181211a_s3', '20181211b_s3')
d = 'mesopic'

path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_saad/',expDate,'datasets')
fname_dataFile = os.path.join(path_dataset,(expDate+'_dataset_'+datasetsToLoad+'.h5'))

    
t_frame = 8
filt_temporal_width = 0
idx_cells = None
thresh_rr = 0

frac_val = 0.1
frac_test = 0
 
data_train,data_val,data_test,data_quality,dataset_rr = load_data(fname_dataFile,frac_val=frac_val,frac_test=frac_test,filt_temporal_width=filt_temporal_width,idx_cells=idx_cells,thresh_rr=thresh_rr)

f = h5py.File(fname_dataFile,'r')
t_frame = np.array(f['data']['stim_0']['stim_frames'].attrs['t_frame'])
pix_rgb_low = np.array(f['data']['stim_0']['stim_frames'].attrs['pix_rgb_low'])
pix_rgb_high = np.array(f['data']['stim_0']['stim_frames'].attrs['pix_rgb_high'])
pix_rgbVals = (pix_rgb_low,pix_rgb_high)
Rstar_60 = np.array(f['data']['stim_0']['stim_frames'].attrs['Rstar_60'].astype('int64'))
Rstar_0 = np.array(f['data']['stim_0']['stim_frames'].attrs['Rstar_0'])
f.close()

Rstar_0_intensities = {
    'mesopic': Rstar_0,
    }

meanIntensities = {
    'mesopic': Rstar_60,  # 1012
}

# dataset_rr = None
if convertToRstar == False:
    fname_data_train_val_test = os.path.join(path_dataset,(expDate+'_dataset_train_val_test_'+datasetsToLoad+'_'+d+'.h5'))
    # rgbToNormalize(data)
    data_train = rgbToNormalize(data_train)
    data_val = rgbToNormalize(data_val)
    data_test = rgbToNormalize(data_test)


else:
    meanIntensity = meanIntensities[d]
    zeroIntensity = Rstar_0_intensities[d]
    
    fname_data_train_val_test = os.path.join(path_dataset,(expDate+'_dataset_train_val_test_'+datasetsToLoad+'_'+d+'-'+str(meanIntensity)+'.h5'))
     
    data_train = rgbToRstar(meanIntensity,zeroIntensity,data_train)
    data_val = rgbToRstar(meanIntensity,zeroIntensity,data_val)
    data_test = rgbToRstar(meanIntensity,zeroIntensity,data_test)


samps_shift = 0
parameters = {
't_frame': t_frame,
'filt_temporal_width': filt_temporal_width,
'frac_val': frac_val,
'frac_test':frac_test,
'thresh_rr': thresh_rr,
'samps_shift': samps_shift
}

# fname_data_train_val_test = os.path.join(path_dataset,(expDate+'_dataset_train_val_test_'+datasetsToLoad+'.h5'))
save_h5Dataset(fname_data_train_val_test,data_train,data_val,data_test,data_quality,dataset_rr,parameters)
