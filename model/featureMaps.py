#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:03:48 2021

@author: saad
"""
from keras.models import Model
import numpy as np
from scipy.stats import zscore
from model.data_handler import rolling_window
import math
import gc



def genStim_CB(n_samples,termporal_width,size_r,size_c):
    # size_r = 32
    # size_c = 53
    low = 0
    high = 120
    # n_samples  = 100000
    rgb = np.random.randint(0,2,size=(n_samples,size_r*size_c))
    # rgb = np.unique(rgb,axis=0)
    rgb = rgb.reshape(rgb.shape[0],size_r,size_c)
    
    rgb[rgb==0] = low
    rgb[rgb==1] = high
    
    rgb = zscore(rgb)
    stim_feat = rolling_window(rgb,termporal_width,time_axis=0) 
    del rgb
    return stim_feat



def get_featureMaps(stim,mdl,mdl_params,layer_name,strongestUnit,chunk_size):
    # layer_name = 'conv2d'
    if mdl.name[:6] == 'CNN_3D':
        stim_size = np.array(mdl.input.shape[2:4])
    else:    
        stim_size = np.array(mdl.input.shape[-2:])
    idx_extra = 3
    # strongestUnit = np.array([8,16])
    # strongestUnit = np.array([9,15])
    
    
    # n_frames = 100000
    # chunk_size = 10000
    chunks_idx = np.concatenate((np.arange(0,stim.shape[0],chunk_size),np.atleast_1d(stim.shape[0])),axis=0)
    chunks_idx = np.array([chunks_idx[:-1],chunks_idx[1:]])
    chunks_n = chunks_idx.shape[1]
           
                           
    layer_output = mdl.get_layer(layer_name).output
    nchan = layer_output.shape[1]
    filt_temporal_width = mdl_params['filt_temporal_width']
    filt_size = mdl_params['chan_s']
    print(str(chunk_size))
    
    rwa_all = np.empty((chunks_n,nchan,filt_temporal_width,filt_size+(2*idx_extra),filt_size+(2*idx_extra)))
    _ = gc.collect()

    for j in range(chunks_n):
        # data_feat = genStim_CB(chunk_size,filt_temporal_width,32,53)
        layer_output = mdl.get_layer(layer_name).output
        new_model = Model(mdl.input, outputs=layer_output)
        layer_output = new_model.predict(stim[chunks_idx[0,j]:chunks_idx[1,j]])
        _ = gc.collect()
        
        if mdl.name[:6] == 'CNN_3D':
            layer_output = layer_output[:,:,strongestUnit[0],strongestUnit[1],strongestUnit[2]]
            layer_size = np.array(new_model.output.shape[2:4])
            stim_size = np.array(new_model.input.shape[2:4])            
        else:
            layer_output = layer_output[:,:,strongestUnit[0],strongestUnit[1]]
            layer_size = np.array(new_model.output.shape[-2:])
            stim_size = np.array(new_model.input.shape[-2:])
        # layer_output = np.concatenate((layer_output,np.array(layer_output_temp[:,:,strongestUnit[0],strongestUnit[1]])),axis=0)
    
        
        inputOutput_boundary = (stim_size - layer_size)/2
        strongestUnit_stim = strongestUnit[:2] + inputOutput_boundary
        
        cut_side = 0.5*(filt_size-1)+idx_extra
        idx_cut_r = np.arange(strongestUnit_stim[0]-cut_side,strongestUnit_stim[0]+cut_side+1).astype('int32')
        idx_cut_c = np.arange(strongestUnit_stim[1]-cut_side,strongestUnit_stim[1]+cut_side+1).astype('int32')
      
        idx_cut_rmesh, idx_cut_cmesh = np.meshgrid(idx_cut_r, idx_cut_c, indexing='ij')
        if mdl.name[:6] == 'CNN_3D':
            data_feat_filtCut = stim[chunks_idx[0,j]:chunks_idx[1,j],0,idx_cut_rmesh,idx_cut_cmesh,:]   
            data_feat_filtCut = np.moveaxis(data_feat_filtCut,-1,1)
        else:
            data_feat_filtCut = stim[:,:,idx_cut_rmesh,idx_cut_cmesh]   
            
        # data_feat_filtCut = np.concatenate((data_feat_filtCut,data_feat_filtCut_temp),axis = 0)
        # del stim
        
    
    
        # filtStrength = np.empty((layer_output.shape[2],layer_output.shape[3]))
        for i in range(layer_output.shape[1]):
            idx_filt = i
            prog = 'Processing Chunk %d of %d, filt %d of %d' %(j+1,chunks_n,idx_filt+1,layer_output.shape[1])
            print(prog)
            
            rwa = layer_output[:,idx_filt]
            # del layer_output
            rwa = data_feat_filtCut*rwa[:,None,None,None]
            rwa_all[j,i,:,:,:] = np.mean(rwa,axis=0)   
            del rwa
            _ = gc.collect()
            
    rwa_mean = np.mean(rwa_all,axis=0)
    return rwa_mean



