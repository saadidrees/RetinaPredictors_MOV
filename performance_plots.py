#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:01:49 2021

@author: saad
"""
import h5py
import numpy as np
import os
import re
 
from global_scripts import utils_si
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.colors import NoNorm


from scipy.stats import wilcoxon
import gc
import csv
# import pylustrator
from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])

from model.load_savedModel import load
from model.data_handler import load_data, load_h5Dataset, prepare_data_cnn2d, prepare_data_cnn3d, prepare_data_convLSTM, prepare_data_pr_cnn2d
from model.performance import getModelParams, model_evaluate,model_evaluate_new,paramsToName, get_weightsDict
from model import metrics
from model import featureMaps
from model.train_model import chunker
# from pyret.filtertools import sta, decompose
 
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Input

from numba import cuda
gpu_cuda = cuda.get_current_device()

expDates = ('20180502_s3',)    # ('20180502_s3', '20180919_s3','20181211a_s3', '20181211b_s3'
subFold = '' #'8ms_clark' #'8ms_trainablePR' # test_coneParams
mdl_subFold = 'old/lr-0.0020'#'50epochs'
lightLevel_1 = 'SACC_mesopic-2026'#_mesopic-2026' #
models_all = ('PRFR_CNN2D',) # (PR_CNN2D, 'CNN_2D','CNN_3D','CNN_3D_LSTM','convLSTM')  

writeToCSV = False

path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors/performance'

models_list = []

param_list_keys = ['U', 'T','C1_n','C1_s','C1_3d','C2_n','C2_s','C2_3d','C3_n','C3_s','C3_3d','BN','MP','TR','P']   # model parameters to group by
csv_header = ['mdl_name','expDate','temp_window','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','FEV_median','FracExVar','corr_median','rr_corr_median']

cols_lightLevels = {
    'photopic': 'r',
    'scotopic': 'b',
    }
 
    
perf_allExps = {}   # Performance dict for all experiments, models, conditions
params_allExps = {} # Dict containing parameter values for all experiments, models, conditions
paramNames_allExps = {} #Dict containing the folder name of the model, which is basically concatenating the params together

for idx_exp in range(len(expDates)):
    
    path_mdl_base = os.path.join('/home/saad/data/analyses/data_saad/',expDates[idx_exp],subFold)
    path_dataset_base = path_mdl_base

    perf_allModels = {}
    params_allModels = {}
    paramNames_allModels = {}
    exp_select = expDates[idx_exp]
    
    path_mdl_drive = os.path.join(path_mdl_base,lightLevel_1,mdl_subFold)
    path_dataset = os.path.join(path_dataset_base,'datasets')
    
    # mdl_select = models_all[0]
    for mdl_select in models_all:

        if writeToCSV==True:
            fname_csv_file = 'performance_'+exp_select+'_'+lightLevel_1+'_avgAcrossTrials_'+mdl_select+'.csv'
            fname_csv_file = os.path.join(path_save_performance,fname_csv_file)
            with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header) 

        # Get the folder names (paramNames) for the different runs of mdl_select
        paramNames_temp = os.listdir(os.path.join(path_mdl_drive,mdl_select))
        paramNames = ([])
        for p in paramNames_temp:
            try:     # some model runs are incomplete so don't consider them. They will not contain the performance file
                if os.listdir(os.path.join(path_mdl_drive,mdl_select,p,'performance')):
                    paramNames.append(p)
            except:
                pass
        
        # Cut out the trial number to group different trials of same models/params together
        paramNames_cut = []
        for i in range(0,len(paramNames)):
            rgb = re.compile(r'_TR-(\d+)')
            idx_paramMention = rgb.search(paramNames[i]).span()
            paramNames_cut_rgb = paramNames[i][:idx_paramMention[0]] + paramNames[i][idx_paramMention[1]:]
            paramNames_cut.append(paramNames_cut_rgb)
        paramNames_unique = list(set(paramNames_cut))   # This contains the unique runs
        
        # Locate where the unique names lie in the non-unique array. So this will also tell how many trials for each model/param
        idx_paramName = ([])
        for i in range(len(paramNames_unique)):
            name = paramNames_unique[i]
            rgb = [j for j,k in enumerate(paramNames_cut) if name==k]
            idx_paramName.append(rgb)
            
    
        perf_paramNames = {}
        param_list = dict([(key, []) for key in param_list_keys])


        # param_unique_idx = 0
        for param_unique_idx in range(len(paramNames_unique)):
            
            # initialize variables where the last dimension size is number of trials for that model/params combo
            num_trials = len(idx_paramName[param_unique_idx])
            
            fev_allUnits_allEpochs_allTr = np.zeros((1000,1000,num_trials))           
            fev_allUnits_bestEpoch_allTr = np.zeros((1000,num_trials))
            fev_medianUnits_allEpochs_allTr = np.zeros((1000,num_trials))
            fev_medianUnits_bestEpoch_allTr = np.zeros((num_trials))

            predCorr_allUnits_allEpochs_allTr = np.zeros((1000,1000,num_trials))           
            predCorr_allUnits_bestEpoch_allTr = np.zeros((1000,num_trials))
            predCorr_medianUnits_allEpochs_allTr = np.zeros((1000,num_trials))
            predCorr_medianUnits_bestEpoch_allTr = np.zeros((num_trials))
            
            trial_id = np.zeros((num_trials))
            
            
            for c_tr in range(num_trials):
            
                paramFileName = paramNames[idx_paramName[param_unique_idx][c_tr]]
                paramName_path = os.path.join(path_mdl_drive,mdl_select,paramFileName)
                fname_performanceFile = os.path.join(paramName_path,'performance',exp_select+'_'+paramFileName+'.h5')
        
                f = h5py.File(fname_performanceFile,'r')
                
                rgb = np.atleast_1d(f['model_performance']['fev_allUnits_allEpochs'])
                num_units = rgb.shape[1]
                num_epochs = rgb.shape[0]
                fev_allUnits_allEpochs_allTr[:num_epochs,:num_units,c_tr] = rgb            

                rgb = np.array(f['model_performance']['fev_allUnits_bestEpoch'])
                fev_allUnits_bestEpoch_allTr[:rgb.shape[0],c_tr] = rgb
                try:
                    rgb = np.array(f['model_performance']['fev_median_allEpochs'])
                except:
                    rgb = np.array(f['model_performance']['fev_medianUnits_allEpochs'])
                rgb = np.array(f['model_performance']['fev_medianUnits_allEpochs'])
                fev_medianUnits_allEpochs_allTr[:rgb.shape[0],c_tr] = rgb
                rgb = np.atleast_1d(f['model_performance']['fev_medianUnits_bestEpoch'])
                fev_medianUnits_bestEpoch_allTr[c_tr] = rgb
                
                rgb = np.array(f['model_performance']['predCorr_allUnits_bestEpoch'])
                predCorr_allUnits_bestEpoch_allTr[:rgb.shape[0],c_tr] = rgb
                rgb = np.array(f['model_performance']['predCorr_medianUnits_allEpochs'])
                predCorr_medianUnits_allEpochs_allTr[:rgb.shape[0],c_tr] = rgb
                rgb = np.atleast_1d(f['model_performance']['predCorr_medianUnits_bestEpoch'])
                predCorr_medianUnits_bestEpoch_allTr[c_tr] = rgb
                
                rgb = getModelParams(os.path.split(fname_performanceFile)[-1])
                trial_id[c_tr] = np.atleast_1d(rgb['TR'])
                
            
            
            fev_allUnits_allEpochs_allTr = fev_allUnits_allEpochs_allTr[:num_epochs,:num_units,:]
            fev_allUnits_bestEpoch_allTr = fev_allUnits_bestEpoch_allTr[:num_units,:]
            fev_medianUnits_allEpochs_allTr = fev_medianUnits_allEpochs_allTr[:num_epochs,:]
            fev_medianUnits_bestEpoch_allTr = fev_medianUnits_bestEpoch_allTr
            fracExVar_allUnits = np.array(f['model_performance']['fracExVar_allUnits'])
            fracExVar_medianUnits = np.array(f['model_performance']['fracExVar_medianUnits'])
            
            
            predCorr_allUnits_allEpochs_allTr = fev_allUnits_allEpochs_allTr[:num_epochs,:num_units,:]
            predCorr_allUnits_bestEpoch_allTr = predCorr_allUnits_bestEpoch_allTr[:num_units,:]
            predCorr_medianUnits_allEpochs_allTr = fev_medianUnits_allEpochs_allTr[:num_epochs,:]
            predCorr_medianUnits_bestEpoch_allTr = predCorr_medianUnits_bestEpoch_allTr
            rrCorr_allUnits =  np.array(f['model_performance']['rrCorr_allUnits'])
            rrCorr_medianUnits = np.array(f['model_performance']['rrCorr_medianUnits'])

            
            perf = {
                'fev_allUnits_allEpochs_allTr': fev_allUnits_allEpochs_allTr,
                'fev_allUnits_bestEpoch_allTr': fev_allUnits_bestEpoch_allTr,
                'fev_medianUnits_allEpochs_allTr': fev_medianUnits_allEpochs_allTr,
                'fev_medianUnits_bestEpoch_allTr': fev_medianUnits_bestEpoch_allTr,
                'fracExVar_allUnits':  fracExVar_allUnits,
                'fracExVar_medianUnits':  fracExVar_medianUnits,    
                'idx_bestEpoch': np.array(f['model_performance']['idx_bestEpoch']),
                
                
                'predCorr_allUnits_allEpochs_allTr': predCorr_allUnits_allEpochs_allTr,
                'predCorr_allUnits_bestEpoch_allTr': predCorr_allUnits_bestEpoch_allTr,
                'predCorr_medianUnits_allEpochs_allTr': predCorr_medianUnits_allEpochs_allTr,
                'predCorr_medianUnits_bestEpoch_allTr': predCorr_medianUnits_bestEpoch_allTr,
                'rrCorr_allUnits':  rrCorr_allUnits,
                'rrCorr_medianUnits':  rrCorr_medianUnits,                
                'num_trials': num_trials,
                # 'val_dataset_name': utils_si.h5_tostring(np.array(f['model_performance']['val_dataset_name'])),
                'trial_id': trial_id,
                }
            
            performance = {}
            performance['model_performance'] = perf

            select_groups = ('model_params','data_quality')
            

            for j in select_groups:
                perf_group = {}
                
                keys = list(f[j].keys())
                
                for i in keys:
                    rgb = np.array(f[j][i])
                    rgb_type = rgb.dtype.name
                       
                    if 'bytes' in rgb_type:
                        perf_group[i] = utils_si.h5_tostring(rgb)
                    else:
                        perf_group[i] = rgb
                        
                performance[j] = perf_group
                    
            filt_temporal_width = np.array(f['stim_info']['temporal_width'])
            performance['model_params']['thresh_rr'] = np.array(f['thresh_rr'])
            performance['model_params']['idx_unitsToTake'] = np.array(f['idx_unitsToTake'])
            performance['model_params']['temporal_width'] = filt_temporal_width
            
            pr_param_list = ('pr_alpha','pr_beta','pr_gamma','pr_nY','pr_nZ','pr_tauY','pr_tauZ')
            perf_keys = list(f['model_performance'].keys())
            if 'pr_alpha' in perf_keys:
                for j in pr_param_list:
                    performance['model_params'][j] = np.array(f['model_performance'][j])
            
            
            rgb = getModelParams(paramFileName)
            for i in list(rgb.keys()):
                param_list[i].append(rgb[i])
            
            
            dataset_pred = {
                'obs_rate': np.array(f['dataset_pred']['obs_rate']),
                'pred_rate': np.array(f['dataset_pred']['pred_rate']),
                }
            performance['dataset_pred'] = dataset_pred
            # performance['resp_median_trainingData_allUnits'] = resp_median_allUnits
            perf_paramNames[paramNames_unique[param_unique_idx]] = performance
            
            
            
            if writeToCSV == True:
                csv_data = [mdl_select,exp_select,filt_temporal_width,rgb['C1_n'], rgb['C1_s'], rgb['C1_3d'], rgb['C2_n'], rgb['C2_s'], rgb['C2_3d'], rgb['C3_n'], rgb['C3_s'], rgb['C3_3d'],np.nanmean(fev_medianUnits_bestEpoch_allTr),fracExVar_medianUnits,np.mean(predCorr_medianUnits_bestEpoch_allTr),rrCorr_medianUnits]
                with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
                        csvwriter = csv.writer(csvfile) 
                        csvwriter.writerow(csv_data) 
                        
                        
            f.close()

            
            # p_U
            
        perf_allModels[mdl_select] = perf_paramNames
        params_allModels[mdl_select] = param_list
        paramNames_allModels[mdl_select] = paramNames_unique

        
    perf_allExps[exp_select] = perf_allModels       
    params_allExps[exp_select] = params_allModels
    paramNames_allExps[exp_select] = paramNames_allModels

    
# del perf_allModels, perf_paramNames, performance, perf_group, param_list, params_allModels, paramNames_allModels, paramNames

# %% D1: Test model
select_exp = expDates[0]
select_mdl = models_all[0] #'CNN_2D' #'CNN_2D_chansVary'#'CNN_2D_filtsVary'
# params_mdl = params_allExps[select_exp][select_mdl]

val_dataset_1 = lightLevel_1  #lightLevel_1 #      # ['scotopic','photopic']
correctMedian = False

select_U = 0#0.15
select_P = 130 #180
select_T = 70 #70 #120
select_BN = 1
select_MP = 1
# select_TR = 1
select_C1_n = 15 #13 #20
select_C1_s = 11
select_C1_3d = 0#50#25
select_C2_n = 25 #26 #24
select_C2_s = 7#2#2
select_C2_3d = 0#10#5
select_C3_n = 0 #25 #24 #22
select_C3_s = 0 #3#1#1
select_C3_3d = 0#62#32

paramFileName = paramsToName(select_mdl,U=select_U,P=select_P,T=select_T,BN=select_BN,MP=select_MP,
                 C1_n=select_C1_n,C1_s=select_C1_s,C1_3d=select_C1_3d,
                 C2_n=select_C2_n,C2_s=select_C2_s,C2_3d=select_C2_3d,
                 C3_n=select_C3_n,C3_s=select_C3_s,C3_3d=select_C3_3d)

# val_dataset_1 = perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['val_dataset_name'][0]


# assert val_dataset_1 != val_dataset_1, 'same datasets selected'

idx_bestTrial = np.nanargmax(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
# idx_bestEpoch = perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial].shape[0]-1
idx_bestEpoch = np.nanargmax(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])
trial_num = perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['trial_id'][idx_bestTrial]
# plt.plot(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_allEpochs_allTr'][10:,idx_bestTrial])

select_TR = int(trial_num)

mdlFolder = paramFileName+'_TR-%02d' % select_TR
path_model = os.path.join(path_mdl_drive,select_mdl,mdlFolder)
mdl = load(os.path.join(path_model,mdlFolder))
fname_bestWeight = 'weights_'+mdlFolder+'_epoch-%03d' % (idx_bestEpoch+1)
try:
    mdl.load_weights(os.path.join(path_model,fname_bestWeight))
except:
    mdl.load_weights(os.path.join(path_model,fname_bestWeight+'.h5'))
weights_dict = get_weightsDict(mdl)


fname_data_train_val_test = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+val_dataset_1+'.h5'))
result = load_h5Dataset(fname_data_train_val_test,nsamps_val=1,nsamps_train=5,LOAD_TR=True,RETURN_VALINFO=True)   # give samples in minutes
data_train=result[0]; data_val = result[1]; data_quality = result[3]; parameters = result[5]; data_val_info = result[7]
# data_train,data_val,_,data_quality,_,parameters,_ = load_h5Dataset(fname_data_train_val_test,nsamps_val=20000,nsamps_train=20000,LOAD_TR=False)
# samps_shift = int(np.array(parameters['samps_shift']))
samps_shift = 0
# data_val=data_train


# resp_orig = resp_orig['train']

filt_temporal_width = select_T
# obs_rate_allStimTrials_d1 = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]


if select_mdl[:6]=='CNN_2D' or select_mdl == 'replaceDense_2D':
    data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
elif select_mdl[:6]=='CNN_3D':
    data_val = prepare_data_cnn3d(data_val,select_T,np.arange(data_val.y.shape[1]))
    
elif select_mdl[:6]=='convLS' or select_mdl == 'LSTM_CNN_2D':
    data_val = prepare_data_convLSTM(data_val,select_T,np.arange(data_val.y.shape[1]))
elif select_mdl[:8]=='PR_CNN2D' or select_mdl[:10]=='PRFR_CNN2D':
    pr_temporal_width = perf_allExps[select_exp][select_mdl][paramFileName]['model_params']['pr_temporal_width']
    data_val = prepare_data_pr_cnn2d(data_val,pr_temporal_width,np.arange(data_val.y.shape[1]))
    # obs_rate_allStimTrials_d1 = dataset_rr['stim_0']['val'][:,pr_temporal_width:,:]

elif select_mdl[:8]=='PR_CNN3D':
    pr_temporal_width = perf_allExps[select_exp][select_mdl][paramFileName]['model_params']['pr_temporal_width']
    data_val = prepare_data_cnn3d(data_val,pr_temporal_width,np.arange(data_val.y.shape[1]))
    obs_rate_allStimTrials_d1 = dattaset_rr['stim_0']['val'][:,pr_temporal_width:,:]

# plt.imshow(data_val.X[1000,0,:,:],cmap='gray',norm=NoNorm())
# plt.gca().invert_yaxis()
# plt.show()


batch_size = 2000
gen = chunker(data_val.X,batch_size,mode='predict')
pred_rate = mdl.predict(gen,steps=int(np.ceil(data_val.X.shape[0]/batch_size)))
# pred_rate = mdl.predict(data_val.X)
_ = gc.collect()
# gpu_cuda.reset()
obs_noise = data_quality['var_noise'] #data_quality['var_noise_dset_all'] #data_quality['var_noise']
obs_rate = data_val.y


num_iters = 1
fev_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d1_allUnits[:,i], fracExplainableVar[:,i], predCorr_d1_allUnits[:,i], rrCorr_d1_allUnits[:,i] = model_evaluate_new(obs_rate,pred_rate,0,RR_ONLY=False,lag = samps_shift,obs_noise=obs_noise)


fev_d1_allUnits = np.mean(fev_d1_allUnits,axis=1)
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d1_allUnits = np.mean(predCorr_d1_allUnits,axis=1)
rrCorr_d1_allUnits = np.mean(rrCorr_d1_allUnits,axis=1)

idx_allUnits = np.arange(fev_d1_allUnits.shape[0])
idx_d1_valid = idx_allUnits
idx_d1_valid = np.logical_and(fev_d1_allUnits>-1,fev_d1_allUnits<1.1)
idx_d1_valid = np.logical_and(idx_d1_valid,data_quality['fev_allUnits']>0)
idx_d1_valid = np.logical_and(idx_d1_valid,~np.isnan(predCorr_d1_allUnits))
idx_d1_valid = idx_allUnits[idx_d1_valid]

# fev_d1_validUnits = fev_d1_allUnits[idx_d1_valid]
# predCorr_d1_validUnits = predCorr_d1_allUnits[idx_d1_valid]
# rrCorr_d1_validUnits = rrCorr_d1_allUnits[idx_d1_valid]
# # fev_d1_allUnits = fev_d1_allUnits[idx_valid]

fev_d1_medianUnits = np.median(fev_d1_allUnits[idx_d1_valid])
fev_d1_stdUnits = np.std(fev_d1_allUnits[idx_d1_valid])
fev_d1_ci = 1.96*(fev_d1_stdUnits/len(idx_d1_valid)**.5)

predCorr_d1_medianUnits = np.nanmedian(predCorr_d1_allUnits[idx_d1_valid])
predCorr_d1_stdUnits = np.nanstd(predCorr_d1_allUnits[idx_d1_valid])
predCorr_d1_ci = 1.96*(predCorr_d1_stdUnits/len(idx_d1_valid)**.5)

rrCorr_d1_medianUnits = np.nanmedian(rrCorr_d1_allUnits[idx_d1_valid])
rrCorr_d1_stdUnits = np.nanstd(rrCorr_d1_allUnits[idx_d1_valid])
rrCorr_d1_ci = 1.96*(rrCorr_d1_stdUnits/len(idx_d1_valid)**.5)

idx_units_sorted = np.argsort(fev_d1_allUnits[idx_d1_valid])
idx_units_sorted = idx_d1_valid[idx_units_sorted]
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[1],idx_units_sorted[0]]
idx_unitsToPred = [30,43,16,21]

t_start = 1000
t_dur = 3000#obs_rate.shape[0]
t_end = t_start+t_dur-20
win_display = (t_start,t_start+t_dur)
font_size_ticks = 14

t_frame = 8
t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)/1000 # so time is now in seconds

# Trigger info
trig_saccs = data_val_info['triggers'][select_T:].copy(); trig_saccs[trig_saccs!=0] = np.nan;
trig_flash_bright = data_val_info['triggers'][select_T:].copy(); trig_flash_bright[trig_flash_bright!=+1] = np.nan 
trig_flash_dark = data_val_info['triggers'][select_T:].copy(); trig_flash_dark[trig_flash_dark!=-1] = np.nan; trig_flash_dark = np.abs(trig_flash_dark)

col_mdl = ('r')
lineWidth_mdl = [1.5]
lim_y = (0,6)
fig,axs = plt.subplots(2,2,figsize=(25,10))
axs = np.ravel(axs)
fig.suptitle('Training: '+lightLevel_1+' | Prediction: '+ val_dataset_1,fontsize=16)


for i in range(len(idx_unitsToPred)):
    l_base, = axs[i].plot(t_axis[t_start+samps_shift:t_end],obs_rate[t_start:t_end-samps_shift,idx_unitsToPred[i]],linewidth=6,color='darkgrey')
    l_base.set_label('Actual')
    l, = axs[i].plot(t_axis[t_start+samps_shift:t_end],pred_rate[t_start+samps_shift:t_end,idx_unitsToPred[i]],cols_lightLevels['photopic'],linewidth=lineWidth_mdl[0])
    l.set_label('Predicted: FEV = %.02f, Corr = %0.2f' %(fev_d1_allUnits[idx_unitsToPred[i]],predCorr_d1_allUnits[idx_unitsToPred[i]]))
    
    rate_max = np.max([obs_rate[t_start:t_end-samps_shift,idx_unitsToPred[i]],pred_rate[t_start+samps_shift:t_end,idx_unitsToPred[i]]])
    
    s = axs[i].stem(t_axis[t_start+samps_shift:t_end],rate_max*(1+trig_saccs[t_start+samps_shift:t_end]),markerfmt='')
    plt.setp(s,'color','green')

    s = axs[i].stem(t_axis[t_start+samps_shift:t_end],1+(rate_max*trig_flash_dark[t_start+samps_shift:t_end]),markerfmt='')
    plt.setp(s,'color','black')
    
    s = axs[i].stem(t_axis[t_start+samps_shift:t_end],1+(rate_max*trig_flash_bright[t_start+samps_shift:t_end]),markerfmt='')
    plt.setp(s,'color','orange')

    
    
    # axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (s)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Normalized spike rate (spikes/second)',fontsize=font_size_ticks)
    axs[i].set_title('unit_id: %d' %idx_unitsToPred[i])
    axs[i].legend()
    plt.setp(axs[i].get_xticklabels(), fontsize=font_size_ticks)

fevs = [fev_d1_medianUnits,0]
cis = [fev_d1_ci,0]
fig,axs = plt.subplots(2,1,figsize=(5,12))
fig.suptitle('')
font_size_ticks = 20

col_scheme = ('darkgrey','red')
# ax.yaxis.grid(True)
xlabel = [val_dataset_1,'']
axs[0].bar(xlabel,fevs,yerr=cis,align='center',capsize=6,alpha=.7,color='red',width=0.4)
axs[0].set_ylabel('FEV',fontsize=font_size_ticks)
axs[0].set_title('Training: '+lightLevel_1,fontsize=16)
axs[0].set_ylim((0,1.1))
axs[0].set_yticks(np.arange(0,1.1,.2))
# axs[0].text(.75,.45,'N = %d RGCs' %fev_allUnits.shape[1],fontsize=font_size_ticks)
axs[0].tick_params(axis='both',labelsize=font_size_ticks)


fevs = [predCorr_d1_medianUnits,0]
cis = [predCorr_d1_ci,0]
xlabel = [val_dataset_1,'']
axs[1].bar(xlabel,fevs,yerr=cis,align='center',capsize=6,alpha=.7,color=[col_scheme[1]],width=0.4)
axs[1].set_ylabel('Correlation Coeff',fontsize=font_size_ticks)
axs[1].set_title('',fontsize=font_size_ticks)
axs[1].set_ylim((0,1.1))
axs[1].set_yticks(np.arange(0,1.1,.2))
# axs[0].text(.75,.45,'N = %d RGCs' %fev_allUnits.shape[1],fontsize=font_size_ticks)
axs[1].tick_params(axis='both',labelsize=font_size_ticks)

# plt.hist(b,np.arange(-1,1.1,.1))
# plt.hist(a,np.arange(-1,1.1,.1))
# plt.ylim((0,6))
# plt.ylabel('num of cells')
# plt.xlabel('FEV')
# plt.show()


# %% D2: Test model
transferModel = False
if transferModel == True:
    mdl_select_d2 = 'CNN_2D'
    subFold_d2 = '8ms_resamp'
else:
    mdl_select_d2 = mdl_select
    subFold_d2 = subFold

idx_CNN_start = 6
val_dataset_2 = 'scotopic-1_mdl-rieke_s-10_p-10_e-20_g-2.5_k-0.01_h-3_b-15.7_hc-5.2_gd-20_preproc-added_norm-1_tb-4_RungeKutta_RF-2' 
correctMedian = False
samps_shift_2 = samps_shift

path_dataset_d2 = os.path.join('/home/saad/data/analyses/data_kiersten/',expDates[idx_exp],subFold_d2,'datasets')
fname_data_train_val_test = os.path.join(path_dataset_d2,(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
resp_orig = resp_orig['train']


obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]

filt_width = filt_temporal_width

if mdl_select_d2[:6]=='CNN_2D':
    data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
elif mdl_select_d2[:6] == 'CNN_3D':
    data_val = prepare_data_cnn3d(data_val,select_T,np.arange(data_val.y.shape[1]))
elif mdl_select_d2[:6] == 'convLS' or mdl_select_d2 == 'LSTM_CNN_2D':
    data_val = prepare_data_convLSTM(data_val,select_T,np.arange(data_val.y.shape[1]))
elif mdl_select_d2[:2]=='PR':
    pr_temporal_width = perf_allExps[select_exp][select_mdl][paramFileName]['model_params']['pr_temporal_width']
    data_val = prepare_data_pr_cnn2d(data_val,pr_temporal_width,np.arange(data_val.y.shape[1]))
    filt_width = pr_temporal_width
# elif mdl_select_d2[:4]=='PRFR':
#     data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))

elif select_mdl[:8]=='PR_CNN3D':
    pr_temporal_width = perf_allExps[select_exp][select_mdl][paramFileName]['model_params']['pr_temporal_width']
    data_val = prepare_data_cnn3d(data_val,pr_temporal_width,np.arange(data_val.y.shape[1]))
    obs_rate_allStimTrials_d1 = dataset_rr['stim_0']['val'][:,pr_temporal_width:,:]


if transferModel == True:
    x = Input(shape=data_val.X.shape[1:])
    n_cells = data_val.y.shape[1]
    y = x # BatchNormalization(axis=1,name='BatchNorm_postPR')(x)

    for layer in mdl.layers[idx_CNN_start:]:
        y = layer(y)

    mdl_d2 = Model(x, y, name='CNN_2D')
else:
    mdl_d2 = mdl

obs_rate = data_val.y
pred_rate = mdl_d2.predict(data_val.X)

if correctMedian==True:
    fname_data_train_val_test_d1 = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+val_dataset_1+'.h5'))
    _,_,_,_,_,_,resp_med_d1 = load_h5Dataset(fname_data_train_val_test_d1)
    resp_med_d1 = np.nanmedian(resp_med_d1['train'],axis=0)
    resp_med_d2 = np.nanmedian(resp_orig,axis=0)
    resp_mulFac = resp_med_d2/resp_med_d1;
    
    obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]
    obs_rate_allStimTrials_scotpic = obs_rate_allStimTrials_scotpic / resp_mulFac[None,None,:]
    
    obs_rate = obs_rate / resp_mulFac[None,:]

    
else:       
    obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_width:,:]



num_iters = 50
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = samps_shift_2)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)
# fev_d2_allUnits[fev_d2_allUnits<0] = 0
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
idx_d2_valid = idx_allUnits
idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<1.1)
idx_d2_valid = idx_allUnits[idx_d2_valid]

# idx_d2_valid = idx_d1_valid

# fev_d2_validUnits = fev_d2_allUnits[idx_d2_valid]
# predCorr_d2_validUnits = predCorr_d2_allUnits[idx_d2_valid]
# rrCorr_d2_validUnits = rrCorr_d2_allUnits[idx_d2_valid]
# # fev_d2_allUnits = fev_d2_allUnits[idx_valid]

fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)

predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)

rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)

# idx_units_sorted = np.argsort(predCorr_d2_allUnits[idx_d2_valid])
idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
idx_units_sorted = idx_d2_valid[idx_units_sorted]
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[1],idx_units_sorted[0]]
# idx_unitsToPred = [26,3,16,32]

# %
t_start = 10 + 45
t_dur = obs_rate.shape[0]
t_end = t_dur-20
win_display = (t_start,t_start+t_dur)
font_size_ticks = 14

t_frame = 8
t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)

col_mdl = ('r')
lineWidth_mdl = [2]
lim_y = (0,6)
fig,axs = plt.subplots(2,2,figsize=(25,10))
axs = np.ravel(axs)
fig.suptitle('Training: '+lightLevel_1+' | Prediction: '+ val_dataset_2,fontsize=16)


for i in range(len(idx_unitsToPred)):
    l_base, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate[t_start:t_end-samps_shift_2,idx_unitsToPred[i]],linewidth=4,color='darkgrey')
    l_base.set_label('Actual')
    l, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],pred_rate[t_start+samps_shift_2:t_end,idx_unitsToPred[i]],cols_lightLevels[val_dataset_2[:8]],linewidth=lineWidth_mdl[0])
    l.set_label('Predicted: FEV = %.02f, Corr = %0.2f' %(fev_d2_allUnits[idx_unitsToPred[i]],predCorr_d2_allUnits[idx_unitsToPred[i]]))
    
    # axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Normalized spike rate (spikes/second)',fontsize=font_size_ticks)
    axs[i].set_title('unit_id: %d' %idx_unitsToPred[i])
    axs[i].legend()
    plt.setp(axs[i].get_xticklabels(), fontsize=font_size_ticks)


# %Bar plots of photopic and scotopic light levels

fevs_d1 = [fev_d1_medianUnits,0]
fevs_d2 = [fev_d2_medianUnits,0]
cis_d1 = [fev_d1_ci,0]
cis_d2 = [fev_d2_ci,0]
fig,axs = plt.subplots(2,1,figsize=(5,10))
fig.suptitle(select_exp+'\nTraining: '+lightLevel_1,fontsize=16)
font_size_ticks = 16
font_size_labels = 16

col_scheme = (cols_lightLevels[val_dataset_1[:8]],cols_lightLevels[val_dataset_2[:8]])
# ax.yaxis.grid(True)
xpoints = np.atleast_1d((0,1))
xlabel_fev = [select_mdl,'']
axs[0].bar(xpoints-.2,fevs_d1,yerr=cis_d1,align='center',capsize=6,alpha=.7,color=col_scheme[0],width=0.4,label=val_dataset_1[:8])
axs[0].bar(xpoints+.2,fevs_d2,yerr=cis_d2,align='center',capsize=6,alpha=.7,color=col_scheme[1],width=0.4,label=val_dataset_2[:8])
axs[0].plot((-0.5,len(xlabel_fev)-0.5),(np.nanmax(fevs_d1),np.nanmax(fevs_d1)),color='green')
axs[0].set_xticks(xpoints)#(2*np.arange(0,fev_d1_medianUnits_allMdls.shape[0]))
axs[0].set_xticklabels(xlabel_fev)

axs[0].set_ylabel('FEV',fontsize=font_size_ticks)
axs[0].set_title('',fontsize=font_size_ticks)
axs[0].set_yticks(np.arange(-1,1.1,.1))
axs[0].set_ylim((0,1.1))
# axs[0].text(.75,.45,'N = %d RGCs' %fev_allUnits.shape[1],fontsize=font_size_ticks)
axs[0].tick_params(axis='both',labelsize=16)
axs[0].legend(loc='best',fontsize=font_size_labels)


fevs_d1 = [predCorr_d1_medianUnits,rrCorr_d1_medianUnits]
fevs_d2 = [predCorr_d2_medianUnits,rrCorr_d2_medianUnits]
cis_d1 = [predCorr_d1_ci,rrCorr_d1_ci]
cis_d2 = [predCorr_d2_ci,rrCorr_d2_ci]


# ax.yaxis.grid(True)
xpoints = np.atleast_1d((0,1))
xlabel_corr = [select_mdl,'RetinaReliab']

axs[1].bar(xpoints-.2,fevs_d1,yerr=cis_d1,align='center',capsize=6,alpha=.7,color=col_scheme[0],width=0.4,label=val_dataset_1[:8])
axs[1].bar(xpoints+.2,fevs_d2,yerr=cis_d2,align='center',capsize=6,alpha=.7,color=col_scheme[1],width=0.4,label=val_dataset_2[:8])
axs[1].plot((-0.5,len(xlabel_fev)-0.5),(np.nanmax(predCorr_d1_medianUnits),np.nanmax(predCorr_d1_medianUnits)),color='green')
axs[1].set_xticks(xpoints)#(2*np.arange(0,fev_d1_medianUnits_allMdls.shape[0]))
axs[1].set_xticklabels(xlabel_corr)

axs[1].set_ylabel('Correlation Coefficient',fontsize=font_size_ticks)
axs[1].set_title('',fontsize=font_size_ticks)
axs[1].set_yticks(np.arange(-0.6,1.1,.2))
axs[1].set_ylim((0,1.1))
# axs[0].text(.75,.45,'N = %d RGCs' %fev_allUnits.shape[1],fontsize=font_size_ticks)
axs[1].tick_params(axis='both',labelsize=16)
axs[1].legend(loc='best',fontsize=font_size_labels)

# %% Sustained, Transient split
unames_valid = perf_allExps[select_exp][select_mdl][paramFileName]['data_quality']['uname_selectedUnits'][idx_d2_valid]
idx_sust = np.arange(0,29)
idx_trans = np.arange(29,64)

idx_sust_valid = np.intersect1d(idx_d2_valid,idx_sust)
idx_trans_valid = np.intersect1d(idx_d2_valid,idx_trans)

fev_sust_allUnits = fev_d2_allUnits[idx_sust_valid]
fev_sust_medianUnits = np.median(fev_sust_allUnits)
fev_sust_stdUnits = np.std(fev_sust_allUnits)
fev_sust_ci = 1.96*(fev_sust_stdUnits/idx_sust_valid.shape[0]**.5)

fev_trans_allUnits = fev_d2_allUnits[idx_trans_valid]
fev_trans_medianUnits = np.median(fev_trans_allUnits)
fev_trans_stdUnits = np.std(fev_trans_allUnits)
fev_trans_ci = 1.96*(fev_trans_stdUnits/idx_trans_valid.shape[0]**.5)

fevs_d1 = [fev_d1_medianUnits,0]
fevs_d2 = [fev_sust_medianUnits,0]
fevs_d3 = [fev_trans_medianUnits,0]

cis_d1 = [fev_d1_ci,0]
cis_d2 = [fev_sust_ci,0]
cis_d3 = [fev_trans_ci,0]

fig,axs = plt.subplots(2,1,figsize=(5,10))
fig.suptitle(select_exp+'\nTraining: '+lightLevel_1,fontsize=16)
font_size_ticks = 16
font_size_labels = 16

col_scheme = (cols_lightLevels[val_dataset_1[:8]],cols_lightLevels[val_dataset_2[:8]])
# ax.yaxis.grid(True)
xpoints = np.atleast_1d((0,1))
xlabel_fev = [select_mdl,'']
axs[0].bar(xpoints-.2,fevs_d1,yerr=cis_d1,align='center',capsize=6,alpha=.7,color=col_scheme[0],width=0.4,label=val_dataset_1[:8])
axs[0].bar(xpoints+.2,fevs_d2,yerr=cis_d2,align='center',capsize=6,alpha=.7,color=col_scheme[1],width=0.4,label=val_dataset_2[:8]+'_sust')
axs[0].bar(xpoints+.6,fevs_d2,yerr=cis_d3,align='center',capsize=6,alpha=.7,color='m',width=0.4,label=val_dataset_2[:8]+'_trans')
axs[0].plot((-0.5,len(xlabel_fev)-0.5),(np.nanmax(fevs_d1),np.nanmax(fevs_d1)),color='green')
axs[0].set_xticks(xpoints)#(2*np.arange(0,fev_d1_medianUnits_allMdls.shape[0]))
axs[0].set_xticklabels(xlabel_fev)

axs[0].set_ylabel('FEV',fontsize=font_size_ticks)
axs[0].set_title('',fontsize=font_size_ticks)
axs[0].set_yticks(np.arange(-1,1.1,.1))
axs[0].set_ylim((0,1.1))
# axs[0].text(.75,.45,'N = %d RGCs' %fev_allUnits.shape[1],fontsize=font_size_ticks)
axs[0].tick_params(axis='both',labelsize=16)
axs[0].legend(loc='best',fontsize=font_size_labels)

# %%
unit_id = 54
plt.plot(t_axis[t_start+samps_shift:t_end],obs_rate[t_start:t_end-samps_shift,unit_id],linewidth=2,color='darkgrey')
plt.plot(t_axis[t_start+samps_shift:t_end],pred_rate[t_start+samps_shift:t_end,unit_id],cols_lightLevels[val_dataset_2],linewidth=1)

# %% D1: Heat maps
select_exp = 'retina1'
select_mdl = models_all[0]
select_param_x = 'C1_3d'
select_param_y = 'C2_3d'
select_param_z = 'C3_3d'
thresh_fev = 0

params_mdl = params_allExps[select_exp][select_mdl]

select_U = 0
select_T = 120
select_BN = 1
select_MP = 0
# select_TR = 1
select_C1_n = np.unique(params_mdl['C1_n'])#13
select_C1_s = np.unique(params_mdl['C1_s'])
select_C1_3d = np.unique(params_mdl['C1_3d'])
select_C2_n = np.unique(params_mdl['C2_n'])#13
select_C2_s = np.unique(params_mdl['C2_s'])
select_C2_3d = np.unique(params_mdl['C2_3d'])
select_C3_n = np.unique(params_mdl['C3_n'])#25
select_C3_s = np.unique(params_mdl['C3_s'])
select_C3_3d = np.unique(params_mdl['C3_3d'])

idx_interest = np.in1d(params_mdl['U'],select_U)
idx_interest = idx_interest & np.in1d(params_mdl['T'],select_T)
idx_interest = idx_interest & np.in1d(params_mdl['BN'],select_BN)
idx_interest = idx_interest & np.in1d(params_mdl['MP'],select_MP)
# idx_interest = idx_interest & np.in1d(params_mdl['TR'],select_TR)

idx_interest = idx_interest & np.in1d(params_mdl['C1_n'],select_C1_n)
idx_interest = idx_interest & np.in1d(params_mdl['C1_s'],select_C1_s)
idx_interest = idx_interest & np.in1d(params_mdl['C1_3d'],select_C1_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C2_n'],select_C2_n)
idx_interest = idx_interest & np.in1d(params_mdl['C2_s'],select_C2_s)
idx_interest = idx_interest & np.in1d(params_mdl['C2_3d'],select_C2_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C3_n'],select_C3_n)
idx_interest = idx_interest & np.in1d(params_mdl['C3_s'],select_C3_s)
idx_interest = idx_interest & np.in1d(params_mdl['C3_3d'],select_C3_3d)

# make array of all fevs for min max vals
fev_grand = ([])
for i in perf_allExps[select_exp][select_mdl]:
    if perf_allExps[select_exp][select_mdl][i]['model_performance']['num_trials'] == 1:
        rgb = perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_bestEpoch_allTr']
    else:
        rgb = np.mean(perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
    
    fev_grand.append(rgb)
fev_min, fev_max = np.round(np.nanmin(fev_grand),2), np.round(np.nanmax(fev_grand),2)
# fev_min, fev_max = .8, .9


# plot heatmap

fev_heatmap = np.zeros((len(eval('select_'+select_param_y)),len(eval('select_'+select_param_x)),len(eval('select_'+select_param_z))))
fev_heatmap[:] = np.nan

totalCombs = sum(idx_interest)
idx_interest = np.where(idx_interest)[0]

for i in idx_interest:
    idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
    idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
    idx_z = np.where(params_mdl[select_param_z][i] == eval('select_'+select_param_z))[0][0]
    
    paramName = paramNames_allExps[select_exp][select_mdl][i]
    if perf_allExps[select_exp][select_mdl][paramName]['model_performance']['num_trials'] == 1:
        rgb = perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr']
    else:
        rgb = np.mean(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
    rgb = np.nanmedian(rgb,axis=0)
    if rgb>thresh_fev:
        fev_heatmap[idx_y,idx_x,idx_z] = rgb

font_size_ticks = 10
fig_rows = 3
fig_cols = int(np.ceil(len(eval('select_'+select_param_z))/fig_rows))
fig,axs = plt.subplots(fig_rows,fig_cols,figsize=(20,15))
axs = np.ravel(axs)
axs = axs[:len(eval('select_'+select_param_z))]

color_map = plt.cm.get_cmap('hot')
reversed_color_map = color_map.reversed()

for l in range(len(eval('select_'+select_param_z))):
    param_z = eval('select_'+select_param_z)[l]

    im = axs[l].imshow(fev_heatmap[:,:,l],cmap=reversed_color_map,vmin = fev_min, vmax = fev_max)
    axs[l].set_yticks(range(0,len(eval('select_'+select_param_y))))
    axs[l].set_yticklabels(eval('select_'+select_param_y),fontsize=font_size_ticks)
    axs[l].set_xticks(range(0,len(eval('select_'+select_param_x))))
    axs[l].set_xticklabels(eval('select_'+select_param_x),fontsize=font_size_ticks)
    axs[l].set_xlabel(select_param_x)
    axs[l].set_ylabel(select_param_y)
    axs[l].set_title(select_param_z+' = '+str(param_z))
    # axs[l].set_aspect('equal', adjustable='datalim')
fig.colorbar(im)

plt.setp(axs,aspect='equal')

# %% D2: heatmaps
val_dataset_2 = 'photopic_preproc_cones_norm_1'

select_exp = 'retina1'
select_mdl = 'CNN_2D'
select_param_x = 'C1_n'
select_param_y = 'C2_n'
select_param_z = 'C3_n'
thresh_fev = 0.0

params_mdl = params_allExps[select_exp][select_mdl]

select_U = 0
select_T = 60
select_BN = 1
select_MP = 0
# select_TR = 1
select_C1_n = np.unique(params_mdl['C1_n'])  #np.atleast_1d(13)#np.unique(params_mdl['C1_n'])#13
select_C1_s = np.unique(params_mdl['C1_s'])
select_C1_3d = np.unique(params_mdl['C1_3d'])
select_C2_n = np.unique(params_mdl['C2_n'])#13
select_C2_s = np.unique(params_mdl['C2_s'])
select_C2_3d = np.unique(params_mdl['C2_3d'])
select_C3_n = np.unique(params_mdl['C3_n'])#25
select_C3_s = np.unique(params_mdl['C3_s'])
select_C3_3d = np.unique(params_mdl['C3_3d'])

idx_interest = np.in1d(params_mdl['U'],select_U)
idx_interest = idx_interest & np.in1d(params_mdl['T'],select_T)
idx_interest = idx_interest & np.in1d(params_mdl['BN'],select_BN)
idx_interest = idx_interest & np.in1d(params_mdl['MP'],select_MP)
# idx_interest = idx_interest & np.in1d(params_mdl['TR'],select_TR)

idx_interest = idx_interest & np.in1d(params_mdl['C1_n'],select_C1_n)
idx_interest = idx_interest & np.in1d(params_mdl['C1_s'],select_C1_s)
idx_interest = idx_interest & np.in1d(params_mdl['C1_3d'],select_C1_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C2_n'],select_C2_n)
idx_interest = idx_interest & np.in1d(params_mdl['C2_s'],select_C2_s)
idx_interest = idx_interest & np.in1d(params_mdl['C2_3d'],select_C2_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C3_n'],select_C3_n)
idx_interest = idx_interest & np.in1d(params_mdl['C3_s'],select_C3_s)
idx_interest = idx_interest & np.in1d(params_mdl['C3_3d'],select_C3_3d)


fev_d2_heatmap = np.zeros((len(eval('select_'+select_param_y)),len(eval('select_'+select_param_x)),len(eval('select_'+select_param_z))))
fev_d2_heatmap[:] = np.nan

totalCombs = sum(idx_interest)
idx_interest = np.where(idx_interest)[0]
#
for i in idx_interest:
    idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
    idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
    idx_z = np.where(params_mdl[select_param_z][i] == eval('select_'+select_param_z))[0][0]
    
    paramName = paramNames_allExps[select_exp][select_mdl][i]
    
    
    idx_bestTrial = np.argsort(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
    
    select_TR = idx_bestTrial[-1] + 1
    
    
    mdlFolder = paramName+'_TR-%02d' % select_TR
    
    path_model = os.path.join(path_mdl_drive,mdl_select,mdlFolder)
    
    try:
        mdl = load(os.path.join(path_model,mdlFolder))
        # idx_bestEpoch = np.argmax(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial[-1]])
        # fname_bestEpoch = os.path.join(path_model,'weights_'+mdlFolder+'_epoch-%03d.h5'%idx_bestEpoch)
        # mdl.load_weights(fname_bestEpoch)
       
        fname_data_train_val_test = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
        _,data_val,data_test,data_quality,dataset_rr,_,_ = load_h5Dataset(fname_data_train_val_test)
        if mdl_select[:6]=='CNN_2D':
            data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
        else:
            data_val = prepare_data_cnn3d(data_val,select_T,np.arange(data_val.y.shape[1]))

        resp_median_scotopic = data_quality['resp_median_allUnits']
        obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]
      
        samps_shift = 2
        
        obs_rate = data_val.y
        pred_rate = mdl.predict(data_val.X)
      
        
        num_iters = 2
        fev_scot_allUnits = np.empty((pred_rate.shape[1],num_iters))
        
        for i in range(num_iters):
            # fev_scot_allUnits[:,i],_,_,_ = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = samps_shift)
            _,_,fev_scot_allUnits[:,i],_ = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = samps_shift)       # correlation
        fev_scot_allUnits = np.mean(fev_scot_allUnits,axis=1)
    
        idx_scotopic_valid = np.logical_and(fev_scot_allUnits>0,fev_scot_allUnits<1.1)
        fev_scot_allUnits = fev_scot_allUnits[idx_scotopic_valid]
        
        fev_scot_medianUnits = np.nanmedian(fev_scot_allUnits)
       
        
        if fev_scot_medianUnits>thresh_fev:
            fev_d2_heatmap[idx_y,idx_x,idx_z] = fev_scot_medianUnits
    except:
        fev_d2_heatmap[idx_y,idx_x,idx_z] = np.nan
        
#%
fev_min, fev_max = np.round(np.nanmin(fev_d2_heatmap),2), np.round(np.nanmax(fev_d2_heatmap),2)
# fev_min, fev_max = 0.56,0.7
# %

font_size_ticks = 10
fig_rows = 3
fig_cols = int(np.ceil(len(eval('select_'+select_param_z))/fig_rows))
fig,axs = plt.subplots(fig_rows,fig_cols,figsize=(20,15))
axs = np.ravel(axs)
axs = axs[:len(eval('select_'+select_param_z))]

color_map = plt.cm.get_cmap('hot')
reversed_color_map = color_map.reversed()

for l in range(len(eval('select_'+select_param_z))):
    param_z = eval('select_'+select_param_z)[l]

    im = axs[l].imshow(fev_d2_heatmap[:,:,l],cmap=reversed_color_map,vmin = fev_min, vmax = fev_max)
    axs[l].set_yticks(range(0,len(eval('select_'+select_param_y))))
    axs[l].set_yticklabels(eval('select_'+select_param_y),fontsize=font_size_ticks)
    axs[l].set_xticks(range(0,len(eval('select_'+select_param_x))))
    axs[l].set_xticklabels(eval('select_'+select_param_x),fontsize=font_size_ticks)
    axs[l].set_xlabel(select_param_x)
    axs[l].set_ylabel(select_param_y)
    axs[l].set_title(select_param_z+' = '+str(param_z))
    # axs[l].set_aspect('equal', adjustable='datalim')
fig.colorbar(im)

plt.setp(axs,aspect='equal')

# %% D1 + D2 Heatmaps
val_dataset_1 = 'photopic'
val_dataset_2 = 'scotopic'

select_exp = 'retina1'
select_mdl = models_all[0]
select_param_x = 'C1_3d'
select_param_y = 'C2_3d'
select_param_z = 'C3_3d'
thresh_fev = 0.0

params_mdl = params_allExps[select_exp][select_mdl]

select_U = 0
select_T = 60
select_BN = 1
select_MP = 0
# select_TR = 1
select_C1_n = np.unique(params_mdl['C1_n'])  #np.atleast_1d(13)#np.unique(params_mdl['C1_n'])#13
select_C1_s = np.unique(params_mdl['C1_s'])
select_C1_3d = np.unique(params_mdl['C1_3d'])
select_C2_n = np.unique(params_mdl['C2_n'])#13
select_C2_s = np.unique(params_mdl['C2_s'])
select_C2_3d = np.unique(params_mdl['C2_3d'])
select_C3_n = np.unique(params_mdl['C3_n'])#25
select_C3_s = np.unique(params_mdl['C3_s'])
select_C3_3d = np.unique(params_mdl['C3_3d'])

idx_interest = np.in1d(params_mdl['U'],select_U)
idx_interest = idx_interest & np.in1d(params_mdl['T'],select_T)
idx_interest = idx_interest & np.in1d(params_mdl['BN'],select_BN)
idx_interest = idx_interest & np.in1d(params_mdl['MP'],select_MP)
# idx_interest = idx_interest & np.in1d(params_mdl['TR'],select_TR)

idx_interest = idx_interest & np.in1d(params_mdl['C1_n'],select_C1_n)
idx_interest = idx_interest & np.in1d(params_mdl['C1_s'],select_C1_s)
idx_interest = idx_interest & np.in1d(params_mdl['C1_3d'],select_C1_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C2_n'],select_C2_n)
idx_interest = idx_interest & np.in1d(params_mdl['C2_s'],select_C2_s)
idx_interest = idx_interest & np.in1d(params_mdl['C2_3d'],select_C2_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C3_n'],select_C3_n)
idx_interest = idx_interest & np.in1d(params_mdl['C3_s'],select_C3_s)
idx_interest = idx_interest & np.in1d(params_mdl['C3_3d'],select_C3_3d)


fev_d1_d2_heatmap = np.zeros((len(eval('select_'+select_param_y)),len(eval('select_'+select_param_x)),len(eval('select_'+select_param_z))))
fev_d1_d2_heatmap[:] = np.nan


totalCombs = sum(idx_interest)
idx_interest = np.where(idx_interest)[0]
#
for i in idx_interest:
    idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
    idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
    idx_z = np.where(params_mdl[select_param_z][i] == eval('select_'+select_param_z))[0][0]
    
    paramName = paramNames_allExps[select_exp][select_mdl][i]
    
    
    
    if perf_allExps[select_exp][select_mdl][paramName]['model_performance']['num_trials'] == 1:
        rgb = perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr']
    else:
        rgb = np.mean(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
    rgb = np.nanmedian(rgb,axis=0)
    if rgb>thresh_fev:
        fev_d1 = rgb

    idx_bestTrial = np.argsort(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_bestEpoch_allTr'])   
    select_TR = idx_bestTrial[-1]+1
    mdlFolder = paramName+'_TR-%02d' % select_TR
    
    path_model = os.path.join(path_mdl_drive,mdl_select,mdlFolder)
    
    try:
        mdl = load(os.path.join(path_model,mdlFolder))
        # idx_bestEpoch = np.argmax(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial[-1]])
        # fname_bestEpoch = os.path.join(path_model,'weights_'+mdlFolder+'_epoch-%03d.h5'%idx_bestEpoch)
        # mdl.load_weights(fname_bestEpoch)
       
        fname_data_train_val_test = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
        _,data_val,data_test,data_quality,dataset_rr,_,_ = load_h5Dataset(fname_data_train_val_test)
        if mdl_select[:6]=='CNN_2D':
            data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
        else:
            data_val = prepare_data_cnn3d(data_val,select_T,np.arange(data_val.y.shape[1]))

        resp_median_scotopic = data_quality['resp_median_allUnits']
        obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]
      
        samps_shift = 2
        
        obs_rate = data_val.y
        pred_rate = mdl.predict(data_val.X)
      
        
        num_iters = 2
        fev_scot_allUnits = np.empty((pred_rate.shape[1],num_iters))
        
        for i in range(num_iters):
            fev_scot_allUnits[:,i],_,_,_ = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = samps_shift)   
        fev_scot_allUnits = np.mean(fev_scot_allUnits,axis=1)
    
        idx_scotopic_valid = np.logical_and(fev_scot_allUnits>0,fev_scot_allUnits<1.1)
        fev_scot_allUnits = fev_scot_allUnits[idx_scotopic_valid]
        
        fev_scot_medianUnits = np.nanmedian(fev_scot_allUnits)
       
        
        if fev_scot_medianUnits>thresh_fev:
            fev_d2 = fev_scot_medianUnits
        else:
            fev_d2 = np.nan
    except:
        fev_d2 = np.nan
        
    fev_d1_d2 = np.nanmean((fev_d1,fev_d2))
    fev_d1_d2_heatmap[idx_y,idx_x,idx_z] = fev_d1_d2
        
#%
fev_min, fev_max = np.round(np.nanmin(fev_d1_d2_heatmap),2), np.round(np.nanmax(fev_d1_d2_heatmap),2)
fev_max_idx = np.nanargmax(fev_d1_d2_heatmap)
fev_max_idx = np.unravel_index(fev_max_idx,fev_d1_d2_heatmap.shape)

# fev_min, fev_max = 0.5,0.87
# %

font_size_ticks = 10
fig_rows = 3
fig_cols = int(np.ceil(len(eval('select_'+select_param_z))/fig_rows))
fig,axs = plt.subplots(fig_rows,fig_cols,figsize=(20,15))
axs = np.ravel(axs)
axs = axs[:len(eval('select_'+select_param_z))]

color_map = plt.cm.get_cmap('hot')
reversed_color_map = color_map.reversed()

for l in range(len(eval('select_'+select_param_z))):
    param_z = eval('select_'+select_param_z)[l]

    im = axs[l].imshow(fev_d1_d2_heatmap[:,:,l],cmap=reversed_color_map,vmin = fev_min, vmax = fev_max)
    axs[l].set_yticks(range(0,len(eval('select_'+select_param_y))))
    axs[l].set_yticklabels(eval('select_'+select_param_y),fontsize=font_size_ticks)
    axs[l].set_xticks(range(0,len(eval('select_'+select_param_x))))
    axs[l].set_xticklabels(eval('select_'+select_param_x),fontsize=font_size_ticks)
    axs[l].set_xlabel(select_param_x)
    axs[l].set_ylabel(select_param_y)
    axs[l].set_title(select_param_z+' = '+str(param_z))
    # axs[l].set_aspect('equal', adjustable='datalim')
fig.colorbar(im)

plt.setp(axs,aspect='equal')


# %% Feature maps
select_exp = 'retina1'
select_mdl = models_all[0] #'CNN_2D' #'CNN_2D_chansVary'#'CNN_2D_filtsVary'
params_mdl = params_allExps[select_exp][select_mdl]

select_U = 0#0.15
select_T = 60
select_BN = 1
select_MP = 0
# select_TR = 1
select_C1_n = 18
select_C1_s = 3
select_C1_3d = 25
select_C2_n = 25#24
select_C2_s = 2#2
select_C2_3d = 5
select_C3_n = 18#22
select_C3_s = 1#1
select_C3_3d = 32

paramFileName = paramsToName(select_mdl,U=select_U,T=select_T,BN=select_BN,MP=select_MP,
                 C1_n=select_C1_n,C1_s=select_C1_s,C1_3d=select_C1_3d,
                 C2_n=select_C2_n,C2_s=select_C2_s,C2_3d=select_C2_3d,
                 C3_n=select_C3_n,C3_s=select_C3_s,C3_3d=select_C3_3d)


idx_bestTrial = np.argmax(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
idx_bestEpoch = np.argmax(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])

select_TR = idx_bestTrial+1

mdlFolder = paramFileName+'_TR-%02d' % select_TR
path_model = os.path.join(path_mdl_drive,mdl_select,mdlFolder)
mdl = load(os.path.join(path_model,mdlFolder))
fname_bestEpoch = os.path.join(path_model,'weights_'+mdlFolder+'_epoch-%03d.h5'%idx_bestEpoch)
if os.path.exists(fname_bestEpoch):
    mdl.load_weights(fname_bestEpoch)


fname_data_train_val_test = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+lightLevel_1+'.h5'))
data_train,_,_,_,_,_,_ = load_h5Dataset(fname_data_train_val_test)
if mdl_select[:6]=='CNN_2D':
    data_train = prepare_data_cnn2d(data_train,select_T,np.arange(data_train.y.shape[1]))
    chunk_size = data_train.y.shape[0]
else:
    data_train = prepare_data_cnn3d(data_train,select_T,np.arange(data_train.y.shape[1]))
    chunk_size = 10000

stim_size = np.array(mdl.input.shape[-2:])
idx_extra = 2
# strongestUnit = np.array([9,15])

mdl_params = {
    'chan_n': select_C1_n,
    'filt_temporal_width': select_T,
    'chan_s': select_C1_s
    }

layer_name = 'conv3d_1'
strongestUnit = np.array([4,5,-1])
rwa_mean = featureMaps.get_featureMaps(data_train.X,mdl,mdl_params,layer_name,strongestUnit,chunk_size)
# 
font_size_ticks = 10
fig_rows = 5
fig_cols = int(np.ceil(rwa_mean.shape[0]/fig_rows))
t_frame = 17
t_axis = np.flip(-1*np.arange(0,select_T*t_frame,t_frame))

fig_spat,axs_spat = plt.subplots(fig_rows,fig_cols,figsize=(15,12))
fig_spat.suptitle('Layer: '+layer_name+' | Spatial features',fontsize=16)
axs_spat = np.ravel(axs_spat)
axs_spat = axs_spat[:rwa_mean.shape[0]]

fig_temp,axs_temp = plt.subplots(fig_rows,fig_cols,figsize=(20,12))
fig_temp.suptitle('Layer: '+layer_name+' | Temporal features',fontsize=16)
axs_temp = np.ravel(axs_temp)
axs_temp = axs_temp[:rwa_mean.shape[0]]

for i in range(rwa_mean.shape[0]):
    spatial_feature, temporal_feature = decompose(rwa_mean[i,:,:,:])
    

        
    im = axs_spat[i].imshow(spatial_feature,cmap='bwr')
    axs_temp[i].plot(t_axis,temporal_feature)
    # if len(layer_name) == 6:
    #     axs_temp[i].plot(t_axis,temporal_feature)
    # else:
    #     axs_temp[i].plot(temporal_feature)
fig_spat.colorbar(im)   # axs_temp[i].yticks([])


# %% find how many epochs are required
select_exp = 'retina1'
select_mdl = 'CNN_2D'
for i in paramNames_unique:#[800:1000]:
    idx_bestTrial = np.argmax(perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
    fev_allEpochs = perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial]
    
    plt.plot(fev_allEpochs)
plt.show()
    


# %% correlation
def cross_corr(y1, y2):
  """Calculates the cross correlation and lags without normalization.

  The definition of the discrete cross-correlation is in:
  https://www.mathworks.com/help/matlab/ref/xcorr.html

  Args:
    y1, y2: Should have the same length.

  Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
  """
  if len(y1) != len(y2):
    raise ValueError('The lengths of the inputs should be the same.')

  y1_auto_corr = np.dot(y1, y1) / len(y1)
  y2_auto_corr = np.dot(y2, y2) / len(y1)
  corr = np.correlate(y1, y2, mode='same')
  # The unbiased sample size is N - lag.
  unbiased_sample_size = np.correlate(
      np.ones(len(y1)), np.ones(len(y1)), mode='same')
  corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
  shift = len(y1) // 2

  max_corr = np.max(corr)
  argmax_corr = np.argmax(corr)
  return max_corr, argmax_corr - shift


# % cross correlation

obs_rate = data_val.y[100:,:]
pred_rate = mdl.predict(data_val.X)[100:,:]

lags = np.empty(obs_rate.shape[1])

for i in range(obs_rate.shape[1]):
    a = obs_rate[:,i]
    b = pred_rate[:,i]
    
    corr,lags[i] = cross_corr(a, b)

lags_median = np.median(lags)
print(lags_median)
# lags = lags+2
# lags[lags>7]=7
# lags[lags<0]=7

# %%3D plot
# fev_surfacePlot = np.zeros((len(eval('select_'+select_param_x)),len(eval('select_'+select_param_y)),len(eval('select_'+select_param_z))))
# fev_surfacePlot[:] = np.nan

# totalCombs = sum(idx_interest)
# idx_interest = np.where(idx_interest)[0]

# for i in idx_interest:
#     idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
#     idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
#     idx_z = np.where(params_mdl[select_param_z][i] == eval('select_'+select_param_z))[0][0]
    
#     paramName = paramNames_allExps[select_exp][select_mdl][i]
#     if perf_allExps[select_exp][select_mdl][paramName]['model_performance']['num_trials'] == 1:
#         rgb = perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr']
#     else:
#         rgb = np.mean(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
#     rgb = np.nanmedian(rgb,axis=0)
    
#     if rgb>.85:#thresh_fev:
#         fev_surfacePlot[idx_x,idx_y,idx_z] = rgb

# fig,axs = plt.subplots(1,1,subplot_kw={"projection": "3d"})

# X,Y,Z = np.meshgrid(eval('select_'+select_param_x),eval('select_'+select_param_y),eval('select_'+select_param_z))

# color_map = plt.cm.get_cmap('hot')
# reversed_color_map = color_map.reversed()
# surf = axs.scatter3D(X,Y,Z,c=fev_surfacePlot,cmap = reversed_color_map,linewidth=0,antialiased=False)
# axs.set_xlabel(select_param_x)
# axs.set_ylabel(select_param_y)
# axs.set_zlabel(select_param_z)
# axs.set_title(select_mdl)
# fig.colorbar(surf)
# axs.view_init(10, 320)
# plt.draw()

# %% Bar graph all models


def perf_bar(select_exp,select_mdl,mdl,val_dataset,select_T):
    fname_data_train_val_test = os.path.join(path_dataset,(select_exp+'_dataset_train_val_test_'+val_dataset+'.h5'))
    _,data_val,_,_,dataset_rr,_,resp_orig = load_h5Dataset(fname_data_train_val_test)
    
    resp_orig = resp_orig['train']
    
    
    if select_mdl[:6]=='CNN_2D':
        data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
    elif select_mdl[:6]=='CNN_3D':
        data_val = prepare_data_cnn3d(data_val,select_T,np.arange(data_val.y.shape[1]))
        
    elif select_mdl[:6]=='convLS' or select_mdl == 'LSTM_CNN_2D':
        data_val = prepare_data_convLSTM(data_val,select_T,np.arange(data_val.y.shape[1]))
        
    
    filt_temporal_width = select_T
    obs_rate_allStimTrials_d1 = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]
    obs_rate = data_val.y
    
    if correctMedian==True:
        fname_data_train_val_test_d1 = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+val_dataset_train+'.h5'))
        _,_,_,_,_,_,resp_med_d1 = load_h5Dataset(fname_data_train_val_test_d1)
        resp_med_d1 = np.nanmedian(resp_med_d1['train'],axis=0)
        resp_med_d2 = np.nanmedian(resp_orig,axis=0)
        resp_mulFac = resp_med_d2/resp_med_d1;
        
        pred_rate = mdl.predict(data_val.X)
        pred_rate = pred_rate * resp_mulFac[None,:]
    
        
    else:       
        pred_rate = mdl.predict(data_val.X)
    _ = gc.collect()
    tf.keras.backend.clear_session()
    # obs_rate = data_test.y
    # pred_rate = mdl.predict(data_test.X)
    
    
    num_iters = 50
    fev_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
    fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
    predCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
    rrCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
    
    for i in range(num_iters):
        fev_d1_allUnits[:,i], fracExplainableVar[:,i], predCorr_d1_allUnits[:,i], rrCorr_d1_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_d1,pred_rate,0,RR_ONLY=False,lag = samps_shift)
    
    
    fev_d1_allUnits = np.mean(fev_d1_allUnits,axis=1)
    fracExplainableVar = np.mean(fracExplainableVar,axis=1)
    predCorr_d1_allUnits = np.mean(predCorr_d1_allUnits,axis=1)
    rrCorr_d1_allUnits = np.mean(rrCorr_d1_allUnits,axis=1)
    
    idx_allUnits = np.arange(fev_d1_allUnits.shape[0])
    idx_d1_valid = idx_allUnits
    idx_d1_valid = np.logical_and(fev_d1_allUnits>-1,fev_d1_allUnits<1.1)
    idx_d1_valid = idx_allUnits[idx_d1_valid]
    
    
    fev_d1_medianUnits = np.median(fev_d1_allUnits[idx_d1_valid])
    fev_d1_stdUnits = np.std(fev_d1_allUnits[idx_d1_valid])
    fev_d1_ci = 1.96*(fev_d1_stdUnits/len(idx_d1_valid)**.5)
    
    predCorr_d1_medianUnits = np.median(predCorr_d1_allUnits[idx_d1_valid])
    predCorr_d1_stdUnits = np.std(predCorr_d1_allUnits[idx_d1_valid])
    predCorr_d1_ci = 1.96*(predCorr_d1_stdUnits/len(idx_d1_valid)**.5)
    
    rrCorr_d1_medianUnits = np.median(rrCorr_d1_allUnits[idx_d1_valid])
    rrCorr_d1_stdUnits = np.std(rrCorr_d1_allUnits[idx_d1_valid])
    rrCorr_d1_ci = 1.96*(rrCorr_d1_stdUnits/len(idx_d1_valid)**.5)
    
    
    
    return fev_d1_allUnits,fev_d1_medianUnits, fev_d1_ci, predCorr_d1_allUnits, predCorr_d1_medianUnits, predCorr_d1_ci, rrCorr_d1_medianUnits, rrCorr_d1_ci, idx_d1_valid 

select_exp = expDates[0]
select_mdl = models_all[0] #'CNN_2D' #'CNN_2D_chansVary'#'CNN_2D_filtsVary'

val_dataset_train= lightLevel_1      # ['scotopic','photopic']
val_dataset_test = lightLevel_1
val_dataset_test_2 = 'scotopic' #'scotopic' #'scotopic-1_preproc-added_norm-1_rfac-2' 

if len(val_dataset_train) == 8 and len(val_dataset_test_2)!=8:
    raise ValueError('wrong datasets')

if len(val_dataset_train) > 8 and len(val_dataset_test_2)==8:
    raise ValueError('wrong datasets')    
    
correctMedian = False

fname_data_train_val_test = os.path.join(path_dataset,(select_exp+'_dataset_train_val_test_'+val_dataset_test+'.h5'))
_,_,_,_,_,parameters,_ = load_h5Dataset(fname_data_train_val_test)
samps_shift = int(np.array(parameters['samps_shift']))

fev_d1_medianUnits_allMdls = np.atleast_1d([])
fev_d1_ci_allMdls = np.atleast_1d([])
predCorr_d1_medianUnits_allMdls = np.atleast_1d([])
predCorr_d1_ci_allMdls = np.atleast_1d([])

fev_d2_medianUnits_allMdls = np.atleast_1d([])
fev_d2_ci_allMdls = np.atleast_1d([])
predCorr_d2_medianUnits_allMdls = np.atleast_1d([])
predCorr_d2_ci_allMdls = np.atleast_1d([])

trainingSamps_allMdls = np.atleast_1d([])

mdl_name_params = []
item_xlabel = 'BN'


for select_mdl in models_all:

    paramFileName = list(perf_allExps[select_exp][select_mdl].keys())   
    
    for p in range(0,len(paramFileName)):
        
        print('Evaluating %d of %d' %(p+1,len(paramFileName)))
    
        idx_bestTrial = np.nanargmax(perf_allExps[select_exp][select_mdl][paramFileName[p]]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
        idx_bestEpoch = np.nanargmax(perf_allExps[select_exp][select_mdl][paramFileName[p]]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])
        # plt.plot(perf_allExps[select_exp][select_mdl][paramFileName[p]]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])
        
        select_TR = idx_bestTrial+1
        
        # resp_median_photopic = perf_allExps[select_exp][select_mdl][paramFileName]['dataset_pred']['resp_median_allUnits']
        mdlFolder = paramFileName[p]+'_TR-%02d' % select_TR
        
        path_model = os.path.join(path_mdl_drive,select_mdl,mdlFolder)
        mdl = load(os.path.join(path_model,mdlFolder))
    
        select_T = perf_allExps[select_exp][select_mdl][paramFileName[p]]['model_params']['temporal_width']
        
        fev_d1_allUnits,fev_d1_medianUnits, fev_d1_ci, predCorr_d1_allUnits, predCorr_d1_medianUnits, predCorr_d1_ci, rrCorr_d1_medianUnits, rrCorr_d1_ci, idx_d1_valid = perf_bar(select_exp,select_mdl,mdl,val_dataset_test,select_T)

        fev_d1_medianUnits_allMdls = np.append(fev_d1_medianUnits_allMdls,fev_d1_medianUnits)
        fev_d1_ci_allMdls = np.append(fev_d1_ci_allMdls,fev_d1_ci)
        predCorr_d1_medianUnits_allMdls = np.append(predCorr_d1_medianUnits_allMdls,predCorr_d1_medianUnits)
        predCorr_d1_ci_allMdls = np.append(predCorr_d1_ci_allMdls,predCorr_d1_ci)
        
        if val_dataset_test_2 != 0:
            fev_d2_allUnits,fev_d2_medianUnits, fev_d2_ci, predCorr_d2_allUnits, predCorr_d2_medianUnits, predCorr_d2_ci, rrCorr_d2_medianUnits, rrCorr_d2_ci, idx_d2_valid = perf_bar(select_exp,select_mdl,mdl,val_dataset_test_2,select_T)

            fev_d2_medianUnits_allMdls = np.append(fev_d2_medianUnits_allMdls,fev_d2_medianUnits)
            fev_d2_ci_allMdls = np.append(fev_d2_ci_allMdls,fev_d2_ci)
            predCorr_d2_medianUnits_allMdls = np.append(predCorr_d2_medianUnits_allMdls,predCorr_d2_medianUnits)
            predCorr_d2_ci_allMdls = np.append(predCorr_d2_ci_allMdls,predCorr_d2_ci)


        trainingSamps_allMdls = np.append(trainingSamps_allMdls,int(paramFileName[p][-2:]))

        # rgb = select_mdl#+'_'+paramFileName[p][13:-10]
        rgb = item_xlabel+'-'+str(params_allExps[select_exp][select_mdl][item_xlabel][p])
        mdl_name_params.append(rgb)
        
# %        
fig,axs = plt.subplots(2,1,figsize=(5,10))
fig.suptitle(select_exp+'\nTraining: '+val_dataset_train,fontsize=16)
font_size_ticks = 16
font_size_labels = 16


col_scheme = ('darkgrey',cols_lightLevels[val_dataset_test[:8]])
# ax.yaxis.grid(True)
xpoints = np.arange(fev_d1_medianUnits_allMdls.shape[0])
xlabel_fev = mdl_name_params
axs[0].bar(xpoints-.2,fev_d1_medianUnits_allMdls,yerr=fev_d1_ci_allMdls,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test[:8]],width=0.4,label=val_dataset_test[:8])
axs[0].bar(xpoints+.2,fev_d2_medianUnits_allMdls,yerr=fev_d2_ci_allMdls,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test_2[:8]],width=0.4,label=val_dataset_test_2[:8])
axs[0].bar(len(xpoints)-.2,0,yerr=0,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test[:8]],width=0.4)
axs[0].bar(len(xpoints)+.2,0,yerr=0,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test_2[:8]],width=0.4)

axs[0].set_xticks(xpoints)#(2*np.arange(0,fev_d1_medianUnits_allMdls.shape[0]))
axs[0].set_xticklabels(xlabel_fev)
axs[0].set_yticks(np.arange(-1,1.1,.1))
axs[0].set_ylabel('FEV',fontsize=font_size_ticks)
axs[0].set_title('',fontsize=font_size_ticks)
axs[0].set_ylim((0,1.1))
axs[0].tick_params(axis='both',labelsize=16)
axs[0].plot((-0.5,len(mdl_name_params)-0.5),(np.nanmax(fev_d1_medianUnits_allMdls),np.nanmax(fev_d1_medianUnits_allMdls)),color='green')
axs[0].legend(loc='best',fontsize=font_size_labels)


axs[1].bar(xpoints-.2,predCorr_d1_medianUnits_allMdls,yerr=predCorr_d1_ci_allMdls,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test[:8]],width=0.4,label=val_dataset_test[:8])
axs[1].bar(xpoints+.2,predCorr_d2_medianUnits_allMdls,yerr=predCorr_d2_ci_allMdls,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test_2[:8]],width=0.4,label=val_dataset_test_2[:8])
axs[1].bar(len(xpoints)-.2,rrCorr_d1_medianUnits,yerr=rrCorr_d1_ci,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test[:8]],width=0.4)
axs[1].bar(len(xpoints)+.2,rrCorr_d2_medianUnits,yerr=rrCorr_d2_ci,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test_2[:8]],width=0.4)
xpoints = np.arange(fev_d1_medianUnits_allMdls.shape[0]+1)
xlabel_corr = list(mdl_name_params)
xlabel_corr.append('RetinaReliab')
axs[1].set_xticks(xpoints)#(2*np.arange(0,fev_d1_medianUnits_allMdls.shape[0]))
axs[1].set_xticklabels(xlabel_corr)
axs[1].set_yticks(np.arange(-0.6,1.1,.2))
axs[1].set_ylabel('Correlation Coefficient',fontsize=font_size_ticks)
axs[1].set_title('',fontsize=font_size_ticks)
axs[1].set_ylim((0,1.1))
axs[1].tick_params(axis='both',labelsize=16)
axs[1].plot((-0.5,len(mdl_name_params)),(np.nanmax(predCorr_d1_medianUnits_allMdls),np.nanmax(predCorr_d1_medianUnits_allMdls)),color='green')
# axs[1].legend(loc='best',fontsize=font_size_labels)


