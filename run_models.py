#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:29:28 2021

@author: saad
"""



from model.parser import parser_run_model


def run_model(expDate,mdl_name,path_model_save_base,fname_data_train_val_test,path_existing_mdl='',idx_CNN_start=5,
                            saveToCSV=1,runOnCluster=0,
                            temporal_width=40, thresh_rr=0,
                            chan1_n=8, filt1_size=13, filt1_3rdDim=20,
                            chan2_n=0, filt2_size=0, filt2_3rdDim=0,
                            chan3_n=0, filt3_size=0, filt3_3rdDim=0,
                            pr_temporal_width = 180,
                            nb_epochs=100,bz_ms=10000,trainingSamps_dur=0,validationSamps_dur=1,
                            BatchNorm=1,BatchNorm_train=0,MaxPool=1,c_trial=1,
                            lr=0.01,USE_CHUNKER=0,CONTINUE_TRAINING=1,info='',
                            path_dataset_base='/home/saad/data/analyses/data_kiersten'):

          
# %% prepare data
    
# import needed modules
    import numpy as np
    import os
    import math
    import csv

    import tensorflow as tf
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth  = True
    config.gpu_options.per_process_gpu_memory_fraction = .9
    tf.compat.v1.Session(config=config)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # tf.compat.v1.disable_eager_execution()

    from tensorflow.keras.layers import Input
    
    from model.data_handler import load_h5Dataset, prepare_data_cnn3d, prepare_data_cnn2d, prepare_data_convLSTM, check_trainVal_contamination, prepare_data_pr_cnn2d,pr_cnn2d_multipr
    from model.performance import save_modelPerformance, model_evaluate, model_evaluate_new
    import model.metrics as metrics
    from model.models import model_definitions, get_model_memory_usage, cnn_3d, cnn_2d, pr_cnn2d, prfr_cnn2d,pr_cnn2d_fixed, pr_cnn3d, prfr_cnn2d_fixed, prfr_cnn2d_noTime, prfr_cnn2d_multipr
    from model.train_model import train, chunker
    from model.load_savedModel import load
    
    import gc
    import datetime
    
    from collections import namedtuple
    Exptdata = namedtuple('Exptdata', ['X', 'y'])

    # if only 1 layer cnn then set all parameters for next layers to 0
    if chan2_n == 0:
        filt2_size = 0
        filt2_3rdDim = 0
        
        chan3_n = 0
        filt3_size = 0
        filt3_3rdDim = 0 
        
    if chan3_n == 0:
        filt3_size = 0
        filt3_3rdDim = 0 

    # path to save results to - PARAMETERIZE THIS
    if runOnCluster==1:
        path_save_performance = '/home/sidrees/scratch/RetinaPredictors_MOV/performance'
    else:
        path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors_MOV/performance'
    
    
    if not os.path.exists(path_save_performance):
        os.makedirs(path_save_performance)
          
    
# load train val and test datasets from saved h5 file
    # load_h5dataset is a function to load training and validation data from h5 dataset. We can extract all data or a subset using the nsamps arguments.
    # data_train, val and test are named tuples. data_train.X contains the stimulus with dimensions [samples,y pixels, x pixels]
    # and data_train.y contains the spikerate normalized by median [samples,numOfCells]
    data_train,data_val,data_test,data_quality,dataset_rr,parameters,_ = load_h5Dataset(fname_data_train_val_test,nsamps_val=validationSamps_dur,nsamps_train=trainingSamps_dur,LOAD_ALL_TR=True)
    t_frame = parameters['t_frame']     # time in ms of one frame/sample 
    
    
# Arrange data according to the model
    idx_unitsToTake = data_quality['idx_unitsToTake']   # unit/cell id of the cells present in the dataset. [length should be same as 2nd dimension of data_train.y]
    idx_unitsToTake
    
    # Data will be rolled so that each sample has a temporal width. Like N frames of movie in one sample. The duration of each frame is in t_frame
    # if the model has a photoreceptor layer, then the PR layer has a termporal width of pr_temporal_width, which before convs will be chopped off to temporal width
    # this is done to get rid of boundary effects. pr_temporal_width > temporal width
    if mdl_name[:2] == 'PR':    # in this case the rolling width should be that of PR
        temporal_width_prepData = pr_temporal_width
        temporal_width_eval = pr_temporal_width
        
    else:   # in all other cases its same as temporal width
        temporal_width_prepData = temporal_width
        temporal_width_eval = temporal_width    # termporal width of each sample. Like how many frames of movie in one sample

    
    modelNames_all = model_definitions()    # get all model names
    modelNames_2D = modelNames_all[0]
    modelNames_3D = modelNames_all[1]
    
    # prepare data according to model. Roll and adjust dimensions according to 2D or 3D model
    if mdl_name in modelNames_2D:
        data_train = prepare_data_cnn2d(data_train,temporal_width_prepData,np.arange(len(idx_unitsToTake)))     # [samples,temporal_width,rows,columns]
        data_test = prepare_data_cnn2d(data_test,temporal_width_prepData,np.arange(len(idx_unitsToTake)))
        data_val = prepare_data_cnn2d(data_val,temporal_width_prepData,np.arange(len(idx_unitsToTake)))   
        
        filt1_3rdDim=0
        filt2_3rdDim=0
        filt3_3rdDim=0

        
    elif mdl_name in modelNames_3D:
        data_train = prepare_data_cnn3d(data_train,temporal_width_prepData,np.arange(len(idx_unitsToTake)))
        data_test = prepare_data_cnn3d(data_test,temporal_width_prepData,np.arange(len(idx_unitsToTake)))
        data_val = prepare_data_cnn3d(data_val,temporal_width_prepData,np.arange(len(idx_unitsToTake)))

    # Clean this up
    if BatchNorm==1:
        bn_val=1
    else:
        bn_val=0
    
    
    if MaxPool==1:
        mp_val=1
    else:
        mp_val=0       
        
    
    bz = math.ceil(bz_ms/t_frame)   # input batch size (bz_ms) is in ms. Convert into samples
    
    x = Input(shape=data_train.X.shape[1:]) # keras input layer
    n_cells = data_train.y.shape[1]         # number of units in output layer

# %% Select model 
    # Figure out a better way of fname_model
    
    if mdl_name == 'CNN_3D':       
        mdl = cnn_3d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim, chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim, chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_T-%03d_C1-%02d-%02d-%02d_C2-%02d-%02d-%02d_C3-%02d-%02d-%02d_BN-%d_MP-%d_LR-%0.4f_TR-%02d' %(thresh_rr,temporal_width,chan1_n,filt1_size,filt1_3rdDim,
                                                                                     chan2_n,filt2_size,filt2_3rdDim,
                                                                                     chan3_n,filt3_size,filt3_3rdDim,
                                                                                     bn_val,mp_val,lr,c_trial)
    elif mdl_name=='CNN_2D':
        mdl = cnn_2d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_LR-%0.4f_TR-%02d' %(thresh_rr,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,lr,c_trial)
        
    elif mdl_name=='PR_CNN2D':
        mdl = pr_cnn2d(x, n_cells, filt_temporal_width = temporal_width, chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_P-%03d_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_LR-%0.4f_TR-%02d' %(thresh_rr,pr_temporal_width,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,lr,c_trial)
        
        
    elif mdl_name=='PR_CNN2D_MULTIPR':
        mdl = pr_cnn2d_multipr(x, n_cells, filt_temporal_width = temporal_width, chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_P-%03d_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_LR-%0.4f_TR-%02d' %(thresh_rr,pr_temporal_width,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,lr,c_trial)        


    elif mdl_name=='PRFR_CNN2D':
        mdl = prfr_cnn2d(x, n_cells, filt_temporal_width = temporal_width, chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_P-%03d_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_LR-%0.4f_TR-%02d' %(thresh_rr,pr_temporal_width,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,lr,c_trial)
        
    elif mdl_name=='PRFR_CNN2D_MULTIPR':
        mdl = prfr_cnn2d_multipr(x, n_cells, filt_temporal_width = temporal_width, chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_P-%03d_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_LR-%0.4f_TR-%02d' %(thresh_rr,pr_temporal_width,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,lr,c_trial)        
       
    elif mdl_name=='PR_CNN3D':
        mdl = pr_cnn3d(x, n_cells, filt_temporal_width = temporal_width, chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_P-%03d_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_LR-%0.4f_TR-%02d' %(thresh_rr,pr_temporal_width,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,lr,c_trial)
        
    elif mdl_name=='PR_CNN2D_fixed':    # this is where the PR layer is fixed and only CNNs are trained. Hmm or the other way around.
        rgb = os.path.split(path_existing_mdl)[-1]
        mdl_existing = load(os.path.join(path_existing_mdl,rgb))
        # idx_CNN_start = 4
        

        mdl = pr_cnn2d_fixed(mdl_existing,idx_CNN_start,x, n_cells, filt_temporal_width=temporal_width,
                             chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size,
                             BatchNorm=BatchNorm,MaxPool=MaxPool,BatchNorm_train = BatchNorm_train)
        
        fname_model = 'U-%0.2f_P-%03d_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_LR-%0.4f_TR-%02d' %(thresh_rr,pr_temporal_width,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,lr,c_trial)
        

    elif mdl_name=='PRFR_CNN2D_fixed': # freds model
        rgb = os.path.split(path_existing_mdl)[-1]
        mdl_existing = load(os.path.join(path_existing_mdl,rgb))
        # idx_CNN_start = 5
        

        mdl = prfr_cnn2d_fixed(mdl_existing,idx_CNN_start,x, n_cells, filt_temporal_width=temporal_width,
                             chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size,
                             BatchNorm=BatchNorm,MaxPool=MaxPool,BatchNorm_train = BatchNorm_train)
        
        fname_model = 'U-%0.2f_P-%03d_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,pr_temporal_width,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,c_trial)


    else:
        raise ValueError('Wrong model name')
    
    
    path_model_save = os.path.join(path_model_save_base,mdl_name,fname_model)   # the model save directory is the fname_model appened to save path
    
    if not os.path.exists(path_model_save):
        os.makedirs(path_model_save)
    
    path_save_model_performance = os.path.join(path_model_save,'performance')
    if not os.path.exists(path_save_model_performance):
        os.makedirs(path_save_model_performance)
                
    
    fname_excel = 'performance_'+fname_model+'.csv'
     
    # %% Train model
    gbytes_usage = get_model_memory_usage(bz, mdl)  # for PRFR layer models, this is not a good estimate.
    print('Memory required = %0.2f GB' %gbytes_usage)
    # continue a halted training: load existing model checkpoint and initial_epoch value to pass on for continuing the training
    if CONTINUE_TRAINING==1:       
        # glob.glob()
        initial_epoch = len([f for f in os.listdir(path_model_save) if f.endswith('index')])
        if initial_epoch == 0:
            initial_epoch = len([f for f in os.listdir(path_model_save) if f.startswith('weights')])
    else:
        initial_epoch = 0

    print('-----RUNNING MODEL-----')
    validation_batch_size = 100 # samples
    mdl_history = train(mdl, data_train, data_val, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=500,USE_CHUNKER=USE_CHUNKER,initial_epoch=initial_epoch,lr=lr)  
    mdl_history = mdl_history.history
    _ = gc.collect()
    
    # %% Model Evaluation
    nb_epochs = np.max([initial_epoch,nb_epochs])   # number of epochs. Update this variable based on the epoch at which training ended
    val_loss_allEpochs = np.empty(nb_epochs)
    val_loss_allEpochs[:] = np.nan
    fev_medianUnits_allEpochs = np.empty(nb_epochs)
    fev_medianUnits_allEpochs[:] = np.nan
    fev_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    fev_allUnits_allEpochs[:] = np.nan
    fracExVar_medianUnits_allEpochs = np.empty(nb_epochs)
    fracExVar_medianUnits_allEpochs[:] = np.nan
    fracExVar_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    fracExVar_allUnits_allEpochs[:] = np.nan
    
    predCorr_medianUnits_allEpochs = np.empty(nb_epochs)
    predCorr_medianUnits_allEpochs[:] = np.nan
    predCorr_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    predCorr_allUnits_allEpochs[:] = np.nan
    rrCorr_medianUnits_allEpochs = np.empty(nb_epochs)
    rrCorr_medianUnits_allEpochs[:] = np.nan
    rrCorr_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    rrCorr_allUnits_allEpochs[:] = np.nan
    
    try:    # for compatibility with greg's dataset
        obs_rate_allStimTrials = dataset_rr['stim_0']['val']
        obs_noise = None
        num_iters = 10
    except:
        obs_rate_allStimTrials = data_val.y
        obs_noise = data_quality['var_noise']
        num_iters = 1
    
    
        samps_shift = 0 # number of samples to shift the response by. This was to correct some timestamp error in gregs data

    # Check if any stimulus frames from the validation set are present in the training set
    # check_trainVal_contamination(data_train.X,data_val.X,temporal_width)  # commented out because it takes long for my dataset and I did it once while preparing the dataset
    
    obs_rate = data_val.y   # the actual data

    
    print('-----EVALUATING PERFORMANCE-----')
    for i in range(0,nb_epochs):
        # weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % (i+1)
        weight_file = 'weights_'+fname_model+'_epoch-%03d' % (i+1)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
        mdl.load_weights(os.path.join(path_model_save,weight_file))
        gen = chunker(data_val.X,bz,mode='predict') # use generators to generate batches of data
        pred_rate = mdl.predict(gen,steps=int(np.ceil(data_val.X.shape[0]/bz)))
        # pred_rate = mdl.predict(data_val.X)
        _ = gc.collect()
        val_loss = None
        val_loss_allEpochs[i] = val_loss
        
        fev_loop = np.zeros((num_iters,n_cells))
        fracExVar_loop = np.zeros((num_iters,n_cells))
        predCorr_loop = np.zeros((num_iters,n_cells))
        rrCorr_loop = np.zeros((num_iters,n_cells))

        for j in range(num_iters):  # nunm_iters is 1 with my dataset. This was mainly for greg's data where we would randomly split the dataset to calculate performance metrics 
            fev_loop[j,:], fracExVar_loop[j,:], predCorr_loop[j,:], rrCorr_loop[j,:] = model_evaluate_new(obs_rate_allStimTrials,pred_rate,temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
            
        fev = np.mean(fev_loop,axis=0)
        fracExVar = np.mean(fracExVar_loop,axis=0)
        predCorr = np.mean(predCorr_loop,axis=0)
        rrCorr = np.mean(rrCorr_loop,axis=0)
        
        if np.isnan(rrCorr).all():  # if retinal reliability is in quality datasets
            fracExVar = data_quality['fev_allUnits']
            rrCorr = data_quality['dist_cc']


        fev_allUnits_allEpochs[i,:] = fev
        fev_medianUnits_allEpochs[i] = np.nanmedian(fev)      
        fracExVar_allUnits_allEpochs[i,:] = fracExVar
        fracExVar_medianUnits_allEpochs[i] = np.nanmedian(fracExVar)
        
        predCorr_allUnits_allEpochs[i,:] = predCorr
        predCorr_medianUnits_allEpochs[i] = np.nanmedian(predCorr)
        rrCorr_allUnits_allEpochs[i,:] = rrCorr
        rrCorr_medianUnits_allEpochs[i] = np.nanmedian(rrCorr)
        

        _ = gc.collect()
    
    
    idx_bestEpoch = np.nanargmax(fev_medianUnits_allEpochs)
    fev_medianUnits_bestEpoch = np.round(fev_medianUnits_allEpochs[idx_bestEpoch],2)
    fev_allUnits_bestEpoch = fev_allUnits_allEpochs[(idx_bestEpoch),:]
    fracExVar_medianUnits = np.round(fracExVar_medianUnits_allEpochs[idx_bestEpoch],2)
    fracExVar_allUnits = fracExVar_allUnits_allEpochs[(idx_bestEpoch),:]
    
    predCorr_medianUnits_bestEpoch = np.round(predCorr_medianUnits_allEpochs[idx_bestEpoch],2)
    predCorr_allUnits_bestEpoch = predCorr_allUnits_allEpochs[(idx_bestEpoch),:]
    rrCorr_medianUnits = np.round(rrCorr_medianUnits_allEpochs[idx_bestEpoch],2)
    rrCorr_allUnits = rrCorr_allUnits_allEpochs[(idx_bestEpoch),:]

    
    # fname_bestWeight = 'weights_'+fname_model+'_epoch-%03d.h5' % (idx_bestEpoch+1)
    fname_bestWeight = 'weights_'+fname_model+'_epoch-%03d' % (idx_bestEpoch+1)
    mdl.load_weights(os.path.join(path_model_save,fname_bestWeight))
    pred_rate = mdl.predict(gen,steps=int(np.ceil(data_val.X.shape[0]/bz)))
    fname_bestWeight = np.array(fname_bestWeight,dtype='bytes')

    
# %% Save performance
    data_test=data_val

    fname_save_performance = os.path.join(path_save_model_performance,(expDate+'_'+fname_model+'.h5'))

    print('-----SAVING PERFORMANCE STUFF TO H5-----')
    
    
    model_performance = {
        'fev_medianUnits_allEpochs': fev_medianUnits_allEpochs,
        'fev_allUnits_allEpochs': fev_allUnits_allEpochs,
        'fev_medianUnits_bestEpoch': fev_medianUnits_bestEpoch,
        'fev_allUnits_bestEpoch': fev_allUnits_bestEpoch,
        
        'fracExVar_medianUnits': fracExVar_medianUnits,
        'fracExVar_allUnits': fracExVar_allUnits,
        
        'predCorr_medianUnits_allEpochs': predCorr_medianUnits_allEpochs,
        'predCorr_allUnits_allEpochs': predCorr_allUnits_allEpochs,
        'predCorr_medianUnits_bestEpoch': predCorr_medianUnits_bestEpoch,
        'predCorr_allUnits_bestEpoch': predCorr_allUnits_bestEpoch,
        
        'rrCorr_medianUnits': rrCorr_medianUnits,
        'rrCorr_allUnits': rrCorr_allUnits,          
        
        'fname_bestWeight': np.atleast_1d(fname_bestWeight),
        'idx_bestEpoch': idx_bestEpoch,
        
        'val_loss_allEpochs': val_loss_allEpochs,
        # 'val_dataset_name': dataset_rr['stim_0']['dataset_name'],
        }
        

    metaInfo = {
       'mdl_name': mdl.name,
       'existing_mdl': np.array(path_existing_mdl,dtype='bytes'),
       'path_model_save': path_model_save,
       'uname_selectedUnits': np.array(data_quality['uname_selectedUnits'],dtype='bytes'),#[idx_unitsToTake],dtype='bytes'),
       'idx_unitsToTake': idx_unitsToTake,
       'thresh_rr': thresh_rr,
       'trial_num': c_trial,
       'Date': np.array(datetime.datetime.now(),dtype='bytes'),
       'info': np.array(info,dtype='bytes')
       }
        
    model_params = {
                'chan1_n' : chan1_n,
                'filt1_size' : filt1_size,
                'filt1_3rdDim': filt1_3rdDim,
                'chan2_n' : chan2_n,
                'filt2_size' : filt2_size,
                'filt2_3rdDim': filt2_3rdDim,
                'chan3_n' : chan3_n,
                'filt3_size' : filt3_size,
                'filt3_3rdDim': filt3_3rdDim,            
                'bz_ms' : bz_ms,
                'nb_epochs' : nb_epochs,
                'BatchNorm': BatchNorm,
                'MaxPool': MaxPool,
                'pr_temporal_width': pr_temporal_width,
                'lr': lr,
                }
    
    stim_info = {
         'fname_data_train_val_test':fname_data_train_val_test,
         'n_trainingSamps': data_train.X.shape[0],
         'n_valSamps': data_val.X.shape[0],
         'temporal_width':temporal_width,
         'pr_temporal_width': pr_temporal_width
         }
    
    
    datasets_val = {
        'data_val_X': data_val.X,
        'data_val_y': data_val.y,
        'data_test_X': data_test.X,
        'data_test_y': data_test.y,
        }
    
    
    dataset_pred = {
        'obs_rate': obs_rate,
        'pred_rate': pred_rate,
        }

    dataset_rr = None
    save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val,dataset_pred)   # It would really help to have a universal h5 writing function

    
    
# %% Write performance to csv file
    print('-----WRITING TO CSV FILE-----')
    if saveToCSV==1:
        csv_header = ['mdl_name','expDate','thresh_rr','RR','temp_window','batch_size','epochs','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','BatchNorm','MaxPool','c_trial','FEV_median','predCorr_median','rrCorr_median']
        csv_data = [mdl_name,expDate,thresh_rr,fracExVar_medianUnits,temporal_width,bz_ms,nb_epochs,chan1_n, filt1_size, filt1_3rdDim, chan2_n, filt2_size, filt2_3rdDim, chan3_n, filt3_size, filt3_3rdDim,bn_val,mp_val,c_trial,fev_medianUnits_bestEpoch,predCorr_medianUnits_bestEpoch,rrCorr_medianUnits]
        
        fname_csv_file = 'performance_'+expDate+'.csv'
        fname_csv_file = os.path.join(path_save_performance,fname_csv_file)
        if not os.path.exists(fname_csv_file):
            with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header) 
                
        with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_data) 

    fname_validation_excel = os.path.join(path_save_model_performance,expDate+'_validation_'+fname_model+'.csv')
    csv_header = ['epoch','val_fev']
    with open(fname_validation_excel,'w',encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(csv_header) 
        
        for i in range(fev_medianUnits_allEpochs.shape[0]):
            csvwriter.writerow([str(i),str(np.round(fev_medianUnits_allEpochs[i],2))]) 
        
        
    print('-----FINISHED-----')
    return model_performance, mdl

        
if __name__ == "__main__":
    args = parser_run_model()
    # Raw print arguments
    print("Arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))       
    run_model(**vars(args))



