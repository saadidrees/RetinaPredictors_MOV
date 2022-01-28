#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:13:52 2021

@author: saad
"""
import tensorflow as tf
from tensorflow import keras 
import numpy as np
from model.data_handler import load_h5Dataset, save_h5Dataset, rolling_window
from scipy.special import gamma as scipy_gamma
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape, Input
from model.RiekeModel import RiekeModel
from scipy.signal import convolve
import multiprocessing as mp
from joblib import Parallel, delayed
import time, gc

fname_data_train_val_test = '/home/saad/postdoc_db/analyses/data_kiersten/retina1/datasets/8ms/retina1_dataset_train_val_test_photopic.h5'
data_train_orig,data_val_orig,_,_,_,_,_ = load_h5Dataset(fname_data_train_val_test)
stim_orig = data_val_orig.X.copy()
 
# %% model
def model_params(timeBin=1):
    ##  cones - monkey
    params_cones = {}
    params_cones['sigma'] =  float(22) #22  # rhodopsin activity decay rate (1/sec) - default 22
    params_cones['phi'] =  float(22)     # phosphodiesterase activity decay rate (1/sec) - default 22
    params_cones['eta'] =  float(2000)  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_cones['gdark'] =  float(28) #28 # concentration of cGMP in darkness - default 20.5
    params_cones['k'] =  float(0.01)     # constant relating cGMP to current - default 0.02
    params_cones['h'] =  float(3)       # cooperativity for cGMP->current - default 3
    params_cones['cdark'] =  float(1)  # dark calcium concentration - default 1
    params_cones['beta'] = float(9/timeBin) #16 # 9	  # rate constant for calcium removal in 1/sec - default 9
    params_cones['betaSlow'] =  float(0)	  
    params_cones['hillcoef'] =  float(4) #4  	  # cooperativity for cyclase, hill coef - default 4
    params_cones['hillaffinity'] =  float(0.5)   # hill affinity for cyclase - default 0.5
    params_cones['gamma'] =  float(10/timeBin) #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_cones['timeStep'] =  float(1e-3)  # freds default is 1e-4
    params_cones['darkCurrent'] =  float(params_cones['gdark']**params_cones['h'] * params_cones['k']/2)
    return params_cones

def run_model(stim_photons,resp,params,meanIntensity,upSampFac,downSampFac=17,n_discard=0,NORM=1,DOWN_SAMP=1,ROLLING_FAC=30,ode_solver='RungeKutta'):
    
    stim = stim_photons
    stim_spatialDims = stim.shape[1:]
    stim = stim.reshape(stim.shape[0],stim.shape[1]*stim.shape[2])
    
    stim = np.repeat(stim,upSampFac,axis=0)

    idx_allPixels = np.arange(0,stim.shape[1])
    
    t = time.time()

    params['tme'] = np.arange(0,stim.shape[0])*params['timeStep']
    params['biophysFlag'] = 1
    _,stim_currents = RiekeModel(params,stim,ode_solver)
     
        
        
    t_elasped_parallel = time.time()-t
    print('time elasped: '+str(round(t_elasped_parallel))+' seconds')
    
    if DOWN_SAMP == 1:
        stim_currents_downsampled = stim_currents[upSampFac-1::upSampFac]
        
        # rgb = stim_currents.T  
        # rgb[np.isnan(rgb)] = np.nanmedian(rgb)
        
        # rollingFac = ROLLING_FAC
        # a = np.empty((130,rollingFac))
        # a[:] = np.nan
        # a = np.concatenate((a,rgb),axis=1)
        # rgb8 = np.nanmean(rolling_window(a,rollingFac,time_axis = -1),axis=-1)
        # rgb8 = rgb8.reshape(rgb8.shape[0],-1, downSampFac)    
        # rgb8 = rgb8[:,:,0]
                
        # stim_currents_downsampled = rgb8
        # stim_currents_downsampled = stim_currents_downsampled.T
        
    else:
        stim_currents_downsampled = stim_currents
    
    stim_currents_reshaped = stim_currents_downsampled.reshape(stim_currents_downsampled.shape[0],stim_spatialDims[0],stim_spatialDims[1])
    stim_currents_reshaped = stim_currents_reshaped[n_discard:]
    
    if NORM==1:
        stim_currents_norm = (stim_currents_reshaped - np.min(stim_currents_reshaped)) / (np.max(stim_currents_reshaped)-np.min(stim_currents_reshaped))
        stim_currents_norm = stim_currents_norm - np.mean(stim_currents_norm)
    else:
        stim_currents_norm = stim_currents_reshaped
    
    # discard response if n_discard > 0
    if n_discard > 0:
        resp = resp[n_discard:]
        
    return stim_currents_norm,resp

# %% run biophys model

meanIntensity = 10000
timeBin = 1 # ms
frameTime = 8 #ms
stim_spatialDims = stim_orig.shape[1:]
stim = stim_orig.copy()
stim = stim.reshape(stim.shape[0],stim_spatialDims[0],stim_spatialDims[1])
stim[stim>0] = 2*meanIntensity
stim[stim<0] = (2*meanIntensity)/300
stim_photons = stim * 1e-3 * timeBin  # photons per time bin 

params = model_params(timeBin)
params['timeStep'] =  1e-3 * timeBin
upSampFac = int(frameTime/timeBin)
DOWN_SAMP = 1
R_orig,_ = run_model(stim_photons,data_val_orig.y,params,meanIntensity,upSampFac,downSampFac=upSampFac,n_discard=0,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=2)
plt.plot(R_orig[100:,1,1])




# %% prepare and train

x = stim_photons.copy().astype('float32')
idx_train = np.arange(0,x.shape[0])

x_train = x[np.newaxis,idx_train,:,:]
shape_orig = x_train.shape
# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2]*x_train.shape[3])
# inputs = Input(shape=x_train.shape[1:])

R_train = R_orig.copy()
R_train = R_train[np.newaxis,idx_train,:,:]

inputs = Input(x_train.shape[1:])
y = tf.keras.layers.Reshape((x_train.shape[1],x_train.shape[2]*x_train.shape[3]))(inputs)
y = photoreceptor_REIKE(units=1)(y)
y = tf.keras.layers.Reshape(shape_orig[1:])(y)
outputs = y
mdl = tf.keras.Model(inputs,outputs)
mdl.summary()

# mdl.compile(optimizer='sgd',loss='mean_squared_error')
mdl.compile(optimizer=Adam(0.01),loss='poisson')
mdl.fit(x_train,R_train,epochs=1)

# idx_test = np.arange(200,x.shape[0]-200)
x_test = x_train
y_pred = mdl.predict(x_test)
# y_pred = np.squeeze(y_pred)
idx_x = 0
idx_y = 1
plt.plot(y_pred[0,100:,idx_x,idx_y])
# plt.plot(R_train[0,100:,idx_x,idx_y])
plt.show()



# %% keras model

sigma = params['sigma']
phi = params['phi']
eta = params['eta']
cgmp2cur = params['k']
cgmphill = params['h']
cdark = params['cdark']
beta = params['beta']
betaSlow = params['betaSlow']
hillcoef = params['hillcoef']
hillaffinity = params['hillaffinity']
gamma = params['gamma']
timeStep = params['timeStep']
darkCurrent = params['darkCurrent']
gdark = params['gdark']

# outputs = riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark)

@tf.function(autograph=True,experimental_relax_shapes=True)
def riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark):

    darkCurrent = gdark**cgmphill * cgmp2cur/2
    gdark = (2 * darkCurrent / cgmp2cur) **(1/cgmphill)
    
    cur2ca = beta * cdark / darkCurrent                # get q using steady state
    smax = eta/phi * gdark * (1 + (cdark / hillaffinity) **hillcoef)		# get smax using steady state
    
    n_spatialDims = X_fun.shape[-1]
    tme = tf.range(0,X_fun.shape[1],dtype='float32')*TimeStep
    NumPts = tme.shape[0]
    
# initial conditions
    g_prev = gdark+(X_fun[:,0,:]*0)
    s_prev = (gdark * eta/phi)+(X_fun[:,0,:]*0)
    c_prev = cdark+(X_fun[:,0,:]*0)
    cslow_prev = cdark+(X_fun[:,0,:]*0)
    r_prev = X_fun[:,0,:] * gamma / sigma
    p_prev = (eta + r_prev)/phi


    g = tf.TensorArray(tf.float32,size=NumPts)
    g.write(0,X_fun[:,0,:]*0)
    
    # solve difference equations
    for pnt in tf.range(1,NumPts):
        # shape_invariants=[tf.TensorShape(()), tf.TensorShape((None, 15, 15)), dats.shape],
        # tf.autograph.experimental.set_loop_options(shape_invariants=[(g_prev,tf.TensorShape((None,None))),(c_prev,tf.TensorShape((None,None))),(cslow_prev,tf.TensorShape((None,None))),(s_prev,tf.TensorShape((None,None)))])

        # tf.autograph.experimental.set_loop_options(shape_invariants=[(g, tf.TensorShape([None,None,None]))]) #,(g_prev,tf.TensorShape((None,None))),(c_prev,tf.TensorShape((None,None))),(cslow_prev,tf.TensorShape((None,None))),(s_prev,tf.TensorShape((None,None)))])
        r_curr = r_prev + TimeStep * (-1 * sigma * r_prev)
        r_curr = r_curr + gamma * X_fun[:,pnt-1,:]
        p_curr = p_prev + TimeStep * (r_prev + eta - phi * p_prev)
        c_curr = c_prev + TimeStep * (cur2ca * (cgmp2cur * g_prev **cgmphill)/2 - beta * c_prev)
        # cslow_curr = cslow_prev - TimeStep * (betaSlow * (cslow_prev-c_prev))
        s_curr = smax / (1 + (c_curr / hillaffinity) **hillcoef)
        g_curr = g_prev + TimeStep * (s_prev - p_prev * g_prev)

        g = g.write(pnt,g_curr)
        # g = tf.concat((g,g_curr[:,None,:]),axis=1)
        
        
        
        # update prev values to current
        g_prev = g_curr#[0,:]
        s_prev = s_curr#[0,:]
        c_prev = c_curr#[0,:]
        p_prev = p_curr
        r_prev = r_curr
        # cslow_prev = cslow_curr#[0,:]
    
    g = g.stack()
    g = tf.transpose(g,(1,0,2))
    
    # g = g[:,1:,:]
    outputs = -(cgmp2cur * g **cgmphill)/2
    return outputs



class photoreceptor_REIKE(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(photoreceptor_REIKE,self).__init__()
        self.units = units
        
    def build(self,input_shape):
        sigma_init = tf.keras.initializers.Constant(1.) # 22
        self.sigma = tf.Variable(name='sigma',initial_value=sigma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        phi_init = tf.keras.initializers.Constant(1.) #22
        self.phi = tf.Variable(name='phi',initial_value=phi_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        eta_init = tf.keras.initializers.Constant(1.) #2000
        self.eta = tf.Variable(name='eta',initial_value=eta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        beta_init = tf.keras.initializers.Constant(1.) #9
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=True)

        cgmp2cur_init = tf.keras.initializers.Constant(0.01)  # 0.01
        self.cgmp2cur = tf.Variable(name='cgmp2cur',initial_value=cgmp2cur_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cgmphill_init = tf.keras.initializers.Constant(3.)  # 3
        self.cgmphill = tf.Variable(name='cgmphill',initial_value=cgmphill_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cdark_init = tf.keras.initializers.Constant(1.)
        self.cdark = tf.Variable(name='cdark',initial_value=cdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        betaSlow_init = tf.keras.initializers.Constant(0.)
        self.betaSlow = tf.Variable(name='betaSlow',initial_value=betaSlow_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillcoef_init = tf.keras.initializers.Constant(1.) # 4
        self.hillcoef = tf.Variable(name='hillcoef',initial_value=hillcoef_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        hillaffinity_init = tf.keras.initializers.Constant(0.5)
        self.hillaffinity = tf.Variable(name='hillaffinity',initial_value=hillaffinity_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        gamma_init = tf.keras.initializers.Constant(10.)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=False)
                
        gdark_init = tf.keras.initializers.Constant(28.)
        self.gdark = tf.Variable(name='gdark',initial_value=gdark_init(shape=(1,self.units),dtype='float32'),trainable=False)

    def call(self,inputs):
        X_fun = inputs
        
        timeBin = 8 # ms
        frameTime = 8 #
        upSamp_fac = int(frameTime/timeBin)
        
        TimeStep = 1e-3*timeBin
        
        if upSamp_fac>1:
            X_fun = tf.keras.backend.repeat_elements(X_fun,upSamp_fac,axis=1) 
            X_fun = X_fun/upSamp_fac     # appropriate scaling for photons/ms

        sigma = self.sigma * 100
        phi = self.phi * 100
        eta = self.eta * 1000
        cgmp2cur = self.cgmp2cur
        cgmphill = self.cgmphill
        cdark = self.cdark
        beta = self.beta * 10
        betaSlow = self.betaSlow
        hillcoef = self.hillcoef * 10
        hillaffinity = self.hillaffinity
        gamma = self.gamma/timeBin
        gdark = self.gdark * 10
        
        
        outputs = riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark)
        
        if upSamp_fac>1:
            outputs = outputs[:,upSamp_fac-1::upSamp_fac]
            
        return outputs
        






# %% run loop func
# res = run_loop(X_fun,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark)

@tf.function(autograph=True,experimental_relax_shapes=True)
def run_loop(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark):
        
    darkCurrent = gdark**cgmphill * cgmp2cur/2
    gdark = (2 * darkCurrent / cgmp2cur) **(1/cgmphill)
    
    cur2ca = beta * cdark / darkCurrent                # get q using steady state
    smax = eta/phi * gdark * (1 + (cdark / hillaffinity) **hillcoef)		# get smax using steady state
    
    n_spatialDims = X_fun.shape[-1]
    tme = tf.range(0,X_fun.shape[1],dtype='float32')*timeStep
    NumPts = tme.shape[0]
    # TimeStep = 1e-3 #tme[1] - tme[0]
    
    # initial conditions
    
    g_prev = gdark
    s_prev = gdark * eta/phi
    c_prev = cdark
    r_prev = X_fun[:,0,:] * gamma / sigma
    p_prev = (eta + r_prev)/phi
    cslow_prev = cdark
        
    g_prev = tf.repeat(g_prev,n_spatialDims)   
    g_prev = tf.expand_dims(g_prev,axis=0)
    
    s_prev = tf.repeat(s_prev,n_spatialDims)    
    s_prev = tf.expand_dims(s_prev,axis=0)
    
    c_prev = tf.repeat(c_prev,n_spatialDims)
    c_prev = tf.expand_dims(c_prev,axis=0)
    
    cslow_prev = tf.repeat(cslow_prev,n_spatialDims)
    cslow_prev = tf.expand_dims(cslow_prev,axis=0)
    
    # g = g_prev + TimeStep * (s_prev - p_prev * g_prev)
    # g = g[:,None,:]
    
    g = X_fun[:,:1,:]
    
    # solve difference equations
    for pnt in tf.range(1,NumPts):
        # shape_invariants=[tf.TensorShape(()), tf.TensorShape((None, 15, 15)), dats.shape],
        tf.autograph.experimental.set_loop_options(shape_invariants=[(g, tf.TensorShape([None,None,None])),(g_prev,tf.TensorShape((None,None))),(c_prev,tf.TensorShape((None,None))),(cslow_prev,tf.TensorShape((None,None))),(s_prev,tf.TensorShape((None,None)))])
        r_curr = r_prev + TimeStep * (-1 * sigma * r_prev)
        r_curr = r_curr + gamma * X_fun[:,pnt-1,:]
        p_curr = p_prev + TimeStep * (r_prev + eta - phi * p_prev)
        c_curr = c_prev + TimeStep * (cur2ca * (cgmp2cur * g_prev **cgmphill)/2 - beta * c_prev)
        cslow_curr = cslow_prev - TimeStep * (betaSlow * (cslow_prev-c_prev))
        s_curr = smax / (1 + (c_curr / hillaffinity) **hillcoef)
        g_curr = g_prev + TimeStep * (s_prev - p_prev * g_prev)
        
        # if pnt==1:
        #     g = tf.concat((g_curr[:,None,:],g_curr[:,None,:]),axis=1)
        # else:
        g = tf.concat((g,g_curr[:,None,:]),axis=1)
        
        
        # update prev values to current
        g_prev = g_curr#[0,:]
        s_prev = s_curr#[0,:]
        c_prev = c_curr#[0,:]
        p_prev = p_curr
        r_prev = r_curr
        cslow_prev = cslow_curr#[0,:]
    
    # g = g.stack()
    
    outputs = -(cgmp2cur * g **cgmphill)/2
    # outputs = outputs[:,0,:]
    # outputs = tf.expand_dims(outputs,axis=0)
    return outputs




# %% Tensorflow
@tf.function
def tf_reikeModel(params,X,upSamp_fac):
    
    if upSamp_fac>1:
        X_fun = tf.repeat(X,upSamp_fac,axis=0)
    else:
        X_fun = X
    
    sigma = params['sigma']
    phi = params['phi']
    eta = params['eta']
    cgmp2cur = params['k']
    cgmphill = params['h']
    cdark = params['cdark']
    beta = params['beta']
    betaSlow = params['betaSlow']
    hillcoef = params['hillcoef']
    hillaffinity = params['hillaffinity']
    gamma = params['gamma']
    timeStep = params['timeStep']
    darkCurrent = params['darkCurrent']
    

    
    gdark = params['gdark']
    gdark = (2 * darkCurrent / cgmp2cur) **(1/cgmphill)
    
    cur2ca = beta * cdark / darkCurrent                # get q using steady state
    smax = eta/phi * gdark * (1 + (cdark / hillaffinity) **hillcoef)		# get smax using steady state

    n_spatialDims = X_fun.shape[-1]
    tme = tf.range(0,X_fun.shape[0],dtype='float32')*timeStep
    NumPts = tme.shape[0]
    TimeStep = tme[1] - tme[0]

    
    g = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
    g = g.write(0,tf.repeat(gdark,n_spatialDims))
    
    # initial conditions
    g_prev = tf.repeat(gdark,n_spatialDims)
    s_prev = tf.repeat(gdark * eta/phi,n_spatialDims)
    c_prev = tf.repeat(cdark,n_spatialDims)
    r_prev = X_fun[0,:] * gamma / sigma
    p_prev = (eta + r_prev)/phi
    cslow_prev = tf.repeat(cdark,n_spatialDims)
    
    # solve difference equations
    for pnt in tf.range(1,NumPts):
        r_curr = r_prev + TimeStep * (-sigma * r_prev)
        r_curr = r_curr + gamma * X_fun[pnt-1]
        p_curr = p_prev + TimeStep * (r_prev + eta - phi * p_prev)
        c_curr = c_prev + TimeStep * (cur2ca * (cgmp2cur * g_prev **cgmphill)/2 - beta * c_prev)
        cslow_curr = cslow_prev - TimeStep * (betaSlow * (cslow_prev-c_prev))
        s_curr = smax / (1 + (c_curr / hillaffinity) **hillcoef)
        g_curr = g_prev + TimeStep * (s_prev - p_prev * g_prev)
        
        g = g.write(pnt,g_curr)
        
        # update prev values to current
        g_prev = g_curr
        s_prev = s_curr
        c_prev = c_curr
        p_prev = p_curr
        r_prev = r_curr
        cslow_prev = cslow_curr
    
    g = g.stack()
    response = -(cgmp2cur * g **cgmphill)/2
    
    if upSamp_fac>1:
        response = response[upSamp_fac-1::upSamp_fac]
    
    return response
        