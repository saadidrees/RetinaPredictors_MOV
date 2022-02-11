#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:42:39 2021

@author: saad
"""


# from keras.models import Model, Sequential
# from keras.layers import Dense, Activation, Flatten, Reshape, ConvLSTM2D, LSTM, TimeDistributed, MaxPool3D, MaxPool2D, concatenate, Permute, AveragePooling2D, AveragePooling3D
# from tensorflow.keras.layers import Conv2D, Conv3D
# # from keras.layers.convolutional import Conv2D, Conv3D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.noise import GaussianNoise
# from keras.regularizers import l1, l2
# import numpy as np
# import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape, ConvLSTM2D, LSTM, TimeDistributed, MaxPool3D, MaxPool2D, Concatenate, Permute, AveragePooling2D, AveragePooling3D
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import BatchNormalization, Dropout, concatenate, Permute
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.regularizers import l1, l2
import numpy as np

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

def model_definitions():
    models_2D = ('CNN_2D','PRFR_CNN2D','PRFR_CNN2D_MULTIPR','PRFR_CNN2D_fixed','PR_CNN2D','PR_CNN2D_fixed','PR_CNN2D_MULTIPR')
    models_3D = ('CNN_3D','PR_CNN3D')
    
    return (models_2D,models_3D)

# Some functions used in custom Keras models
def generate_simple_filter(tau,n,t):
   f = (t**n)*tf.math.exp(-t/tau) # functional form in paper
   f = (f/tau**(n+1))/tf.math.exp(tf.math.lgamma(n+1)) # normalize appropriately
   return f

def conv_oper(x,kernel_1D):
    spatial_dims = x.shape[-1]
    x_reshaped = tf.expand_dims(x,axis=2)
    kernel_1D = tf.squeeze(kernel_1D)
    kernel_1D = tf.reverse(kernel_1D,[0])
    tile_fac = tf.constant([spatial_dims])
    kernel_reshaped = tf.tile(kernel_1D,tile_fac)
    kernel_reshaped = tf.reshape(kernel_reshaped,(1,spatial_dims,1,kernel_1D.shape[-1]))
    kernel_reshaped = tf.experimental.numpy.moveaxis(kernel_reshaped,-1,0)
    pad_vec = [[0,0],[kernel_1D.shape[-1]-1,0],[0,0],[0,0]]
    conv_output = tf.nn.depthwise_conv2d(x_reshaped,kernel_reshaped,strides=[1,1,1,1],padding=pad_vec,data_format='NHWC')
    return conv_output

    
class Normalize(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(Normalize,self).__init__()
        self.units = units
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "units": self.units,
         })
         return config   
             
    def call(self,inputs):
        value_min = tf.math.reduce_min(inputs)
        value_max = tf.math.reduce_max(inputs)
        R_norm = (inputs - value_min)/(value_max-value_min)
        R_mean = tf.math.reduce_mean(R_norm)       
        R_norm = R_norm - R_mean
        return R_norm


class photoreceptor_DA(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(photoreceptor_DA,self).__init__()
        self.units = units
            
    def build(self,input_shape):
        alpha_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(16.2) #tf.keras.initializers.Constant(1.) #tf.random_normal_initializer(mean=1)
        self.alpha = tf.Variable(name='alpha',initial_value=alpha_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        beta_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(-13.46) #tf.keras.initializers.Constant(0.36) #tf.random_normal_initializer(mean=0.36)
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        gamma_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(16.49) #tf.keras.initializers.Constant(0.448) #tf.random_normal_initializer(mean=0.448)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        # self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        tauY_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(0.928) #tf.keras.initializers.Constant(10.) #tf.random_normal_initializer(mean=2) #tf.random_uniform_initializer(minval=1)
        self.tauY = tf.Variable(name='tauY',initial_value=tauY_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        tauZ_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(0.008) #tf.keras.initializers.Constant(166) #tf.random_normal_initializer(mean=166) #tf.random_uniform_initializer(minval=100)
        self.tauZ = tf.Variable(name='tauZ',initial_value=tauZ_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        nY_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(1.439) #tf.keras.initializers.Constant(4.33) #tf.random_normal_initializer(mean=4.33) #tf.random_uniform_initializer(minval=1)
        self.nY = tf.Variable(name='nY',initial_value=nY_init(shape=(1,self.units),dtype='float32'),trainable=True)   
        
        nZ_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(0.29) #tf.keras.initializers.Constant(1) #tf.random_uniform_initializer(minval=1)
        self.nZ = tf.Variable(name='nZ',initial_value=nZ_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        
        tauY_mulFac = tf.keras.initializers.Constant(10.) #tf.keras.initializers.Constant(10.) 
        self.tauY_mulFac = tf.Variable(name='tauY_mulFac',initial_value=tauY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        tauZ_mulFac = tf.keras.initializers.Constant(10.) #tf.keras.initializers.Constant(10.) 
        self.tauZ_mulFac = tf.Variable(name='tauZ_mulFac',initial_value=tauZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
        nY_mulFac = tf.keras.initializers.Constant(10.) #tf.keras.initializers.Constant(10.) 
        self.nY_mulFac = tf.Variable(name='nY_mulFac',initial_value=nY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
        nZ_mulFac = tf.keras.initializers.Constant(10.) #tf.keras.initializers.Constant(10.) 
        self.nZ_mulFac = tf.Variable(name='nZ_mulFac',initial_value=nZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
    
        
    def call(self,inputs):
       
        timeBin = 8
        
        alpha =  self.alpha / timeBin
        beta = self.beta / timeBin
        # beta = tf.sigmoid(float(self.beta / timeBin))
        gamma =  self.gamma
        # gamma =  tf.sigmoid(float(self.gamma))
        tau_y =  (self.tauY_mulFac*self.tauY) / timeBin
        tau_z =  (self.tauZ_mulFac*self.tauZ) / timeBin
        n_y =  (self.nY_mulFac*self.nY)
        n_z =  (self.nZ_mulFac*self.nZ)
        
        t = tf.range(0,1000/timeBin,dtype='float32')
        
        Ky = generate_simple_filter(tau_y,n_y,t)   
        Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter(tau_z,n_z,t))
       
        y_tf = conv_oper(inputs,Ky)
        z_tf = conv_oper(inputs,Kz)
    
        outputs = (alpha*y_tf)/(1+(beta*z_tf))
        
        return outputs


class Normalize_PRDA(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(Normalize_PRDA,self).__init__()
        self.units = units
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "units": self.units,
         })
         return config   
             
    def call(self,inputs):
        value_min = 0.004992888155901156 #tf.math.reduce_min(inputs)
        value_max = 0.02672805318583508 #tf.math.reduce_max(inputs)
        R_norm = (inputs - value_min)/(value_max-value_min)
        R_mean = 0.5233899120505345 #tf.math.reduce_mean(R_norm)       
        R_norm = R_norm - R_mean
        return R_norm



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
        r_curr = r_prev + TimeStep * (-1 * sigma * r_prev)
        r_curr = r_curr + gamma * X_fun[:,pnt-1,:]
        p_curr = p_prev + TimeStep * (r_prev + eta - phi * p_prev)
        c_curr = c_prev + TimeStep * (cur2ca * (cgmp2cur * g_prev **cgmphill)/2 - beta * c_prev)
        # c_curr = c_prev + TimeStep * (cur2ca * cgmp2cur * g_prev**cgmphill /(1+(cslow_prev/cdark)) - beta * c_prev)
        # cslow_curr = cslow_prev - TimeStep * (betaSlow * (cslow_prev-c_prev))
        s_curr = smax / (1 + (c_curr / hillaffinity) **hillcoef)
        g_curr = g_prev + TimeStep * (s_prev - p_prev * g_prev)

        g = g.write(pnt,g_curr)
        
        
        # update prev values to current
        g_prev = g_curr#[0,:]
        s_prev = s_curr#[0,:]
        c_prev = c_curr#[0,:]
        p_prev = p_curr
        r_prev = r_curr
        # cslow_prev = cslow_curr#[0,:]
    
    g = g.stack()
    g = tf.transpose(g,(1,0,2))
    outputs = -(cgmp2cur * g **cgmphill)/2
    
    return outputs
 
class photoreceptor_REIKE(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(photoreceptor_REIKE,self).__init__()
        self.units = units
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

    def build(self,input_shape):
        sigma_init = tf.keras.initializers.Constant(1.) # 22
        self.sigma = tf.Variable(name='sigma',initial_value=sigma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        sigma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.sigma_scaleFac = tf.Variable(name='sigma_scaleFac',initial_value=sigma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        phi_init = tf.keras.initializers.Constant(1.) #22
        self.phi = tf.Variable(name='phi',initial_value=phi_init(shape=(1,self.units),dtype='float32'),trainable=True)
        phi_scaleFac = tf.keras.initializers.Constant(10.) 
        self.phi_scaleFac = tf.Variable(name='phi_scaleFac',initial_value=phi_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
       
        eta_init = tf.keras.initializers.Constant(1.) #2000
        self.eta = tf.Variable(name='eta',initial_value=eta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        eta_scaleFac = tf.keras.initializers.Constant(1000.) 
        self.eta_scaleFac = tf.Variable(name='eta_scaleFac',initial_value=eta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        beta_init = tf.keras.initializers.Constant(1.) #9
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        beta_scaleFac = tf.keras.initializers.Constant(10.) 
        self.beta_scaleFac = tf.Variable(name='beta_scaleFac',initial_value=beta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        cgmp2cur_init = tf.keras.initializers.Constant(0.01) # 0.01
        self.cgmp2cur = tf.Variable(name='cgmp2cur',initial_value=cgmp2cur_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cgmphill_init = tf.keras.initializers.Constant(3.)  # 3
        self.cgmphill = tf.Variable(name='cgmphill',initial_value=cgmphill_init(shape=(1,self.units),dtype='float32'),trainable=False)
        cgmphill_scaleFac = tf.keras.initializers.Constant(1.) 
        self.cgmphill_scaleFac = tf.Variable(name='cgmphill_scaleFac',initial_value=cgmphill_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cdark_init = tf.keras.initializers.Constant(1.)
        self.cdark = tf.Variable(name='cdark',initial_value=cdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        betaSlow_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(1.) # 0
        self.betaSlow = tf.Variable(name='betaSlow',initial_value=betaSlow_init(shape=(1,self.units),dtype='float32'),trainable=False)
        betaSlow_scaleFac = tf.keras.initializers.Constant(1.) 
        self.betaSlow_scaleFac = tf.Variable(name='betaSlow_scaleFac',initial_value=betaSlow_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillcoef_init = tf.keras.initializers.Constant(4.) #tf.keras.initializers.Constant(1.) # 4
        self.hillcoef = tf.Variable(name='hillcoef',initial_value=hillcoef_init(shape=(1,self.units),dtype='float32'),trainable=True)
        hillcoef_scaleFac = tf.keras.initializers.Constant(1.) 
        self.hillcoef_scaleFac = tf.Variable(name='hillcoef_scaleFac',initial_value=hillcoef_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillaffinity_init = tf.keras.initializers.Constant(1.) # 0.5
        self.hillaffinity = tf.Variable(name='hillaffinity',initial_value=hillaffinity_init(shape=(1,self.units),dtype='float32'),trainable=True)
        hillaffinity_scaleFac = tf.keras.initializers.Constant(1.) 
        self.hillaffinity_scaleFac = tf.Variable(name='hillaffinity_scaleFac',initial_value=hillaffinity_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        gamma_init = tf.keras.initializers.Constant(1.)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=False)
        gamma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.gamma_scaleFac = tf.Variable(name='gamma_scaleFac',initial_value=gamma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        gdark_init = tf.keras.initializers.Constant(0.28)    # 28 for cones; 20 for rods 
        self.gdark = tf.Variable(name='gdark',initial_value=gdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        self.timeBin = 8 # find a way to fix this in the model  #tf.Variable(name='timeBin',initial_value=timeBin(shape=(1,self.units),dtype='float32'),trainable=False)


    def call(self,inputs):
        X_fun = inputs

        timeBin = float(self.timeBin) # ms
        frameTime = 8 # ms
        upSamp_fac = int(frameTime/timeBin)
        TimeStep = 1e-3*timeBin
        
        if upSamp_fac>1:
            X_fun = tf.keras.backend.repeat_elements(X_fun,upSamp_fac,axis=1) 
            X_fun = X_fun/upSamp_fac     # appropriate scaling for photons/ms

        sigma = self.sigma * self.sigma_scaleFac
        phi = self.phi * self.phi_scaleFac
        eta = self.eta * self.eta_scaleFac
        cgmp2cur = self.cgmp2cur
        cgmphill = self.cgmphill * self.cgmphill_scaleFac
        cdark = self.cdark
        beta = self.beta * self.beta_scaleFac
        betaSlow = self.betaSlow * self.betaSlow_scaleFac
        hillcoef = self.hillcoef * self.hillcoef_scaleFac
        hillaffinity = self.hillaffinity * self.hillaffinity_scaleFac
        gamma = (self.gamma*self.gamma_scaleFac)/timeBin
        gdark = self.gdark*100
        
        
        outputs = riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark)
        
        if upSamp_fac>1:
            outputs = outputs[:,upSamp_fac-1::upSamp_fac]
            
        return outputs

class Normalize_PRFR(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(Normalize_PRFR,self).__init__()
        self.units = units
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "units": self.units,
         })
         return config   
             
    def call(self,inputs):
        value_min = -103 #tf.math.reduce_min(inputs)
        value_max = -87 #tf.math.reduce_max(inputs)
        R_norm = (inputs - value_min)/(value_max-value_min)
        R_mean = 0.54 #tf.math.reduce_mean(R_norm)       
        R_norm = R_norm - R_mean
        return R_norm

        
def prfr_cnn2d(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):

    BatchNorm = bool(BatchNorm)
    MaxPool = bool(MaxPool)
    
    sigma = 0.1
    
    # keras_prLayer = photoreceptor_REIKE(units=1)
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_REIKE(units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y = Normalize_PRFR(units=1)(y)
    
    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    if BatchNorm is True:
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        # y = BatchNormalization(axis=1)(y)   
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
            
        # if MaxPool is True:
        #     y = MaxPool2D(2,data_format='channels_first')(y)

        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PRFR_CNN2D'
    return Model(inputs, outputs, name=mdl_name)

def prfr_cnn2d_multipr(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):

    BatchNorm = bool(BatchNorm)
    MaxPool = bool(MaxPool)
    
    sigma = 0.1
    
    # PR Channel 1
    y1 = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y1 = photoreceptor_REIKE(units=1)(y1)
    y1 = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y1)
    y1 = y1[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y1 = Normalize(units=1)(y1)
    y1 = tf.keras.backend.expand_dims(y1,axis=-1)
    
    y2 = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y2 = photoreceptor_REIKE(units=1)(y2)
    y2 = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y2)
    y2 = y2[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y2 = Normalize(units=1)(y2)
    y2 = tf.keras.backend.expand_dims(y2,axis=-1)
    
    y = tf.keras.layers.concatenate((y1,y2), axis=-1)
    y = Permute((4,2,3,1))(y)

   
    
    # CNN - first layer
    y = Conv3D(chan1_n, (filt1_size,filt1_size,filt_temporal_width), data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    y = tf.keras.backend.squeeze(y,-1)
    if BatchNorm is True:
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        # y = BatchNormalization(axis=1)(y)   
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
            
        # if MaxPool is True:
        #     y = MaxPool2D(2,data_format='channels_first')(y)

        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PRFR_CNN2D_MULTIPR'
    return Model(inputs, outputs, name=mdl_name)


def prfr_cnn2d_fixed(mdl_existing,idx_CNN_start,inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):

    BatchNorm = bool(BatchNorm)
    MaxPool = bool(MaxPool)
    mdl_name = 'PRFR_CNN2D_fixed'
    
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]),name='Reshape_1_pr')(inputs)
    y = photoreceptor_REIKE(units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]),name='Reshape_2_pr')(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y = Normalize(units=1)(y)
       
    for layer in mdl_existing.layers[idx_CNN_start:]:
        layer.trainable = False
        y = layer(y)
    
    outputs = y
    
    return Model(inputs, outputs, name=mdl_name)

def prfr_cnn2d_noTime(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    BatchNorm = bool(BatchNorm)
    MaxPool = bool(MaxPool)
    sigma = 0.1
    
    # keras_prLayer = photoreceptor_REIKE(units=1)
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_REIKE(units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,-1,:,:]
    y = y[:,None,:,:]
    
    y = Normalize(units=1)(y)
    
    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    if BatchNorm is True:
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        # y = BatchNormalization(axis=1)(y)   
    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'PRFR_CNN2D_NOTIME'
    return Model(inputs, outputs, name=mdl_name)


def pr_cnn2d(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
        
    BatchNorm = bool(BatchNorm)
    MaxPool = bool(MaxPool)
    sigma = 0.1
    
    keras_prLayer = photoreceptor_DA(units=1)
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = keras_prLayer(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    
    y = Normalize(units=1)(y)
    
    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    if BatchNorm is True:
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        # y = BatchNormalization(axis=1)(y)   
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'PR_CNN2D'
    return Model(inputs, outputs, name=mdl_name)


def pr_cnn2d_multipr(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):

    BatchNorm = bool(BatchNorm)
    MaxPool = bool(MaxPool)
    
    sigma = 0.1
    
    # PR Channel 1
    y1 = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y1 = photoreceptor_DA(units=1)(y1)
    y1 = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y1)
    y1 = y1[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y1 = Normalize_PRDA(units=1)(y1)
    y1 = tf.keras.backend.expand_dims(y1,axis=-1)
    
    y2 = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y2 = photoreceptor_DA(units=1)(y2)
    y2 = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y2)
    y2 = y2[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y2 = Normalize_PRDA(units=1)(y2)
    y2 = tf.keras.backend.expand_dims(y2,axis=-1)
    
    y = tf.keras.layers.concatenate((y1,y2), axis=-1)
    y = Permute((4,2,3,1))(y)

   
    
    # CNN - first layer
    y = Conv3D(chan1_n, (filt1_size,filt1_size,filt_temporal_width), data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    y = tf.keras.backend.squeeze(y,-1)
    if BatchNorm is True:
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        # y = BatchNormalization(axis=1)(y)   
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
            
        # if MaxPool is True:
        #     y = MaxPool2D(2,data_format='channels_first')(y)

        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PR_CNN2D_MULTIPR'
    return Model(inputs, outputs, name=mdl_name)




def pr_cnn2d_fixed(mdl_existing,idx_CNN_start,inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    BatchNorm = bool(BatchNorm)
    MaxPool = bool(MaxPool)
    mdl_name = 'PR_CNN2D_fixed'
    
    keras_prLayer = photoreceptor_DA(units=1)
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]),name='Reshape_1_pr')(inputs)
    y = keras_prLayer(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]),name='Reshape_2_pr')(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
       
    # y = BatchNormalization(axis=1,trainable=False,name='postPR')(y)
    y = Normalize(units=1)(y)
    for layer in mdl_existing.layers[idx_CNN_start:]:
        layer.trainable = False
        y = layer(y)
    
    outputs = y
    
    return Model(inputs, outputs, name=mdl_name)

def pr_cnn3d(inputs, n_out, filt_temporal_width=120, chan1_n=12, filt1_size=13, filt1_3rdDim=1, chan2_n=25, filt2_size=13, filt2_3rdDim=1, chan3_n=25, filt3_size=13, filt3_3rdDim=1, BatchNorm=True,BatchNorm_train=False,MaxPool=True):
        
    sigma = 0.1
    
    y = Permute((4,2,3,1))(inputs)
    y = Reshape((y.shape[1],y.shape[-3]*y.shape[-2]))(y)
    y = photoreceptor_DA(units=1)(y)
    y = Reshape((inputs.shape[-1],inputs.shape[-3],inputs.shape[-2]))(y)
    y = y[:,y.shape[1]-filt_temporal_width:,:,:]
    y = Reshape((-1,y.shape[1],y.shape[2],y.shape[3]))(y)
    y = Permute((1,3,4,2))(y)
    
    y = Normalize(units=1)(y)
    
    # CNN - first layer
    y = Conv3D(chan1_n,  (filt1_size,filt1_size,filt1_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    if BatchNorm is True:
        y = Reshape((y.shape[1],y.shape[2], y.shape[3],y.shape[4]))(BatchNormalization(axis=-1)(Flatten()(y)))
    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv3D(chan2_n, (filt2_size,filt2_size,filt2_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            y = Reshape((y.shape[1],y.shape[2], y.shape[3],y.shape[4]))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # CNN - third layer
    if chan3_n>0:
        y = Conv3D(chan3_n, (filt3_size,filt3_size,filt3_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            y = Reshape((y.shape[1],y.shape[2], y.shape[3],y.shape[4]))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'PR_CNN3D'
    return Model(inputs, outputs, name=mdl_name)


def cnn_2d(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    BatchNorm = bool(BatchNorm)
    MaxPool = bool(MaxPool)
    
    sigma = 0.1
    filt_temporal_width=inputs.shape[1]

    # first layer  
    if BatchNorm is True:
        y = inputs
        n1 = int(inputs.shape[-1])
        n2 = int(inputs.shape[-2])
        y = Reshape((filt_temporal_width, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    else:
        y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(inputs)
    
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        if BatchNorm is True: 
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)                  

        y = Activation('relu')(GaussianNoise(sigma)(y))

    # Third layer
    if chan3_n>0:
        if BatchNorm is True: 
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))

        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)       
        y = Activation('relu')(GaussianNoise(sigma)(y))
        
        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'CNN_2D'
    return Model(inputs, outputs, name=mdl_name)


def cnn_3d(inputs, n_out, chan1_n=12, filt1_size=13, filt1_3rdDim=1, chan2_n=25, filt2_size=13, filt2_3rdDim=1, chan3_n=25, filt3_size=13, filt3_3rdDim=1, BatchNorm=True,MaxPool=True):
    BatchNorm = bool(BatchNorm)
    MaxPool = bool(MaxPool)
    
    sigma = 0.1
    filt_temporal_width=inputs.shape[-1]

    # first layer  
    n1 = int(inputs.shape[-2])
    n2 = int(inputs.shape[-3])
    y = Reshape((inputs.shape[1],n2, n1,filt_temporal_width))(BatchNormalization(axis=-1)(Flatten()(inputs)))
    # with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
    y = Conv3D(chan1_n, (filt1_size,filt1_size,filt1_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    if MaxPool:
        y = MaxPool3D(2,data_format='channels_first')(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        n1 = int(y.shape[-2])
        n2 = int(y.shape[-3])
        y = Reshape((y.shape[1],y.shape[2], y.shape[3],y.shape[4]))(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Conv3D(chan2_n, (filt2_size,filt2_size,filt2_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        # y = MaxPool3D(2,data_format='channels_first')(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))
       
    # Third layer
    if chan3_n>0:
        y = Reshape((y.shape[1],y.shape[2], y.shape[3],y.shape[4]))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Conv3D(chan3_n, (filt3_size,filt3_size,filt3_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        # y = MaxPool3D(2,data_format='channels_first')(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Dense layer
    y = Flatten()(y)
    y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)
    
    mdl_name = 'CNN_3D'
    return Model(inputs, outputs, name=mdl_name)

def get_model_memory_usage(batch_size, model):

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem
    
    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    
    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0
    
    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    
    return gbytes
