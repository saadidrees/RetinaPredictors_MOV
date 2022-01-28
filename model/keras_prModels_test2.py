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
from tensorflow.keras.callbacks import EarlyStopping



fname_data_train_val_test = '/home/saad/postdoc_db/analyses/data_kiersten/retina1/datasets/8ms/retina1_dataset_train_val_test_photopic.h5'
data_train_orig,data_val_orig,_,_,_,_,_ = load_h5Dataset(fname_data_train_val_test)
stim_orig = data_train_orig.X.copy()


num_samps = 10000
meanIntensity = 1
timeBin = 8 # ms
stim_spatialDims = stim_orig.shape[1:]
stim = stim_orig.copy()[100:num_samps+100]
stim = stim.reshape(stim.shape[0],stim_spatialDims[0],stim_spatialDims[1])
stim[stim>0] = 2*meanIntensity
stim[stim<0] = (2*meanIntensity)/300
stim_photons = stim * 1e-3 * timeBin  # photons per time bin 
temporal_width = 240
stim_photons = rolling_window(stim_photons,temporal_width,time_axis=0)

# %%
def generate_simple_filter(tau,n,t):
   f = (t**n)*np.exp(-t/tau); # functional form in paper
   f = (f/tau**(n+1))/scipy_gamma(n+1) # normalize appropriately
   return f

    # # retina 1
    # params_rods = {}
    # params_rods['alpha'] =  1  
    # params_rods['beta'] =  0.36 
    # params_rods['gamma'] =  0.448
    # params_rods['tau_y'] =  22
    # params_rods['n_y'] =  4.33
    # params_rods['tau_z'] =  1000
    # params_rods['n_z'] =  1
    # params_rods['timeStep'] = 1e-3
    # params_rods['tau_r'] = 0 #4.78

alpha =  float(1 / timeBin)
beta = float(0.36 / timeBin)
gamma =  float(0.448)
tau_y =  float(4.48 / timeBin) #4.48 / timeBin
n_y =  float(4.33) 
tau_z =  float(1000 / timeBin)
n_z =  float(1)

t = np.ceil(np.arange(0,1000/timeBin))

Ky = generate_simple_filter(tau_y,n_y,t); # functional form in paper
# plt.plot(Ky)
# Ky_mat = np.tile(Ky,stim_photons.shape[1]*stim_photons.shape[2]).reshape(stim_photons.shape[1],stim_photons.shape[2],Ky.shape[0])
# Ky_mat = np.moveaxis(Ky_mat,-1,0)

Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter(tau_z,n_z,t))
# Kz_mat = np.tile(Kz,stim_photons.shape[1]*stim_photons.shape[2]).reshape(stim_photons.shape[1],stim_photons.shape[2],Kz.shape[0])
# Kz_mat = np.moveaxis(Kz_mat,-1,0)


y_pixLoop = np.zeros(stim_photons.shape)
z_pixLoop = np.zeros(stim_photons.shape)
R_orig = np.zeros(stim_photons.shape)

# for b in range(y_pixLoop.shape[0]):
for i in range(y_pixLoop.shape[-2]):
    for j in range(y_pixLoop.shape[-1]):
        y_pixLoop[:,:,i,j] = lfilter(Ky,1,stim_photons[:,:,i,j])
        z_pixLoop[:,:,i,j] = lfilter(Kz,1,stim_photons[:,:,i,j])
        
        R_orig[:,:,i,j] = (alpha*y_pixLoop[:,:,i,j])/(1+(beta*z_pixLoop[:,:,i,j]))
        
plt.plot(R_orig[100,:,0,0])






# %%
x = stim_photons.copy().astype('float32')
time_samps = np.arange(50,x.shape[1])
idx_train = np.arange(0,x.shape[0]-1000)
idx_test = np.setdiff1d(np.arange(0,x.shape[0]),idx_train)

x_train = x[idx_train,:,:,:]
x_train = x_train[:,time_samps,:,:]
R_train = R_orig.copy()
R_train = R_train[idx_train,:,:,:]
R_train = R_train[:,time_samps,:,:]

x_test = stim_photons.copy().astype('float32')
x_test = x_test[idx_test,:,:,:]
x_test = x_test[:,time_samps,:,:]

R_test = R_orig.copy()
R_test = R_test[idx_test,:,:,:]
R_test = R_test[:,time_samps,:,:]
# R_test[:,0,:,:] = 0

R_test = (R_test - R_test.min())/(R_test.max()-R_test.min())
R_test = R_test-R_test.mean()

# %%
keras_prLayer = photoreceptor_DA(units=1)
mdl = tf.keras.Sequential()
mdl.add(tf.keras.layers.Reshape((x_train.shape[1],x_train.shape[2]*x_train.shape[3])))
mdl.add(keras_prLayer)
# mdl.add(Activation('relu'))
mdl.add(tf.keras.layers.Reshape((x_train.shape[1],x_train.shape[2],x_train.shape[3])))
mdl.add(pr_Normalize(units=1))
# mdl.compile(optimizer='sgd',loss='mean_squared_error')
# mdl.trainable = False
mdl.compile(optimizer=Adam(0.01),loss='mean_squared_error')

mdl.fit(x_train,R_train,epochs=2,batch_size=100,validation_data=(x_test,R_test),callbacks=[EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)])
weights = mdl.get_weights()
print(weights)

# PREDICT

y_pred = mdl.predict(x_test)
# y_pred = np.squeeze(y_pred)
plt.plot(y_pred[200,:,1,0])
plt.plot(R_test[200,:,1,0])
plt.show()

# inputs = Input(shape=x_train.shape[1:])

# %% Keras
@tf.function
def generate_simple_filter(tau,n,t):
   f = (t**n)*tf.math.exp(-t/tau) # functional form in paper
   f = (f/tau**(n+1))/tf.math.exp(tf.math.lgamma(n+1)) # normalize appropriately
   return f

@tf.function
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

class pr_Normalize(keras.layers.Layer):
    def __init__(self,units=1):
        super(pr_Normalize,self).__init__()
        self.units = units
            
    def call(self,inputs):
        value_min = tf.math.reduce_min(inputs)
        value_max = tf.math.reduce_max(inputs)
        R_norm = (inputs - value_min)/(value_max-value_min)
        R_mean = tf.math.reduce_mean(R_norm)
        # v = tf.reshape(R_norm,(1,R_norm.shape[1]*R_norm.shape[2]*R_norm.shape[3]))
        # mid = tf.math.floordiv(v.get_shape()[1],2) + 1
        # R_median = tf.nn.top_k(v, mid).values[-1]
    
        
        R_norm = R_norm - R_mean
        outputs = R_norm
        return outputs



class photoreceptor_DA(keras.layers.Layer):
    def __init__(self,units=1):
        super(photoreceptor_DA,self).__init__()
        self.units = units
            
    def build(self,input_shape):
        alpha_init = tf.keras.initializers.Constant(1.) #tf.random_normal_initializer(mean=1)
        self.alpha = tf.Variable(name='alpha',initial_value=alpha_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        beta_init = tf.keras.initializers.Constant(0.36) #tf.random_normal_initializer(mean=0.36)
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        gamma_init = tf.keras.initializers.Constant(0.448) #tf.random_normal_initializer(mean=0.448)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        # self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        tauY_init = tf.keras.initializers.Constant(4.48) #tf.random_normal_initializer(mean=2) #tf.random_uniform_initializer(minval=1)
        self.tauY = tf.Variable(name='tauY',initial_value=tauY_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        tauZ_init = tf.keras.initializers.Constant(166.) #tf.random_normal_initializer(mean=166) #tf.random_uniform_initializer(minval=100)
        self.tauZ = tf.Variable(name='tauZ',initial_value=tauZ_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        nY_init = tf.keras.initializers.Constant(4.33) #tf.random_normal_initializer(mean=4.33) #tf.random_uniform_initializer(minval=1)
        self.nY = tf.Variable(name='nY',initial_value=nY_init(shape=(1,self.units),dtype='float32'),trainable=True)   
        
        nZ_init = tf.keras.initializers.Constant(1.) #tf.random_uniform_initializer(minval=1)
        self.nZ = tf.Variable(name='nZ',initial_value=nZ_init(shape=(1,self.units),dtype='float32'),trainable=True)
    
    @tf.function
    def call(self,inputs):
       
        timeBin = 8
        
        alpha =  float(self.alpha / timeBin)
        beta = float(self.beta / timeBin)
        # beta = tf.sigmoid(float(self.beta / timeBin))
        gamma =  float(self.gamma)
        # gamma =  tf.sigmoid(float(self.gamma))
        tau_y =  float(self.tauY / timeBin)
        tau_z =  float(self.tauZ / timeBin)
        n_y =  float(self.nY)
        n_z =  float(self.nZ)
        
        t = tf.range(0,1000/timeBin,dtype='float32')
        
        Ky = generate_simple_filter(tau_y,n_y,t)   
        Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter(tau_z,n_z,t))
       
        y_tf = conv_oper(inputs,Ky)
        z_tf = conv_oper(inputs,Kz)
    
        outputs = (alpha*y_tf)/(1+(beta*z_tf))
        
        
        return outputs


# print(model.predict([10.0]))
# print(my_layer.variables)




