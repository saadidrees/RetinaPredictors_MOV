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
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape




fname_data_train_val_test = '/home/saad/postdoc_db/analyses/data_kiersten/retina1/datasets/8ms/retina1_dataset_train_val_test_photopic.h5'
_,data_val_orig,_,_,_,_,_ = load_h5Dataset(fname_data_train_val_test)
stim_orig = data_val_orig.X.copy()



meanIntensity = 10000
timeBin = 8 # ms
stim_spatialDims = stim_orig.shape[1:]
stim = stim_orig.copy()
stim = stim.reshape(stim.shape[0],stim_spatialDims[0],stim_spatialDims[1])
stim[stim>0] = 2*meanIntensity
stim[stim<0] = (2*meanIntensity)/300
stim_photons = stim * 1e-3 * timeBin  # photons per time bin 

    # params_cones['alpha'] =  1.27 #1.997 #1 
    # params_cones['beta'] = -0.009 #-0.0026 #0.36
    # params_cones['gamma'] =  1.66 #2.466 #0.448
    # params_cones['tau_y'] =  4.12 #5.83 #4.48
    # params_cones['tau_z'] =  2.35 #3.22 #166
    # params_cones['n_y'] =  6.15 #5.8 #4.33   
    # params_cones['n_z'] =  13.2 #13.2 #1
    # params_cones['timeStep'] = 1e-3
    # params_cones['tau_r'] = 0#100 #4.78

# %%
def generate_simple_filter(tau,n,t):
   f = (t**n)*np.exp(-t/tau); # functional form in paper
   f = (f/tau**(n+1))/scipy_gamma(n+1) # normalize appropriately
   return f


alpha =  1.27 / timeBin
beta = -0.009 / timeBin
gamma =  1.66
tau_y =  4.12 / timeBin
n_y =  6.15
tau_z =  2.35 / timeBin
n_z =  13.2

t = np.ceil(np.arange(0,1000/timeBin))

Ky = generate_simple_filter(tau_y,n_y,t); # functional form in paper
# Ky_mat = np.tile(Ky,stim_photons.shape[1]*stim_photons.shape[2]).reshape(stim_photons.shape[1],stim_photons.shape[2],Ky.shape[0])
# Ky_mat = np.moveaxis(Ky_mat,-1,0)

Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter(tau_z,n_z,t))
# Kz_mat = np.tile(Kz,stim_photons.shape[1]*stim_photons.shape[2]).reshape(stim_photons.shape[1],stim_photons.shape[2],Kz.shape[0])
# Kz_mat = np.moveaxis(Kz_mat,-1,0)


y_pixLoop = np.zeros(stim_photons.shape)
z_pixLoop = np.zeros(stim_photons.shape)
R_orig = np.zeros(stim_photons.shape)

for i in range(y_pixLoop.shape[1]):
    for j in range(y_pixLoop.shape[2]):
        y_pixLoop[:,i,j] = lfilter(Ky,1,stim_photons[:,i,j])
        z_pixLoop[:,i,j] = lfilter(Kz,1,stim_photons[:,i,j])
        
        R_orig[:,i,j] = (alpha*y_pixLoop[:,i,j])/(1+(beta*z_pixLoop[:,i,j]))
        
plt.plot(R_orig[:,0,0])

# %% Tensorflow
@tf.function   
def generate_simple_filter_tf(tau,n,t):
   f = (t**n)*tf.math.exp(-t/tau) # functional form in paper
   f = (f/tau**(n+1))/tf.math.exp(tf.math.lgamma(n+1)) # normalize appropriately
   return f

@tf.function 
def get_kernelTensor_tf(kernel_1D,spatial_dims):
    tile_fac = tf.constant([spatial_dims[0]*spatial_dims[1]])
    kernel_tensor = tf.tile(kernel_1D,tile_fac)
    kernel_tensor = tf.reshape(kernel_tensor,(spatial_dims[0],spatial_dims[1],kernel_1D.shape[0]))
    kernel_tensor = tf.experimental.numpy.moveaxis(kernel_tensor,-1,0)
    return kernel_tensor

@tf.function 
def conv_oper_tf(x,kernel_1D):
    x_reshaped = tf.reshape(x,(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))
    x_reshaped = tf.expand_dims(x_reshaped,axis=2)

    tile_fac = tf.constant([130])
    kernel_reshaped = tf.tile(kernel_1D,tile_fac)
    kernel_reshaped = tf.reshape(kernel_reshaped,(1,130,1,kernel_1D.shape[-1]))
    kernel_reshaped = tf.experimental.numpy.moveaxis(kernel_reshaped,-1,0)
    kernel_flipped = tf.reverse(kernel_reshaped,[0])
    pad_vec = [[0,0],[kernel_1D.shape[-1]-1,0],[0,0],[0,0]]
    conv_output = tf.nn.depthwise_conv2d(x_reshaped,kernel_flipped,strides=[1,1,1,1],padding=pad_vec,data_format='NHWC')
    return conv_output

@tf.function 
def conv_oper_old(x,kernel_1D):
    
    PAD = True
       
    kernel_tensor = get_kernelTensor_tf(kernel_1D,[x.shape[1],x.shape[2]])
    kernel_flipped = tf.reverse(kernel_tensor,[0])
    
    conv_output = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
    kern_size = kernel_tensor.shape[0]
    
    if PAD==True:
        pad_length = kernel_tensor.shape[0]*1
        paddings = tf.constant([[pad_length,0],[0,0],[0,0]])
        x_padded = tf.pad(x,paddings)

        for i in tf.range(x_padded.shape[0]-kern_size):
            rgb = x_padded[i:i+kern_size]
            rgb = rgb*kernel_flipped
            rgb = tf.reduce_sum(rgb,0)
            conv_output = conv_output.write(i,rgb)
        conv_output = conv_output.stack()
        
    # else:
    #     cnt = -1
    #     for i in tf.range(kern_size,x.shape[0]):
    #         cnt += 1
    #         rgb = x[i-kern_size:i]
    #         rgb = rgb*kernel_flipped
    #         rgb = tf.reduce_sum(rgb,0)
    #         # rgb = rgb[pad_length-kernel_tensor.shape[0]:]
    #         conv_output = conv_output.write(cnt,rgb)
    #     conv_output = conv_output.stack()
    # # conv_output = conv_output[pad_length:]
    return conv_output

@tf.function 
def conv_pr_tf(x,opt='conv'):

    timeBin = 8
    alpha =  float(1.27 / timeBin)
    beta = float(-0.009 / timeBin)
    gamma =  float(1.66)
    tau_y =  float(4.12 / timeBin)
    tau_z =  float(2.35 / timeBin)
    n_y =  float(6.15)   
    n_z =  float(13.2)
    
    spatial_dims = [x.shape[1],x.shape[2]]

    t = tf.range(0,1000/timeBin,dtype='float32')
    
    Ky = generate_simple_filter_tf(tau_y,n_y,t)
    # Ky_tens = get_kernelTensor_tf(Ky,spatial_dims)

    Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter_tf(tau_z,n_z,t))
    # Kz_tens = get_kernelTensor_tf(Kz,spatial_dims)
   
    if opt == 'conv':
        x = x[np.newaxis,:,:,:]
        y_tf = conv_oper_tf(x,Ky)
        z_tf = conv_oper_tf(x,Kz)
        
    elif opt =='old':
        y_tf = conv_oper_old(x,Ky)
        z_tf = conv_oper_old(x,Kz)

    
    R = alpha*y_tf/(1+(beta*z_tf))
    
    value_min = tf.math.reduce_min(R[:,200:])
    value_max = tf.math.reduce_max(R[:,200:])
    R_norm = (R - value_min)/(value_max-value_min)
    
    v = tf.reshape(R_norm, [-1])
    mid = v.get_shape()[0]//2 + 1
    # mid = tf.math.floordiv(v.get_shape()[0],2) + 1
    R_median = tf.nn.top_k(v, mid).values[-1]
    R_mean = tf.math.reduce_mean(R_norm)

    
    R_norm = R_norm - R_median

    
    return R_norm


x = stim_photons.copy().astype('float32')
opt = 'conv' #'conv'
R = np.array(conv_pr_tf(x,opt))
R = np.squeeze(R)
R = R.reshape(R.shape[0],10,13)

shift = 0
# plt.plot(R_orig[:,0,0])
plt.plot(R[100:,0,0])
plt.show()

# %%
x = stim_photons.copy().astype('float32')
idx_train = np.arange(100,x.shape[0])

x_train = x[np.newaxis,idx_train,:,:]
R_train = R_orig.copy()
R_train = R_train[np.newaxis,idx_train,:,:]

keras_prLayer = photoreceptor_DA(units=1)
mdl = tf.keras.Sequential()
mdl.add(tf.keras.layers.Reshape((x_train.shape[1],x_train.shape[2]*x_train.shape[3])))
mdl.add(keras_prLayer)
mdl.add(tf.keras.layers.Reshape((x_train.shape[1],x_train.shape[2],x_train.shape[3])))
# mdl.compile(optimizer='sgd',loss='mean_squared_error')
mdl.compile(optimizer=Adam(0.01),loss='poisson')
mdl.fit(x_train,R_train,epochs=500)

# idx_test = np.arange(200,x.shape[0]-200)
x_test = x_train
y_pred = mdl.predict(x_test)
# y_pred = np.squeeze(y_pred)
plt.plot(y_pred[0,:,1,0])
plt.plot(R_train[0,:,1,0])
plt.show()

# %% Keras
@tf.function
def generate_simple_filter(tau,n,t):
   f = (t**n)*tf.math.exp(-t/tau) # functional form in paper
   f = (f/tau**(n+1))/tf.math.exp(tf.math.lgamma(n+1)) # normalize appropriately
   return f

@tf.function
def get_kernelTensor(kernel_1D,spatial_dims):
    tile_fac = tf.constant([spatial_dims[0]*spatial_dims[1],1])    
    kernel_tensor = tf.tile(kernel_1D,tile_fac)
    kernel_tensor = tf.reshape(kernel_tensor,(spatial_dims[0],spatial_dims[1],kernel_1D.shape[-1]))
    kernel_tensor = tf.experimental.numpy.moveaxis(kernel_tensor,-1,0)
    return kernel_tensor

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
    # kernel_flipped = tf.reverse(kernel_reshaped,[0])
    pad_vec = [[0,0],[kernel_1D.shape[-1]-1,0],[0,0],[0,0]]
    conv_output = tf.nn.depthwise_conv2d(x_reshaped,kernel_reshaped,strides=[1,1,1,1],padding=pad_vec,data_format='NHWC')
    
    print(conv_output.shape)
    return conv_output

class photoreceptor_DA(keras.layers.Layer):
    def __init__(self,units=1):
        super(photoreceptor_DA,self).__init__()
        self.units = units
            
    def build(self,input_shape):
        alpha_init = tf.random_normal_initializer()
        self.alpha = tf.Variable(name='alpha',initial_value=alpha_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        beta_init = tf.random_normal_initializer(mean=0.36)
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        gamma_init = tf.random_normal_initializer(mean=0.448)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        tauY_init = tf.random_normal_initializer(mean=2) #tf.random_uniform_initializer(minval=1)
        self.tauY = tf.Variable(name='tauY',initial_value=tauY_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        tauZ_init = tf.random_normal_initializer(mean=150) #tf.random_uniform_initializer(minval=100)
        self.tauZ = tf.Variable(name='tauZ',initial_value=tauZ_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        nY_init = tf.random_normal_initializer(mean=1) #tf.random_uniform_initializer(minval=1)
        self.nY = tf.Variable(name='nY',initial_value=nY_init(shape=(1,self.units),dtype='float32'),trainable=True)   
        
        nZ_init = tf.random_normal_initializer(mean=1) #tf.random_uniform_initializer(minval=1)
        self.nZ = tf.Variable(name='nZ',initial_value=nZ_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
    def call(self,inputs):
       
        timeBin = 8
        
        alpha =  float(self.alpha / timeBin)
        beta = float(self.beta / timeBin)
        gamma =  float(self.gamma)
        tau_y =  float(self.tauY / timeBin)
        tau_z =  float(self.tauZ / timeBin)
        n_y =  float(self.nY)
        n_z =  float(self.nZ)
        
        t = tf.range(0,1000/timeBin,dtype='float32')
        
        Ky = generate_simple_filter(tau_y,n_y,t)
        # Ky_tens = get_kernelTensor(Ky,spatial_dims)
    
        Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter(tau_z,n_z,t))
        # Kz_tens = get_kernelTensor(Kz,spatial_dims)
       
        y_tf = conv_oper(inputs,Ky)
        z_tf = conv_oper(inputs,Kz)
        
        outputs = (alpha*y_tf)/(1+(beta*z_tf))
        
        return outputs


# print(model.predict([10.0]))
# print(my_layer.variables)


