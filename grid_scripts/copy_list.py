#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:24:19 2021

@author: saad
"""
import os

expDate = 'retina1'
pathToModel = '/mnt/cedar/scratch/RetinaPredictors/data/'+expDate+'/CNN_2D/'
pathToModel_actual = '~/scratch/RetinaPredictors/data/retina1/CNN_2D/'
foldList = os.listdir(pathToModel)

fileList = ([])


for foldName in foldList:
    # fileList.append(os.path.join(pathToModel_actual,foldName))
    fileList.append(os.path.join(foldName,'performance'))
    # fileList.append(os.path.join(pathToModel_actual,foldName,'performance',expDate+'_validation_'+foldName+'.csv'))
    # fileList.append(os.path.join(pathToModel_actual,foldName,'performance',expDate+'_'+foldName+'.h5'))
    fileList.append(os.path.join(foldName,foldName,'saved_model.pb'))
    fileList.append(os.path.join(foldName,foldName,'assets'))
    fileList.append(os.path.join(foldName,foldName,'variables'))
    # fileList.append(os.path.join(pathToModel_actual,foldName,foldName,'variables','variables.data-00000-of-00001'))
    # fileList.append(os.path.join(pathToModel_actual,foldName,foldName,'variables','variables.index'))
    
# for foldName in foldList:
#     fileList.append('+'+os.path.join(foldName))
#     fileList.append('+'+os.path.join(foldName,'performance','**'))

fname_copyList = '/home/saad/postdoc_db/projects/RetinaPredictors/grid_scripts/copyList.txt'
with open(fname_copyList, 'w') as filehandle:
    for listitem in fileList:
        filehandle.write('%s\n' % listitem)