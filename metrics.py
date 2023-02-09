#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 23:13:50 2022

@author: rtx3090
"""

# ***************** Metricas ******************
import os, time, warnings, numpy as np, tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from keras.models import load_model
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def get_X_and_Y(ds):
    Xl, Yl = list(), list()
    for image, label in tqdm(ds):
      Yl.append(label.numpy()), Xl.append(image.numpy())
    Y = np.array(Yl)
    X= np.array(Xl)     
    print('ready')
    return X,Y

trial=3
fold=5

path= '/media/rtx3090/8c77f284-2a4d-4b21-8fb7-4b461ebb4e15/Alea/results/trials/0'+str(trial)+'/ds_val_trial_'+str(trial)+'/ds_val_'+str(fold)+'/'
path_model= '/media/rtx3090/8c77f284-2a4d-4b21-8fb7-4b461ebb4e15/Alea/results/trials/0' +str(trial) +'/efficient_trial__fold_'+str(fold-1) +'.hdf5'
print(path_model)
dataset= tf.data.experimental.load(path)

X,Y= get_X_and_Y(dataset)


model = load_model(path_model)

preds = model.predict(X)
preds_cls_idx = preds.argmax(axis=-1)
classification_reporte_g = classification_report(np.argmax(Y, axis = 1), preds_cls_idx, output_dict = True, target_names = ['Fixation','Saccade'])
print(classification_reporte_g)






from sklearn.metrics import cohen_kappa_score
print('COHEN_KAPPA:',cohen_kappa_score(np.argmax(Y, axis = 1),preds_cls_idx))

print(confusion_matrix(np.argmax(Y, axis = 1),preds_cls_idx))









