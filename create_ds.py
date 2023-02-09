# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:00:21 2022

@author: screa
"""

from utils import create_dataset_train_test_val
import tensorflow as tf, os, numpy as np


TF_ENABLE_ONEDNN_OPTS=0
dataset_path = '/home/rtx3090/Desktop/Alea/lund2013_fo/'
create_dataset_train_test_val(dataset_path, datasets_names = ['test','tr','vl'], img_height = 224, img_width = 224)


# test_ds = tf.data.experimental.load(path)
