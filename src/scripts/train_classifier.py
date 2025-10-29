#! /usr/bin/env python
import os, sys
currentdir = os.getcwd()
updir = os.path.dirname(currentdir)
sys.path.append(updir)



import argparse

import numpy as np
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers


from utils import preprocessing as pre
from utils import models_classifyer as models
from utils.callbacks import coef_det_k
import pandas as pd
import yaml

#from tensorflow.keras import mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)


parser = argparse.ArgumentParser(""" """)

parser.add_argument('-c', '--config', type=str, default = 'config.yaml',
                   help = 'Configuration file that configures all parameters')
parser.add_argument('-o', '--output', type=str, default = '../weights/classifier1.h5')

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")
parser.add_argument('-g', '--gpu', type=str, choices=['0', '1','0,1'], default='0,1')




def main(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    
    with open(args.config, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        
        
    data_train, data_val = pre.load_data(config["Data"])
    
    #
    opt = keras.optimizers.Adam(learning_rate=config['Classifier']['learning_rate'])
    
    if config['model'] == "class":
        model = models.Classifier_class(config['Classifier'])
        metric = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        model = models.get_classifier(config['Classifier'], 21)
        metric = coef_det_k
        loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_error')
        
    model.compile(optimizer=opt, loss=loss, metrics=[metric], run_eagerly = False)
    model.summary()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=20, min_lr=0.000001)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta = 0.01)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=config['Classifier']['checkpoint_filepath'],
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
    x_val = data_val.batch(config['Classifier']['batch_size'], drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    x_train = data_train.batch(config['Classifier']['batch_size'], drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE) 
    
    history = model.fit(x_train, epochs=config['Classifier']['epochs'], validation_data = x_val, callbacks=[reduce_lr, early_stop, model_checkpoint_callback])
    
    #model.save(args.output)
    df = pd.DataFrame(history.history)
    df.to_csv(f"{config['Classifier']['history']}/history.csv")
    
    return 0

if __name__ == "__main__":
    
    args = parser.parse_args()
    main(args)
