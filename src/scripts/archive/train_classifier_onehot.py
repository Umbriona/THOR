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
parser.add_argument('-m', '--model_number', type=int, choices=[0, 1, 2], help="Index model to load")
parser.add_argument('-o', '--output', type=str, default = '../weights/classifier1.h5')

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")





def main(args):
    
   
    
    with open(args.config, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)

    data_train, data_val = pre.load_data_finetune_reg(config["Data"])
        

    models_weights = ["../../weights/OGT/Model1/variables/variables", "../../weights/OGT/Model2/variables/variables", "../../weights/OGT/Model3/variables/variables"]
    new_model_weights = ["../../weights/OGT_IMG/Model1/variables/variables", "../../weights/OGT_IMG/Model2/variables/variables", "../../weights/OGT_IMG/Model3/variables/variables"]
    names = ["model1", "model2", "model3"]
    file = "../../config/Classifier/config_classifier1.yaml"
    with open(file, 'r') as file_descriptor:
        config_class = yaml.load(file_descriptor, Loader=yaml.FullLoader)
    model_input = tf.keras.layers.Input(shape=(512,21))
    model = models_class.get_classifier(config_class['Classifier'], 21)

    opt = keras.optimizers.Adam(learning_rate=config['Classifier']['learning_rate'])
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
    
    history = model.fit(x_train, epochs=config['Classifier']['epochs'], validation_data = x_val, callbacks=[reduce_lr, early_stop, model_checkpoint_callback])
    
    #model.save(args.output)
    df = pd.DataFrame(history.history)
    df.to_csv(f"../../weights/OGT_IMG/Model{args.model_number}/history.csv")
    
    return 0

if __name__ == "__main__":
    
    args = parser.parse_args()
    main(args)
