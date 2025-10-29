import os, sys
currentdir = os.path.dirname(os.getcwd())
sys.path.append(currentdir)

import argparse
import yaml
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import models_new
from utils import callbacks
from utils.loaders import load_data, load_optimizers, load_metrics, load_losses
from utils.preprocessing import convert_table


parser = argparse.ArgumentParser(""" """)

parser.add_argument('-c', '--config', type=str, default = 'config.yaml',
                   help = 'Configuration file that configures all parameters')

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")
parser.add_argument('-g', '--gpu', type=str, choices=['0', '1','0,1'], default='0,1')

args = parser.parse_args()
   
    
def evaluate(config, model, data):
    print('Start Evaluating')
    batches_x = data['meso_val'].batch(config['CycleGan']['batch_size'], drop_remainder=True) 
    batches_y = data['thermo_val'].batch(config['CycleGan']['batch_size'], drop_remainder=True)
    
    with open('Test.fasta', 'w') as f:
        for step, x in enumerate(zip(batches_x,batches_y)):
            logits_instyle, ids, W = model.generate_step( batch_data = x)
            _, logits_normal, _ = x[0]
            
            logits_instyle = tf.argmax(logits_instyle[0], axis= -1).numpy()
            logits_normal = tf.argmax(logits_normal, axis= -1).numpy()
           
            ids = ids[0].numpy()
            W   = W[0].numpy()
            for i in range(logits_instyle.shape[0]):
               
                seq_str = convert_table(logits_instyle[i,:],W[i])
                f.write('> '+'instyle_'+str(ids[i]).split(r"b'")[1] +'\n' + seq_str + '\n') 
                
                seq_str = convert_table(logits_normal[i,:],W[i])
                f.write('> '+str(ids[i]).split(r"b'")[1] +'\n' + seq_str + '\n')
    return 0


def main():
    
    # GPU setting

    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    
  
    # Load configuration file
    with open(args.config, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        
    with open(args.config, 'r') as file_descriptor:
        config_str = file_descriptor.read()
    
    result_dir = os.path.dirname(args.config)
    # Load training data
    data = load_data(config['Data'])
    
    # Initiate model
    model = models_new.CycleGan(config)#, callbacks = cb)
    loss_obj  = load_losses(config['CycleGan']['Losses'])
    optimizers = load_optimizers(config['CycleGan']['Optimizers'])
    model.compile(loss_obj, optimizers)
    
    model.load_weights(os.path.join(result_dir,'weights','cycle_gan_model')).expect_partial()
    
    evaluate(config, model, data)    
        
    return 0
    
if __name__ == "__main__":
    main()