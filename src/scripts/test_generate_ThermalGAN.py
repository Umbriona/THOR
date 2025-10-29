#! /usr/bin/env python
import os, sys
try:
    os.chdir('/ThermalGAN/src/scripts')
except:
     os.chdir('./')
currentdir = os.path.dirname(os.getcwd())

print(os.listdir(os.getcwd()))
print(os.listdir(currentdir))
print(os.listdir(os.path.dirname(currentdir)))
sys.path.append(currentdir)

import argparse
import yaml
import datetime
import pandas as pd
import threading

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from threading import Thread

from utils.loaders import load_data, load_optimizers, load_metrics, load_losses
from utils import models_cyclegan_self_loss as models_gan
from utils import models_classifyer as models_class
from utils import callbacks
from utils import preprocessing as pre

from simple_pid import PID

parser = argparse.ArgumentParser(""" """)


parser.add_argument("--epoch", type=int, nargs="+", default=1999,
			help = "Specify epoch to load")

parser.add_argument('-c', '--config', type=str, default = 'config.yaml',
                   help = 'Configuration file that configures all parameters')

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")
parser.add_argument('-g', '--gpu', type=str, choices=['0', '1','0,1'], default='0,1')

args = parser.parse_args()

REPLICATES=100

def generate(config, model, data, time, classifyer, epoch):
    REPLICATES=config['CycleGan']['Generate']['replicates']
    print(f"Making: {REPLICATES} for each sequence")
    try: 
        print("setting dynamic to {}".format(config['CycleGan']["dynamic"]))
        dynamic = config['CycleGan']["dynamic"]
    except:
        print("Setting dynamic to True")
        dynamic = True
        
    print(dynamic)
    #file writers
    result_dir = os.path.join(config['Results']['base_dir'],time)
    
    base_dir = os.path.join(config['Log']['base_dir'],time)

    batches_x = data['meso_train'].batch(config['CycleGan']['batch_size'], drop_remainder=False) 
    batches_y = data['thermo_train'].batch(config['CycleGan']['batch_size'], drop_remainder=False)

    for rep in range(config['CycleGan']['Generate']['replicates']): 

        for batch in zip(batches_x, batches_y):
            if config['CycleGan']['BERT'] in ["generator", "all"]:
                temp_diff_y, temp_diff_x, (temp_real_x, temp_fake_x),  (temp_real_y, temp_fake_y) = model.validate_step_bert(batch)
                ids, fake_y = model.generate_step_bert(batch)

            else:
                ids, fake_y = model.generate_step(batch)

            with open(f"{config['Results']['base_dir']}/{config['CycleGan']['Generate']['name_fasta']}_epoch{epoch}_{REPLICATES}replicates.fasta", "a") as f:
                for _id, fake, temp_diff, temp_fake in zip(ids, fake_y, temp_diff_x, temp_fake_y):
                    #print(f"Writing id: {_id}")
                    f.write(f">{_id}_{rep} {epoch} {temp_diff} {temp_fake}\n{fake}\n")

    return 


def main():
    
    # GPU setting

    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"
    # Get time
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Load configuration file
    with open(args.config, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        
    with open(args.config, 'r') as file_descriptor:
        config_str = file_descriptor.read()
    
    result_dir = os.path.join(config['Results']['base_dir'],time + os.path.basename(args.config))
    os.mkdir(os.path.join(result_dir))
    os.mkdir(os.path.join(result_dir,'weights'))
    
    # Load training data
    if config['CycleGan']['BERT'] in ["generator", "all"]:
        data_train_meso, data_val_meso = pre.load_data_bert(config["Data_meso"])
        data_train_thermo, data_val_thermo = pre.load_data_bert(config["Data_thermo"])
    else:
        data_train_meso, data_val_meso = pre.load_data(config["Data_meso"], model="cycle_gan")
        data_train_thermo, data_val_thermo = pre.load_data(config["Data_thermo"], model="cycle_gan")
    data = {'meso_train': data_train_meso, 'thermo_train': data_train_thermo, 'meso_val':data_val_meso , 'thermo_val': data_val_thermo}
    #data = load_data(config['Data'])
    
    # Callbacks
    #cb = callbacks.PCAPlot(data['thermo_train'].as_numpy_iterator(), data['meso_train'].as_numpy_iterator(), data['n_thermo_train'], data['n_meso_train'], logdir=os.path.join(config['Log']['base_dir'],time,'img')) 
    
        # load classifyer
    models_weights = ["../../weights/OGT/Model1/variables/variables", "../../weights/OGT/Model2/variables/variables", "../../weights/OGT/Model3/variables/variables"]
    names = ["model1", "model2", "model3"]
    file = "../../config/Classifier/config_classifier1.yaml"
    with open(file, 'r') as file_descriptor:
        config_class = yaml.load(file_descriptor, Loader=yaml.FullLoader)
    model_input = tf.keras.layers.Input(shape=(512,21))
    model1 = models_class.get_classifier(config_class['Classifier'], 21)
    model2 = models_class.get_classifier(config_class['Classifier'], 21)
    model3 = models_class.get_classifier(config_class['Classifier'], 21)

    output1 = model1(model_input)
    output2 = model2(model_input)
    output3 = model3(model_input)

    #model1.summary()

    model1.load_weights(models_weights[0]).expect_partial()
    model2.load_weights(models_weights[1]).expect_partial()
    model3.load_weights(models_weights[2]).expect_partial()

    ensemble_output = tf.keras.layers.Average()([output1, output2, output3])
    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)

    #ensemble_model.summary()
    
    
    # Initiate model
    model = models_gan.CycleGan(config, classifier = ensemble_model)
    loss_obj  = load_losses(config['CycleGan']['Losses'])
    optimizers = load_optimizers(config['CycleGan']['Optimizers'])
    model.compile(loss_obj, optimizers)
    for epoch in config['CycleGan']['Generate']['epochs']:
        print(f"Loading weights at: /mimer/NOBACKUP/groups/snic2022-6-127/ThermalGAN/results/{config['CycleGan']['Generate']['dir_model']}/weights/epoch_{epoch}")
        model.load_gan(f"/mimer/NOBACKUP/groups/snic2022-6-127/ThermalGAN/results/{config['CycleGan']['Generate']['dir_model']}/weights/epoch_{epoch}")

    # Initiate Training

        history = generate(config, model, data, time + os.path.basename(args.config), classifyer = ensemble_model, epoch=epoch)

    #writing results

    # Save model
#    model.save_weights(os.path.join(result_dir,'weights','cycle_gan_model_final'))
    # Write history obj
#    df = pd.DataFrame(history)
#    df.to_csv(os.path.join(result_dir,'history.csv'))
    # Save config_file
#    with open(os.path.join(result_dir, 'config.yaml'), 'w') as file_descriptor:
#        file_descriptor.write(config_str)
        
    return 0
    
if __name__ == "__main__":
    main()
