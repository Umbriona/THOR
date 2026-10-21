#! /usr/bin/env python
import os, sys, json
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

from utils.loaders import load_data, load_optimizers, load_losses
from utils import models_cyclegan_self_loss as models_gan
from utils import models_classifyer as models_class
from utils import callbacks
from utils import preprocessing as pre

from simple_pid import PID

parser = argparse.ArgumentParser(""" """)


parser.add_argument("--epoch", type=int, default=1999,
			help = "Specify epoch to load")

parser.add_argument('-i', '--input', type=str, required=True,
                   help = 'Configuration file that configures all parameters')

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")
parser.add_argument('-g', '--gpu', type=str, choices=['0', '1','0,1'], default='0,1')

parser.add_argument('--store_softmax', action="store_true")

args = parser.parse_args()

REPLICATES=50

def generate(config, model, data):

    
    
    print(f"Making: {REPLICATES} for each sequence")

    batches = data.batch(config['CycleGan']['batch_size'], drop_remainder=False) 
    gms_g = model.G.get_layer("gumbel")
    gms_g.hard = True
    for rep in range(REPLICATES): 

        for batch in zip(batches):

            ids, seqs_real, seqs_fake, temps_real, temps_fake = model.generate_step_bert_inference(batch)

            with open(f"{args.input}/variants_{args.epoch}.fasta", "a") as f:
                for _id, real, fake, temp_real, temp_fake in zip(ids, seqs_real, seqs_fake, temps_real, temps_fake):
                    #print(f"Writing id: {_id}")
                    f.write(f">{_id}_wt_{rep} {temp_real}\n{real}\n")
                    f.write(f">{_id}_variant_{rep} {temp_fake}\n{fake}\n")
            
            if rep == 0:
                if args.store_softmax:
                    ids, seqs_real, seqs_fake, temps_real, temps_fake, raw_probs = model.generate_MAX_likelihood_step_bert_inference(batch, return_probs=args.store_softmax)
                else:
                    ids, seqs_real, seqs_fake, temps_real, temps_fake = model.generate_MAX_likelihood_step_bert_inference(batch, return_probs=args.store_softmax)

                with open(f"{args.input}/variants_SM_{args.epoch}.fasta", "a") as f:
                    for _id, real, fake, temp_real, temp_fake in zip(ids, seqs_real, seqs_fake, temps_real, temps_fake):
                        #print(f"Writing id: {_id}")
                        f.write(f">{_id}_wt_{rep} {temp_real}\n{real}\n")
                        f.write(f">{_id}_variant_{rep} {temp_fake}\n{fake}\n")
                if args.store_softmax:
                    probs_path = f"{args.input}/softmax_probs_{args.epoch}.jsonl"
                    with open(probs_path, "a") as prob_file:
                        for _id, probs in zip(ids, raw_probs):
                            prob_file.write(json.dumps({"id": _id, "probs": probs}) + "\n")
                
    return 


def main():
    
    # GPU setting

    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"
    # Get time
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Load configuration file
    config_file = f"{args.input}/config.yaml"
    with open(config_file, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)

    dataSet = pre.load_compact_data_bert_inference("/data/records/PETases/prot_bert_bfd_single/Meso*.tfrecord.gz")

    ###################### load classifyer #######################
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

    model1.load_weights(models_weights[0]).expect_partial()
    model2.load_weights(models_weights[1]).expect_partial()
    model3.load_weights(models_weights[2]).expect_partial()

    ensemble_output = tf.keras.layers.Average()([output1, output2, output3])
    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)

    ####################################################

    
    
    ###################### Initiate model ##############################
    model = models_gan.CycleGan(config, classifier = ensemble_model)
    loss_obj  = load_losses(config['CycleGan']['Losses'])
    optimizers = load_optimizers(config['CycleGan']['Optimizers'])
    model.compile(loss_obj, optimizers)

    #########
    pid_G = PID(config['CycleGan']['PID_P'], config['CycleGan']['PID_I'], config['CycleGan']['PID_D'], setpoint=config['CycleGan']['lambda_self'], sample_time=None)
    pid_F = PID(config['CycleGan']['PID_P'], config['CycleGan']['PID_I'], config['CycleGan']['PID_D'], setpoint=config['CycleGan']['lambda_self'], sample_time=None)

    pid_G.output_limits = (0, 20)
    pid_F.output_limits = (0, 20)

    print("Created PID controlers")
    ##########

    # Load weights 
    model.load_gan(f"{args.input}/weights/epoch_{args.epoch}", pid_G, pid_F)
    #####################################################################
    
    ######################## Generate sequences ##########################

    history = generate(config, model, dataSet)

    print("Done generating sequences!")
        
    return 0
    
if __name__ == "__main__":
    main()
