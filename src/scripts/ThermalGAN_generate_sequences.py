import os
import sys

sys.path.append("/vault/ThermalGAN/src")
sys.path.append("/vault/ThermalGAN/src/utils")


from Bio import SeqIO
import argparse
import yaml
import datetime
import pandas as pd
import threading

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from threading import Thread

from utils.loaders import load_data, load_optimizers, load_metrics, load_losses
from utils import models_cyclegan as models_gan
from utils import models_classifyer as models_class
from utils import callbacks
from utils import preprocessing as pre

import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", type=str, required=True,
                   help="""Fasta file that has Wt sequences that you want to make hot
                   """)
parser.add_argument("-o", "--output", type=str, required=True,
                   help="Output file")

parser.add_argument("--replicate", type=int, default=1)


LIST_GANS=[#"/vault/ThermalGAN/result/20230403-114314train_from_scratch.yaml/weights/epoch_23",
            #"/vault/ThermalGAN/result/20230403-114314train_from_scratch.yaml/weights/epoch_118",
            "/vault/ThermalGAN/result/20230403-114314train_from_scratch.yaml/weights/epoch_125",
            "/vault/ThermalGAN/result/20230401-235116train_from_scratch.yaml/weights/epoch_60",
            "/vault/ThermalGAN/result/20230401-235116train_from_scratch.yaml/weights/epoch_99",
            "/vault/ThermalGAN/result/20230401-235116train_from_scratch.yaml/weights/epoch_102/epoch_102",
            "/vault/ThermalGAN/result/20230403-114239train_from_scratch.yaml/weights/epoch_290",
            "/vault/ThermalGAN/result/20230403-114239train_from_scratch.yaml/weights/epoch_296",
            "/vault/ThermalGAN/result/20230403-114242train_from_scratch.yaml/weights/epoch_17",
            "/vault/ThermalGAN/result/20230403-114242train_from_scratch.yaml/weights/epoch_262"]






def load_regression(args):
    models_weights = ["/vault/ThermalGAN/weights/OGT/Model1/variables/variables",
                      "/vault/ThermalGAN/weights/OGT/Model2/variables/variables",
                      "/vault/ThermalGAN/weights/OGT/Model3/variables/variables"]

    names = ["model1", "model2", "model3"]
    file = "/vault/ThermalGAN/config/Classifier/config_classifier1.yaml"
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
    return ensemble_model

#ensemble_model.summary()

def load_sequences(args):


    query_fasta=args.input

    fasta_dict={"id":[],"seq":[],"seq_bin":[],"EC":[], "data":[], "w":[], "OGT":[]}

    for n_seq, rec in enumerate(SeqIO.parse(query_fasta, "fasta")):
        fasta_dict["id"].append(rec.id)
        fasta_dict["seq"].append(str(rec.seq))
        fasta_dict["seq_bin"].append(tf.constant(pre.to_binary(str(rec.seq), 512).reshape((1,512,21))))
        w = np.zeros((1,512), dtype=np.float32)
        w[0,0:len(str(rec.seq))]=1
        fasta_dict["w"].append(tf.constant(w))
        fasta_dict["EC"].append("1.1.1.37")
        fasta_dict["data"].append("MDH")
        #fasta_dict["OGT"].append(rec.description.split()[-1])
        #if n_seq >= 99: # take first 100
        #    break
    return  fasta_dict

def write_seequences(args, fasta_dict, model, model_name):
    batch = [(fasta_dict["seq_bin"][0], "_", fasta_dict["w"][0]), (fasta_dict["seq_bin"][0], "_", fasta_dict["w"][0])]
    temp_fake= 0
    counter = 0

    temperatures_generated = []
    temperatures_wt = []
    with open(f"/vault/protein_engineering_jira/results/EE_2023-04-10/{args.output}_{model_name}.fasta", "w") as file_writer:
        for i in range(len(fasta_dict["seq_bin"])):
            batch = [(fasta_dict["seq_bin"][i], "_", fasta_dict["w"][i]), (fasta_dict["seq_bin"][i], "_", fasta_dict["w"][i])]
            for replicate in range(args.replicate):
                fake_y = model.generate_step(batch)
                fake_y_bin = tf.constant(pre.to_binary(fake_y[0], 512).reshape((1,512,21)))
                temp_fake_y = model.classifier(fake_y_bin,training=False).numpy()[0,0]

                temp_x = model.classifier(fasta_dict["seq_bin"][i],training=False).numpy()[0,0]

                # Calc identity
                count = 0 
                for res_wt, res_mut in zip(fasta_dict['seq'][i], fake_y[0]):
                    if res_wt == res_mut:
                        count += 1
                identity = count / len(fasta_dict['seq'][i])

                # Write WT
                file_writer.write(f">{fasta_dict['id'][i]} {temp_x}\n{fasta_dict['seq'][i]}\n")
                # Write generated
                file_writer.write(f">{fasta_dict['id'][i]}_{replicate}_gen {identity} {temp_fake_y}\n{fake_y[0]}\n")
    return 0
          

def main(args):
    
    ensemble_model = load_regression(args)
    dir_experiment = "/vault/ThermalGAN/result/20230401-235116train_from_scratch.yaml/"
    config = os.path.join(dir_experiment,"config.yaml")                            
    with open(config, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)
    # Initiate model
    model = models_gan.CycleGan(config, classifier = ensemble_model)
                                
    fasta_dict = load_sequences(args)
    
    for gan in LIST_GANS:
        model.load_gan(gan)
        write_seequences(args, fasta_dict, model, gan.split("_")[-1])
    
    return 0
    
if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    main(args)