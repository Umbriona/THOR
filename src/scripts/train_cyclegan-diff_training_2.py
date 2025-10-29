#! /usr/bin/env python
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 
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
from time import time as TIME

import numpy as np
#import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.python.profiler import profiler_v2 as profiler
#from matplotlib import pyplot as plt
from threading import Thread

from utils.loaders import load_data, load_optimizers, load_metrics, load_losses
from utils import models_cyclegan_self_loss as models_gan
from utils import models_classifyer as models_class
from utils import callbacks
from utils import preprocessing as pre

from simple_pid import PID

mixed_precision.set_global_policy('mixed_bfloat16')

os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

parser = argparse.ArgumentParser(""" """)

parser.add_argument('-c', '--config', type=str, default = 'config.yaml',
                   help = 'Configuration file that configures all parameters')

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")
parser.add_argument('-g', '--gpu', type=str, choices=['0', '1','0,1'], default='0,1')

args = parser.parse_args()

class History():
    def __init__(self,):
        self.history = {
        "Gen_G_loss": [],
        "Cycle_X_loss": [],
        "Disc_X_loss": [],
        "Gen_F_loss": [],
        "Cycle_Y_loss": [],
        "Disc_Y_loss": [],
        "x_acc":[],
        "y_acc":[],
        "x_c_acc":[],
        "y_c_acc":[],
        "temp_diff_x":[],
        "temp_diff_y":[],
        "temp_diff_x_median":[],
        "temp_diff_y_median":[],
        "lambda_G":[],
        "lambda_F":[]
        
    }

    def write_history(self, model):
        


def train(config, model, data, time, classifyer):

    lr_drop_flag = False
    #file writers
    result_dir = os.path.join(config['Results']['base_dir'],time)
    
    base_dir = os.path.join(config['Log']['base_dir'],time)


    ###################################### Setting up BLAST object ###############################
  
    mutex = threading.RLock()
        
    output_file = 'blast_out'
    out_format = '"6 qseqid sseqid score pident"'
    matrix_type = ['BLOSUM45', 'BLOSUM62']
    db_location_meso = config['Log']['test_meso']
    db_location_target = config['Log']['test_target']
    query_meso = callbacks.queryDB(output_file, out_format, [db_location_meso], result_dir,  matrix_type[0], base_dir, name="Meso")
    query_target = callbacks.queryDB(output_file, out_format, [db_location_target], result_dir,  matrix_type[0], base_dir, name="Psychro")
    flag_blast=True
    save_epoch = 0

    #################################################################################################




    ######################################### Initiate PID controlers ########################################
    diff_x=0
    diff_y=0
    pid_G = PID(config['CycleGan']['PID_P'], config['CycleGan']['PID_I'], config['CycleGan']['PID_D'], setpoint=config['CycleGan']['lambda_self'], sample_time=None)
    pid_F = PID(config['CycleGan']['PID_P'], config['CycleGan']['PID_I'], config['CycleGan']['PID_D'], setpoint=config['CycleGan']['lambda_self'], sample_time=None)

    pid_G.output_limits = (0, 100)
    pid_F.output_limits = (0, 100)

    print("Created PID controlers")

    ############################################################################################################

    ############################################## Training loop ###############################################
    
    for epoch in range(config['CycleGan']['epochs']):

        batches_x = data['meso_train'].batch(config['CycleGan']['batch_size'], drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE) # tf.data.AUTOTUNE
        batches_y = data['thermo_train'].batch(config['CycleGan']['batch_size'], drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

        print(f"Created batches for epoch: {epoch}")
        #Anneal schedule for gumbel

        


        start_time = TIME()
        for step, x in enumerate(zip(batches_x,batches_y)):

            model.train_step_bert_all( batch_data = x) # returns losses_, logits = 
            print(f"Acc_X is: {model.metricss['acc_x'].result()}  Acc_Y is: {model.metricss['acc_y'].result()}")
            if step%200 == 0 and step>0:
                print(f"Acc_X is: {model.metricss['acc_x'].result()}  Acc_Y is: {model.metricss['acc_y'].result()}")
                lambda_self_G = pid_G(float(model.metricss['acc_x'].result()), dt=1)
                lambda_self_F = pid_F(float(model.metricss['acc_y'].result()), dt=1)

                tf.keras.backend.set_value(model.lambda_self_G,  lambda_self_G)
                print("Lambda self G", lambda_self_G)
                tf.keras.backend.set_value(model.lambda_self_F,  lambda_self_F)
                print("Lambda self F", lambda_self_F)
                model.metricss['acc_x'].reset_states()
                model.metricss['acc_y'].reset_states()
                gms_g = model.G.get_layer(index=-2)
                gms_f = model.F.get_layer(index=-2)
                tf.keras.backend.set_value(gms_g.tau,    max(0.1, np.exp(-0.0001*(1+epoch)*step)))
                tf.keras.backend.set_value(gms_f.tau,    max(0.1, np.exp(-0.0001*(1+epoch)*step)))


        print(f"Step number: {step}")
        print("Epoch: %d Loss_G: %2.4f Loss_F: %2.4f Loss_cycle_X: %2.4f Loss_cycle_Y: %2.4f Loss_D_Y: %2.4f Loss_D_X %2.4f" % 
              (epoch, float(model.metricss['loss_G'].result()),
               float(model.metricss['loss_F'].result()),
               float(model.metricss['loss_cycle_x'].result()),
               float(model.metricss['loss_cycle_y'].result()),
               float(model.metricss['loss_disc_y'].result()),
               float(model.metricss['loss_disc_x'].result())))               
        print("Epoch: %d acc trans x: %2.4f acc trans y: %2.4f acc cycled x : %2.4f acc cycled y: %2.4f" % 
            (epoch, model.metricss['acc_x'].result(),
            model.metricss['acc_y'].result(),
            model.metricss['cycled_acc_x'].result(),
            model.metricss['cycled_acc_y'].result()))
        stop_time = TIME()
        print(f"Time for trining steps: {stop_time - start_time}")

        ################################ Validation steps #################################
        start_time = TIME()
        val_x = data['meso_val'].batch(config['CycleGan']['batch_size'], drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_y = data['thermo_val'].batch(config['CycleGan']['batch_size'], drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
        for i, x in enumerate(zip(val_x, val_y)):
            model.validate_step_bert(x)

        stop_time = TIME()
        print(f"Time for validation steps: {stop_time - start_time}")

        #####################################################################################

        ############################ Generating test sequences ##############################
        if epoch % 10 == 0:
            print("Generating test sequences")
            val_x = data['meso_val'].batch(config['CycleGan']['batch_size'], drop_remainder=True)
            val_y = data['thermo_val'].batch(config['CycleGan']['batch_size'], drop_remainder=True)

            for batch in zip(val_x, val_y):
                _, fake_y = model.generate_step_bert(batch)

                if flag_blast:
                    generated = callbacks.fastaFromList(fake_y, epoch, query_meso.toTempDir)
                    in_queue_meso = Thread(target=query_meso, args=(generated, epoch, mutex, ))
                    in_queue_meso.start()
                    in_queue_target = Thread(target=query_target, args=(generated, epoch, mutex, ))
                    in_queue_target.start()
                break # Only do one batch 
            for seq in fake_y[:10]:
                print(seq[:200])

	####################################### Saving models ##########################################
        identity_2_wt = model.metricss['acc_x'].result()
        temperature_diff_2_wt = model.metricss['temp_diff_x'].result()
        time_since_save = epoch - save_epoch
        if identity_2_wt >= 0.70 and temperature_diff_2_wt > 20:
            save_epoch = epoch
            model.save_gan(os.path.join(result_dir,'weights',f"epoch_{epoch}"), pid_G, pid_F)
        elif identity_2_wt >= 0.70 and temperature_diff_2_wt > 5 and time_since_save >= 5:
            save_epoch = epoch
            model.save_gan(os.path.join(result_dir,'weights',f"epoch_{epoch}"), pid_G, pid_F)
        elif epoch == 1:
            save_epoch = epoch
            model.save_gan(os.path.join(result_dir,'weights',f"epoch_{epoch}"), pid_G, pid_F)
    ################################################################################################

    ##################################### Learning Rate decay ###################################### 

        if identity_2_wt >= config['CycleGan']['lambda_self'] and lr_drop_flag:
            lr_g  = tf.keras.backend.get_value(model.G.optimizer.learning_rate)
            lr_f  = tf.keras.backend.get_value(model.F.optimizer.learning_rate)
            dx_lr = tf.keras.backend.get_value(model.D_x.optimizer.learning_rate)
            dy_lr = tf.keras.backend.get_value(model.D_y.optimizer.learning_rate)
            tf.keras.backend.set_value(model.G.optimizer.learning_rate,   lr_g / 10)# max(lr_f * 0.2,  0.05 * config['CycleGan']['Optimizers']["learning_rate_generator"]))
            tf.keras.backend.set_value(model.F.optimizer.learning_rate, lr_f / 10)# max(dx_lr * 0.2, 0.05 * config['CycleGan']['Optimizers']["learning_rate_discriminator"]))
            tf.keras.backend.set_value(model.D_x.optimizer.learning_rate, dx_lr / 10)# max(dy_lr * 0.2, 0.05 * config['CycleGan']['Optimizers']["learning_rate_discriminator"]))
            tf.keras.backend.set_value(model.D_y.optimizer.learning_rate, dy_lr / 10)
            lr_drop_flag = False
    
    ################################################################################################
        lr_g  = tf.keras.backend.get_value(model.G.optimizer.learning_rate)
        print(f"Learning rate is {lr_g}")
        
        if args.verbose:    
            print("Epoch: %d Loss_G: %2.4f Loss_F: %2.4f Loss_cycle_X: %2.4f Loss_cycle_Y: %2.4f Loss_D_Y: %2.4f Loss_D_X %2.4f" % 
              (epoch, float(model.metricss['loss_G'].result()),
               float(model.metricss['loss_F'].result()),
               float(model.metricss['loss_cycle_x'].result()),
               float(model.metricss['loss_cycle_y'].result()),
               float(model.metricss['loss_disc_y'].result()),
               float(model.metricss['loss_disc_x'].result())))               
            print("Epoch: %d acc trans x: %2.4f acc trans y: %2.4f acc cycled x : %2.4f acc cycled y: %2.4f" % 
              (epoch, model.metricss['acc_x'].result(),
               model.metricss['acc_y'].result(),
               model.metricss['cycled_acc_x'].result(),
               model.metricss['cycled_acc_y'].result()))
            
            gms_g = model.G.get_layer(index=-2)
            gms_f = model.F.get_layer(index=-2)
            print("G tau %1.4f F tau %1.4f" % (tf.keras.backend.get_value(gms_g.tau), tf.keras.backend.get_value(gms_f.tau)))
            print(f"Temperature diff Mesofiles: {temperature_diff_2_wt}\nTemperature diff Psychrofile: {model.metricss['temp_diff_y'].result()}")
            






            
        # Save history object
        history["Gen_G_loss"].append(model.metricss['loss_G'].result().numpy())
        history["Cycle_X_loss"].append(model.metricss['loss_cycle_x'].result().numpy())
        history["Disc_X_loss"].append(model.metricss['loss_disc_x'].result().numpy())
        history["Gen_F_loss"].append(model.metricss['loss_F'].result().numpy())
        history["Cycle_Y_loss"].append(model.metricss['loss_cycle_y'].result().numpy())
        history["Disc_Y_loss"].append(model.metricss['loss_disc_y'].result().numpy())
        history["x_acc"].append(model.metricss['acc_x'].result().numpy())
        history["x_c_acc"].append(model.metricss['cycled_acc_x'].result().numpy())
        history["y_acc"].append(model.metricss['acc_y'].result().numpy())
        history["y_c_acc"].append(model.metricss['cycled_acc_y'].result().numpy())
        history["temp_diff_x"].append(model.metricss['temp_diff_x'].result().numpy())
        history["temp_diff_y"].append(model.metricss['temp_diff_y'].result().numpy())

        history["temp_diff_x_median"].append(model.metricss['temp_diff_x'].result().numpy())
        history["temp_diff_y_median"].append(model.metricss['temp_diff_y'].result().numpy())

        history["lambda_G"].append(lambda_self_G)
        history["lambda_F"].append(lambda_self_F)
        # Reset states
        model.metricss['loss_G'].reset_states()
        model.metricss['loss_cycle_x'].reset_states()
        model.metricss['loss_disc_y'].reset_states()
        model.metricss['loss_F'].reset_states() 
        model.metricss['loss_cycle_y'].reset_states()
        model.metricss['loss_disc_x'].reset_states()
        model.metricss['loss_id_y'].reset_states()
        model.metricss['loss_id_x'].reset_states()

        model.metricss['acc_x'].reset_states()
        model.metricss['acc_y'].reset_states()
        model.metricss['cycled_acc_x'].reset_states()
        model.metricss['cycled_acc_y'].reset_states()
        model.metricss['id_acc_y'].reset_states()
        model.metricss['id_acc_x'].reset_states()
        
        model.metricss["bert_wt_mut_acc_x"].reset_states()
        model.metricss["bert_wt_mut_acc_y"].reset_states()
        
        model.metricss['temp_diff_x'].reset_states()
        model.metricss['temp_diff_y'].reset_states()

        df = pd.DataFrame(history)
        df.to_csv(os.path.join(result_dir,'history.csv'))
    
    return history


def main():
    print(f"Tf version is {tf.__version__}")
    os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"
    # GPU setting

    #os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    
    # Get time
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Load configuration file
    with open(args.config, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        
    with open(args.config, 'r') as file_descriptor:
        config_str = file_descriptor.read()
    
    result_dir = os.path.join(config['Results']['base_dir'],time + os.path.basename(args.config))
    os.mkdir(os.path.join(result_dir))
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as file_descriptor:
        file_descriptor.write(config_str)
    os.mkdir(os.path.join(result_dir,'weights'))
    
    # Load training data
    if config['CycleGan']['BERT'] in ["generator", "all"]:
        data_train_meso, data_val_meso = pre.load_compact_data_bert(config, "Data_meso")#["Data_meso"])
        data_train_thermo, data_val_thermo = pre.load_compact_data_bert(config, "Data_thermo")#["Data_thermo"])
    else:
        data_train_meso, data_val_meso = pre.load_data(config["Data_meso"], model="cycle_gan")
        data_train_thermo, data_val_thermo = pre.load_data(config["Data_thermo"], model="cycle_gan")
    data = {'meso_train': data_train_meso, 'thermo_train': data_train_thermo, 'meso_val':data_val_meso , 'thermo_val': data_val_thermo}
    #data = load_data(config['Data'])
    
    # Callbacks
    #cb = callbacks.PCAPlot(data['thermo_train'].as_numpy_iterator(), data['meso_train'].as_numpy_iterator(), data['n_thermo_train'], data['n_meso_train'], logdir=os.path.join(config['Log']['base_dir'],time,'img')) 
    print("Loaded data")
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

    print("Initialized regression models")

    output1 = model1(model_input)
    output2 = model2(model_input)
    output3 = model3(model_input)

    #model1.summary()

    model1.load_weights(models_weights[0]).expect_partial()
    model2.load_weights(models_weights[1]).expect_partial()
    model3.load_weights(models_weights[2]).expect_partial()

    print("Loaded weights regression model")

    ensemble_output = tf.keras.layers.Average()([output1, output2, output3])
    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)

    
    #ensemble_model.summary()
    
    
    # Initiate model
    model = models_gan.CycleGan(config, classifier = ensemble_model)

    print("Loaded ThermalGAN")
    
    loss_obj  = load_losses(config['CycleGan']['Losses'])
    optimizers = load_optimizers(config['CycleGan']['Optimizers'])
    model.compile(loss_obj, optimizers)

    print("Compiled ThermalGAN")

    # Initiate Training

    history = train(config, model, data, time + os.path.basename(args.config), classifyer = ensemble_model)
    
    #writing results
    
    
    # Save model
    model.save_weights(os.path.join(result_dir,'weights','cycle_gan_model_final'))
    # Write history obj
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(result_dir,'history.csv'))
    # Save config_file

        
    return 0
    
if __name__ == "__main__":
    main()
