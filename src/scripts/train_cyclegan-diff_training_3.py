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
#import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from matplotlib import pyplot as plt
from threading import Thread

from utils.loaders import load_data, load_optimizers, load_metrics, load_losses
from utils import models_cyclegan_self_loss as models_gan
from utils import models_classifyer as models_class
from utils import callbacks
from utils import preprocessing as pre

from simple_pid import PID

os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

parser = argparse.ArgumentParser(""" """)

parser.add_argument('-c', '--config', type=str, default = 'config.yaml',
                   help = 'Configuration file that configures all parameters')

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")
parser.add_argument('-g', '--gpu', type=str, choices=['0', '1','0,1'], default='0,1')

args = parser.parse_args()


def train(config, model, data, time, classifyer):
    
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
    #except:
    #    flag_blast=False
    #    print ('Initialization Error with BlastDB (for visuals)')

    G_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'G'))
    F_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'F'))

    D_x_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'D_x'))
    D_y_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'D_y'))

    X_c_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'X_c'))
    Y_c_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'Y_c'))
    
    X_id_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'X_id'))
    Y_id_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'Y_id'))

    temp_diff_summary_x = tf.summary.create_file_writer(os.path.join(base_dir,'temp_diff_x'))
    temp_diff_summary_y = tf.summary.create_file_writer(os.path.join(base_dir,'temp_diff_y'))
    
    temp_diff_summary_hist_x = tf.summary.create_file_writer(os.path.join(base_dir,'temp_diff_hist_x'))
    temp_diff_summary_hist_y = tf.summary.create_file_writer(os.path.join(base_dir,'temp_diff_hist_y'))
    
   # weights_writer = tf.summary.create_file_writer(os.path.join(base_dir,'weights'))
    
    loss_writer_id = tf.summary.create_file_writer(os.path.join(base_dir,'loss_param_id'))
    loss_writer_cycle = tf.summary.create_file_writer(os.path.join(base_dir,'loss_param_cycle'))

    print("Setup all log writers")
    
    metrics = load_metrics(config['CycleGan']['Metrics'])

    print("Loaded metrics")

    history = {
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
        "temp_diff_y_median":[]
        
    }
    diff_x=0
    diff_y=0
    pid_G = PID(config['CycleGan']['PID_P'], config['CycleGan']['PID_I'], config['CycleGan']['PID_D'], setpoint=config['CycleGan']['lambda_self'])
    pid_F = PID(config['CycleGan']['PID_P'], config['CycleGan']['PID_I'], config['CycleGan']['PID_D'], setpoint=config['CycleGan']['lambda_self'])

    print("Created PID controlers")
    for epoch in range(config['CycleGan']['epochs']):

        batches_x = data['meso_train'].batch(config['CycleGan']['batch_size'], drop_remainder=True) 
        batches_y = data['thermo_train'].batch(config['CycleGan']['batch_size'], drop_remainder=True)

        print(f"Created batches for epoch: {epoch}")
        #Anneal schedule for gumbel
        if config['CycleGan']['Generator']['use_gumbel']:
            gms_g = model.G.get_layer(index=-1)
            gms_f = model.F.get_layer(index=-1)
            tf.keras.backend.set_value(gms_g.tau,    max(0.1, np.exp(-0.0005*epoch)))
            tf.keras.backend.set_value(gms_f.tau,    max(0.1, np.exp(-0.0005*epoch)))
                
        for step, x in enumerate(zip(batches_x,batches_y)):
            #print(f"Step number: {step}")
            if config['CycleGan']['Discriminator']['use_gp']:
                losses_, logits = model.train_step_gp( batch_data = x)
                w_idx = 2
            elif config['CycleGan']['BERT']=="generator":
                losses_, logits = model.train_step_bert_generator( batch_data = x)
                metrics["bert_wt_mut_acc_x"](x[0][2], x[0][0], logits[0][0], x[0][4])
                metrics["bert_wt_mut_acc_y"](x[1][2], x[1][0], logits[0][1], x[1][4])
                w_idx = 4
            elif config['CycleGan']['BERT']=="all":
                if step%config['CycleGan']['Fractional_training']==0:
                    losses_, logits = model.train_step_bert_all( batch_data = x)
                else:
                    losses_, logits = model.train_step_generator_bert_all( batch_data = x)

                print(f"Training Step {step}")
                
                metrics["bert_wt_mut_acc_x"](x[0][2], x[0][0], logits[0][0], x[0][4])
                metrics["bert_wt_mut_acc_y"](x[1][2], x[1][0], logits[0][1], x[1][4])
                w_idx = 4
            else:
                losses_, logits = model.train_step( batch_data = x)
                w_idx = 2
                
            metrics['loss_G'](losses_["Gen_G_loss"]) 
            metrics['loss_cycle_x'](losses_["Cycle_X_loss"])
            metrics['loss_disc_y'](losses_["Disc_X_loss"])
            metrics['loss_F'](losses_["Gen_F_loss"]) 
            metrics['loss_cycle_y'](losses_["Cycle_Y_loss"])
            metrics['loss_disc_x'](losses_["Disc_Y_loss"])
            metrics['loss_id_x'](losses_["Id_X_loss"])
            metrics['loss_id_y'](losses_["Id_Y_loss"])

            metrics['acc_x'](x[0][0], logits[0][0], x[0][w_idx])
            metrics['acc_y'](x[1][0], logits[0][1], x[1][w_idx])
            metrics['cycled_acc_x'](x[0][0], logits[1][0], x[0][w_idx])
            metrics['cycled_acc_y'](x[1][0], logits[1][1], x[1][w_idx])
            metrics['id_acc_x'](x[0][0], logits[2][0], x[0][w_idx])
            metrics['id_acc_y'](x[1][0], logits[2][1], x[1][w_idx])
        
        #  ((fake_y, fake_x),(cycled_x, cycled_y), (same_x, same_y))
        #if epoch % 10 == 0 :
            # Save model
            # model.save_gan(os.path.join(result_dir,'weights','cycle_gan_model_{}'.format(epoch)))
        print(f"Step number: {step}")

        if epoch % 1 == 0 or epoch == config['CycleGan']['epochs']-1:
            val_x = data['meso_val'].batch(512, drop_remainder=False)
            val_y = data['thermo_val'].batch(512, drop_remainder=False)
            

            
            for i, x in enumerate(zip(val_x, val_y)):

                if config['CycleGan']['BERT'] in ["generator", "all"]:
                    temp_diff_y, temp_diff_x, _, _ = model.validate_step_bert(x)
                else:
                    temp_diff_y, temp_diff_x = model.validate_step(x)
                if i==0:
                    stack_temp_diff_y = temp_diff_y
                    stack_temp_diff_x = temp_diff_x
                else:
                    stack_temp_diff_y = tf.concat([stack_temp_diff_y, temp_diff_y], axis=0)
                    stack_temp_diff_x = tf.concat([stack_temp_diff_x, temp_diff_x], axis=0)
                #if i > 40:
                #    break

            with temp_diff_summary_x.as_default():
                tf.summary.scalar('temp_diff', tf.math.reduce_mean(stack_temp_diff_x), step=epoch, description = 'temp_diff_x')
            with temp_diff_summary_y.as_default():
                tf.summary.scalar('temp_diff', tf.math.reduce_mean(stack_temp_diff_y), step=epoch, description = 'temp_diff_y')

            with temp_diff_summary_hist_x.as_default():
                tf.summary.histogram('temp_diff_hist',stack_temp_diff_x, step=epoch)
            with temp_diff_summary_hist_y.as_default():
                tf.summary.histogram('temp_diff_hist',stack_temp_diff_y, step=epoch)
                    

        if epoch % 10 == 0:
            val_x = data['meso_val'].batch(128, drop_remainder=False)
            val_y = data['thermo_val'].batch(128, drop_remainder=False)

            for batch in zip(val_x, val_y):
                if config['CycleGan']['BERT'] in ["generator", "all"]:
                   #_, fake_y = model.generate_step_bert(batch)
                    fake_y = model.generate_step_bert(batch)
                else:
                    fake_y = model.generate_step(batch)
                #try:
                if flag_blast:
                    generated = callbacks.fastaFromList(fake_y, epoch, query_meso.toTempDir)
                    in_queue_meso = Thread(target=query_meso, args=(generated, epoch, mutex, ))
                    in_queue_meso.start()
                    in_queue_target = Thread(target=query_target, args=(generated, epoch, mutex, ))
                    in_queue_target.start()
                #except:
                #    print('Blast Error in callbacks (for the visuals).')
                break
            for seq in fake_y[:10]:
                print(seq[:200])
            
            #model.save_weights(os.path.join(result_dir,'weights','cycle_gan_model_epoch_{}'.format(epoch)))
	# Saving models
        identity_2_wt = metrics['acc_x'].result()
        temperature_diff_2_wt = np.median(stack_temp_diff_x.numpy())
        time_since_save = epoch - save_epoch
        if identity_2_wt >= 0.70 and temperature_diff_2_wt > 20:
            save_epoch = epoch
            model.save_gan(os.path.join(result_dir,'weights',f"epoch_{epoch}"))
        elif identity_2_wt >= 0.70 and temperature_diff_2_wt > 5 and time_since_save >= 5:
            save_epoch = epoch
            model.save_gan(os.path.join(result_dir,'weights',f"epoch_{epoch}"))


#        if metrics['acc_x'].result() >= 0.80 and np.median(stack_temp_diff_x.numpy()) > 30 and False:
#            lr_g  = tf.keras.backend.get_value(model.G.optimizer.learning_rate)
#            lr_f  = tf.keras.backend.get_value(model.F.optimizer.learning_rate)
#            dx_lr = tf.keras.backend.get_value(model.D_x.optimizer.learning_rate)
#            dy_lr = tf.keras.backend.get_value(model.D_y.optimizer.learning_rate)
#            tf.keras.backend.set_value(model.G.optimizer.learning_rate,   max(lr_g * 0.2, 0.05 * config['CycleGan']['Optimizers']["learning_rate_generator"]))
#            tf.keras.backend.set_value(model.F.optimizer.learning_rate,   max(lr_f * 0.2,  0.05 * config['CycleGan']['Optimizers']["learning_rate_generator"]))
#            tf.keras.backend.set_value(model.D_x.optimizer.learning_rate, max(dx_lr * 0.2, 0.05 * config['CycleGan']['Optimizers']["learning_rate_discriminator"]))
#            tf.keras.backend.set_value(model.D_y.optimizer.learning_rate, max(dy_lr * 0.2, 0.05 * config['CycleGan']['Optimizers']["learning_rate_discriminator"]))
           # model.save_gan(os.path.join(result_dir,'weights',f"epoch_{epoch}_special"))
            
        lr_g  = tf.keras.backend.get_value(model.G.optimizer.learning_rate)
        print(f"Learning rate is {lr_g}")
        
        if args.verbose:    
            print("Epoch: %d Loss_G: %2.4f Loss_F: %2.4f Loss_cycle_X: %2.4f Loss_cycle_Y: %2.4f Loss_D_Y: %2.4f Loss_D_X %2.4f" % 
              (epoch, float(metrics['loss_G'].result()),
               float(metrics['loss_F'].result()),
               float(metrics['loss_cycle_x'].result()),
               float(metrics['loss_cycle_y'].result()),
               float(metrics['loss_disc_y'].result()),
               float(metrics['loss_disc_x'].result())))               
            print("Epoch: %d acc trans x: %2.4f acc trans y: %2.4f acc cycled x : %2.4f acc cycled y: %2.4f" % 
              (epoch, metrics['acc_x'].result(),
               metrics['acc_y'].result(),
               metrics['cycled_acc_x'].result(),
               metrics['cycled_acc_y'].result()))
            
            gms_g = model.G.get_layer(index=-1)
            gms_f = model.F.get_layer(index=-1)
            print("G tau %1.4f F tau %1.4f" % (tf.keras.backend.get_value(gms_g.tau), tf.keras.backend.get_value(gms_f.tau)))
            print(f"Temperature diff Mesofiles: {np.median(stack_temp_diff_x.numpy())}\nTemperature diff Psychrofile: {np.median(stack_temp_diff_y.numpy())}")
            
        
        # set Id
        if dynamic:
            if config['CycleGan']['BERT'] in ["generatorr", "allr"]:
                lambda_self_G = pid_G(float(metrics["bert_wt_mut_acc_x"].result()))
                lambda_self_F = pid_F(float(metrics["bert_wt_mut_acc_y"].result()))
                
                print(f"Bert wt mut diff x : {metrics['bert_wt_mut_acc_x'].result()}")
                print(f"Bert wt mut diff y : {metrics['bert_wt_mut_acc_y'].result()}")
            
            else:
                lambda_self_G = pid_G(float(metrics['acc_x'].result()))
                lambda_self_F = pid_F(float(metrics['acc_y'].result()))

            tf.keras.backend.set_value(model.lambda_self_G,  lambda_self_G)
            print("Lambda self G", lambda_self_G)
            tf.keras.backend.set_value(model.lambda_self_F,  lambda_self_F)
            print("Lambda self F", lambda_self_F)
            #if float(metrics['acc_x'].result()) > 0.9:
                #lambda_cycle_G = tf.keras.backend.get_value(model.lambda_cycle_G)
                #lambda_id_G = tf.keras.backend.get_value(model.lambda_id_G)    
                #tf.keras.backend.set_value(model.lambda_id_G,  max(lambda_id_G * 0.1, 0.000001))
                #tf.keras.backend.set_value(model.lambda_cycle_G,  max(lambda_cycle_G * 0.99, 0.1))
                #print("lambda id G", lambda_id_G)
                #lambda_self_G = tf.keras.backend.get_value(model.lambda_self_G)
            #elif float(metrics['acc_x'].result()) < 0.7:  
                #tf.keras.backend.set_value(model.lambda_id_G,  config['CycleGan']['lambda_id'])
                #tf.keras.backend.set_value(model.lambda_cycle_G, config['CycleGan']['lambda_cycle'])
            #    tf.keras.backend.set_value(model.lambda_self_G,  config['CycleGan']['lambda_self'])

            #if float(metrics['acc_y'].result()) > 0.9:
                #lambda_cycle_F = tf.keras.backend.get_value(model.lambda_cycle_F)
                #lambda_id_F = tf.keras.backend.get_value(model.lambda_id_F)    
                #tf.keras.backend.set_value(model.lambda_id_F,  max(lambda_id_F * 0.1, 0.000001))
                #tf.keras.backend.set_value(model.lambda_cycle_F,  max(lambda_cycle_F * 0.99, 0.1))
                #lambda_self_F = tf.keras.backend.get_value(model.lambda_self_F)
                #tf.keras.backend.set_value(model.lambda_self_F,  max(lambda_self_F * 0.1, 0.000001))
                #print("Lambda self F", lambda_self_F)
            #elif float(metrics['acc_y'].result()) < 0.7:  
                #tf.keras.backend.set_value(model.lambda_id_F,  config['CycleGan']['lambda_id'])
                #tf.keras.backend.set_value(model.lambda_cycle_F, config['CycleGan']['lambda_cycle'])
                #tf.keras.backend.set_value(model.lambda_self_F,  config['CycleGan']['lambda_self'])
                
            

        
            
        # Write log file
        with G_summary_writer.as_default():
                tf.summary.scalar('loss', metrics['loss_G'].result(), step = epoch, description = 'X transform')
                tf.summary.scalar('acc', metrics['acc_x'].result(), step = epoch, description = 'X transform' )


        with F_summary_writer.as_default():
            tf.summary.scalar('loss', metrics['loss_F'].result(), step = epoch, description = 'Y transform')
            tf.summary.scalar('acc', metrics['acc_y'].result(), step = epoch, description = 'Y transform' )

        with D_x_summary_writer.as_default():         
            tf.summary.scalar('loss_discriminator', metrics['loss_disc_y'].result(), step = epoch, description = 'X discriminator')        
        with D_y_summary_writer.as_default():        
            tf.summary.scalar('loss_discriminator', metrics['loss_disc_x'].result(), step = epoch, description = 'Y discriminator')    
        with X_c_summary_writer.as_default(): 
            tf.summary.scalar('loss_cycle', metrics['loss_cycle_x'].result(), step = epoch, description = 'X cycle')
            tf.summary.scalar('acc', metrics['cycled_acc_x'].result(), step = epoch, description = 'X cycle' )         
        with Y_c_summary_writer.as_default():
            tf.summary.scalar('loss_cycle', metrics['loss_cycle_y'].result(), step = epoch, description = 'Y cycle')
            tf.summary.scalar('acc', metrics['cycled_acc_y'].result(), step = epoch, description = 'Y cycle' )
            
        with X_id_summary_writer.as_default(): 
            tf.summary.scalar('loss_cycle', metrics['loss_id_x'].result(), step = epoch, description = 'X Id')
            tf.summary.scalar('acc', metrics['id_acc_x'].result(), step = epoch, description = 'X Id' )         
        with Y_id_summary_writer.as_default():
            tf.summary.scalar('loss_cycle', metrics['loss_id_y'].result(), step = epoch, description = 'Y Id')
            tf.summary.scalar('acc', metrics['id_acc_y'].result(), step = epoch, description = 'Y Id' )

        with loss_writer_cycle.as_default():
            tf.summary.scalar('lambda', lambda_self_G, step=epoch, description = "lambda_self_G") #tf.keras.backend.get_value(model.lambda_cycle_G),step = epoch, description = 'Cycle')
        with loss_writer_id.as_default():
            tf.summary.scalar('lambda', lambda_self_F, step=epoch, description = "lambda_self_F") #tf.keras.backend.get_value(model.lambda_id_G),step = epoch, description = 'Id')
        
       # with weights_writer.as_default():
       #     for idx_layer, layer in enumerate( tf.keras.backend.get_value(model.G.trainable_variables)): 
       #         tf.summary.histogram("G_{}".format(layer.name), layer.numpy(), step=epoch, description = 'G weights')
       # with weights_writer.as_default():
       #     for idx_layer, layer in enumerate( tf.keras.backend.get_value(model.F.trainable_variables)): 
       #         tf.summary.histogram("F_{}".format(layer.name), layer.numpy(), step=epoch, description = 'F weights')
       # with weights_writer.as_default():
       #     for idx_layer, layer in enumerate( tf.keras.backend.get_value(model.D_x.trainable_variables)): 
       #         tf.summary.histogram("D_x_{}".format(layer.name), layer.numpy(), step=epoch, description = 'D x weights')
       # with weights_writer.as_default():
       #     for idx_layer, layer in enumerate( tf.keras.backend.get_value(model.D_y.trainable_variables)): 
       #         tf.summary.histogram("D_y_{}".format(layer.name), layer.numpy(), step=epoch, description = 'D y weights')
            
        # Save history object
        history["Gen_G_loss"].append(metrics['loss_G'].result().numpy())
        history["Cycle_X_loss"].append(metrics['loss_cycle_x'].result().numpy())
        history["Disc_X_loss"].append(metrics['loss_disc_x'].result().numpy())
        history["Gen_F_loss"].append(metrics['loss_F'].result().numpy())
        history["Cycle_Y_loss"].append(metrics['loss_cycle_y'].result().numpy())
        history["Disc_Y_loss"].append(metrics['loss_disc_y'].result().numpy())
        history["x_acc"].append(metrics['acc_x'].result().numpy())
        history["x_c_acc"].append(metrics['cycled_acc_x'].result().numpy())
        history["y_acc"].append(metrics['acc_y'].result().numpy())
        history["y_c_acc"].append(metrics['cycled_acc_y'].result().numpy())
        history["temp_diff_x"].append(diff_x)
        history["temp_diff_y"].append(diff_y)

        history["temp_diff_x_median"].append(np.median(stack_temp_diff_x.numpy()))
        history["temp_diff_y_median"].append(np.median(stack_temp_diff_y.numpy()))
        # Reset states
        metrics['loss_G'].reset_states()
        metrics['loss_cycle_x'].reset_states()
        metrics['loss_disc_y'].reset_states()
        metrics['loss_F'].reset_states() 
        metrics['loss_cycle_y'].reset_states()
        metrics['loss_disc_x'].reset_states()
        metrics['loss_id_y'].reset_states()
        metrics['loss_id_x'].reset_states()

        metrics['acc_x'].reset_states()
        metrics['acc_y'].reset_states()
        metrics['cycled_acc_x'].reset_states()
        metrics['cycled_acc_y'].reset_states()
        metrics['id_acc_y'].reset_states()
        metrics['id_acc_x'].reset_states()
        
        metrics["bert_wt_mut_acc_x"].reset_states()
        metrics["bert_wt_mut_acc_y"].reset_states()

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
        data_train_meso, data_val_meso = pre.load_data_bert(config["Data_meso"])
        data_train_thermo, data_val_thermo = pre.load_data_bert(config["Data_thermo"])
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