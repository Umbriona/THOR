#! /usr/bin/env python
import os, sys
import math
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

from utils.loaders import load_data, load_optimizers, load_losses
from utils import models_cyclegan_self_loss as models_gan
from utils import models_classifyer as models_class
from utils import callbacks
from utils import preprocessing as pre

from simple_pid import PID

mixed_precision.set_global_policy('mixed_bfloat16')
tf.config.experimental.enable_tensor_float_32_execution(True)

print("Global policy:", mixed_precision.global_policy())
print("  compute_dtype:", mixed_precision.global_policy().compute_dtype)
print("  variable_dtype:", mixed_precision.global_policy().variable_dtype)

os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

parser = argparse.ArgumentParser(""" """)

parser.add_argument('-c', '--config', type=str, default = 'config.yaml',
                   help = 'Configuration file that configures all parameters')

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")
parser.add_argument('-g', '--gpu', type=str, choices=['0', '1','0,1'], default='0,1')

args = parser.parse_args()

def reset_training_metrics(model):
        for key in model.training_metrics.keys():
            model.training_metrics[key].reset_states()

def reset_validation_metrics(model):
        for key in model.validation_metrics.keys():
            model.validation_metrics[key].reset_states()
        

class History():
    def __init__(self,model, result_dir="./"):
        self.training_history = {key:[] for key in model.training_metrics.keys()}
        self.training_history["lambda_G"] = []
        self.training_history["lambda_F"] = []
        self.training_history["step"] = []

        self.validation_history = {key:[] for key in model.validation_metrics.keys()}
        self.validation_history["step"] = []

        self.test_history = {key:[] for key in model.validation_metrics.keys()}
        self.test_history["step"] = []

        self.result_dir = result_dir

    def update_training_history(self, model, step=0):
         # Save history object
        for key in model.training_metrics.keys():
            self.training_history[key].append(model.training_metrics[key].result().numpy())
        
        self.training_history["lambda_G"].append(tf.keras.backend.get_value(model.lambda_self_G))
        self.training_history["lambda_F"].append(tf.keras.backend.get_value(model.lambda_self_F))
        self.training_history["step"].append(step)

    def write_training_history(self,):
        df = pd.DataFrame(self.training_history)
        df.to_csv(os.path.join(self.result_dir,'training_history.csv'))

    def update_validation_history(self, model, step=0):
         # Save history object
        for key in model.validation_metrics.keys():
            self.validation_history[key].append(model.validation_metrics[key].result().numpy())
        self.validation_history["step"].append(step)

    def write_validation_history(self,):
        df = pd.DataFrame(self.validation_history)
        df.to_csv(os.path.join(self.result_dir,f'validation_history.csv'))

    def update_test_history(self, model, step=0):
         # Save history object
        for key in model.validation_metrics.keys():
            self.test_history[key].append(model.validation_metrics[key].result().numpy())
        self.test_history["step"].append(step)

    def write_test_history(self,):
        df = pd.DataFrame(self.test_history)
        df.to_csv(os.path.join(self.result_dir,f'test_history.csv'))

def print_stuff(model, current_lrs, global_step):
    print(f"Acc_X is: {model.training_metrics['acc_x'].result()}  Acc_Y is: {model.training_metrics['acc_y'].result()}")

    print("Epoch: %d Loss_G: %2.4f Loss_F: %2.4f Loss_cycle_X: %2.4f Loss_cycle_Y: %2.4f Loss_D_Y: %2.4f Loss_D_X %2.4f" % 
    (global_step, float(model.training_metrics['loss_G'].result()),
    float(model.training_metrics['loss_F'].result()),
    float(model.training_metrics['loss_cycle_x'].result()),
    float(model.training_metrics['loss_cycle_y'].result()),
    float(model.training_metrics['loss_disc_y'].result()),
    float(model.training_metrics['loss_disc_x'].result()))) 

    print("Current learning rates (step {}): G={:.6e} F={:.6e} D_x={:.6e} D_y={:.6e}".format(
            global_step,
            current_lrs["G"],
            current_lrs["F"],
            current_lrs["D_x"],
            current_lrs["D_y"],
        )
        )

class WarmupDecayLRScheduler:
    """
    Step-based learning-rate scheduler with optional warmup and decay.
    Call the instance with the current (1-based) global step to update
    all optimizers in-place.

    Expected configuration format (all keys optional):

    CycleGan:
      LR_Schedule:
        warmup:
          steps: 1000
          start_factor: 0.1           # fraction of base lr to start from
          end_factor: 1.0             # fraction of base lr after warmup
        decay:
          strategy: milestone         # milestone | exponential | cosine | linear
          schedule:
            - step: 20000
              multiplier: 0.5
            - step: 40000
              multiplier: 0.1
          # For exponential:
          #   decay_steps: 5000
          #   rate: 0.8
          #   start_step: 20000
          # For cosine:
          #   decay_steps: 80000
          #   start_step: 20000
          #   alpha: 0.1               # final multiplier
          # For linear:
          #   decay_steps: 80000
          #   start_step: 20000
          #   final_multiplier: 0.1
        min_lr:
          factor: 0.01                # clamp at factor * base lr
          # or provide absolute values per optimizer:
          # G: 1e-6
          # F: 1e-6
          # D_x: 5e-6
          # D_y: 5e-6
    """

    def __init__(self, optimizers, base_learning_rates, schedule_cfg=None):
        self.optimizers = optimizers
        self.base_learning_rates = base_learning_rates
        self.schedule_cfg = schedule_cfg or {}

        warmup_cfg = self.schedule_cfg.get('warmup', {})
        self.warmup_steps = int(warmup_cfg.get('steps', 0))
        self.warmup_start_factor = float(warmup_cfg.get('start_factor', 0.0))
        self.warmup_end_factor = float(warmup_cfg.get('end_factor', 1.0))

        decay_cfg = self.schedule_cfg.get('decay', {}) or {}
        self.decay_strategy = (decay_cfg.get('strategy') or decay_cfg.get('type') or 'milestone').lower()
        self.decay_start_step = int(decay_cfg.get('start_step', max(self.warmup_steps, 0)))
        self.decay_steps = int(decay_cfg.get('decay_steps', decay_cfg.get('interval', 0) or 0))
        self.decay_rate = float(decay_cfg.get('rate', decay_cfg.get('gamma', 0.1) or 0.1))
        self.decay_alpha = float(decay_cfg.get('alpha', 0.0))
        self.decay_final_multiplier = float(decay_cfg.get('final_multiplier', decay_cfg.get('end_factor', 0.0)))
        self.decay_schedule = self._build_decay_schedule(decay_cfg)
        self.min_lrs = self._parse_min_lr(self.schedule_cfg.get('min_lr'))
        self.current_lrs = {name: base_learning_rates[name] for name in self.optimizers}

    def __call__(self, step):
        multiplier = self._compute_multiplier(step)
        lr_values = {}
        for name, optimizer in self.optimizers.items():
            base_lr = self.base_learning_rates[name]
            target_lr = max(self.min_lrs[name], base_lr * multiplier)
            tf.keras.backend.set_value(optimizer.learning_rate, target_lr)
            lr_values[name] = float(tf.keras.backend.get_value(optimizer.learning_rate))
        self.current_lrs = lr_values
        return lr_values

    def _compute_multiplier(self, step):
        step = max(1, int(step))
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            progress = step / self.warmup_steps
            return self.warmup_start_factor + progress * (self.warmup_end_factor - self.warmup_start_factor)

        strategy = self.decay_strategy
        if strategy == 'milestone':
            multiplier = 1.0
            for milestone, factor in self.decay_schedule:
                if step >= milestone:
                    multiplier *= factor
                else:
                    break
            return multiplier

        if strategy == 'exponential':
            if self.decay_steps <= 0 or step < self.decay_start_step:
                return 1.0
            num_decays = (step - self.decay_start_step) // self.decay_steps + 1
            return self.decay_rate ** num_decays

        if strategy == 'cosine':
            if self.decay_steps <= 0 or step <= self.decay_start_step:
                return 1.0
            effective_step = min(step - self.decay_start_step, self.decay_steps)
            progress = effective_step / self.decay_steps
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.decay_alpha + (1.0 - self.decay_alpha) * cosine_decay

        if strategy == 'linear':
            if self.decay_steps <= 0 or step <= self.decay_start_step:
                return 1.0
            effective_step = min(step - self.decay_start_step, self.decay_steps)
            progress = effective_step / self.decay_steps
            return 1.0 + (self.decay_final_multiplier - 1.0) * progress

        return 1.0

    def _build_decay_schedule(self, decay_cfg):
        schedule = []
        entries = decay_cfg.get('schedule')
        if entries:
            for entry in entries:
                if 'step' not in entry:
                    continue
                step = int(entry['step'])
                factor = float(entry.get('multiplier', entry.get('factor', self.decay_rate)))
                schedule.append((step, factor))
        else:
            milestones = decay_cfg.get('milestones') or []
            if isinstance(milestones, dict):
                milestones = milestones.values()
            for step in milestones:
                schedule.append((int(step), self.decay_rate))
        schedule.sort(key=lambda item: item[0])
        return schedule

    def _parse_min_lr(self, min_lr_cfg):
        min_lrs = {}
        if min_lr_cfg is None:
            for name in self.optimizers.keys():
                min_lrs[name] = 0.0
            return min_lrs

        if isinstance(min_lr_cfg, dict):
            factor = min_lr_cfg.get('factor')
            default = min_lr_cfg.get('default', 0.0)
            for name, base in self.base_learning_rates.items():
                if name in min_lr_cfg:
                    min_lrs[name] = float(min_lr_cfg[name])
                elif factor is not None:
                    min_lrs[name] = float(base) * float(factor)
                else:
                    min_lrs[name] = float(default)
            return min_lrs

        value = float(min_lr_cfg)
        for name in self.optimizers.keys():
            min_lrs[name] = value
        return min_lrs


def train(config, model, data, time, classifyer):

    #file writers
    result_dir = os.path.join(config['Results']['base_dir'],time)
    
    base_dir = os.path.join(config['Log']['base_dir'],time)

    history_obj = History(model, result_dir)
    optimizer_map = {
        "G": model.G.optimizer,
        "F": model.F.optimizer,
        "D_x": model.D_x.optimizer,
        "D_y": model.D_y.optimizer,
    }
    history = {
        "Gen_G_loss": [],
        "Cycle_X_loss": [],
        "Disc_X_loss": [],
        "Gen_F_loss": [],
        "Cycle_Y_loss": [],
        "Disc_Y_loss": [],
        "x_acc": [],
        "x_c_acc": [],
        "y_acc": [],
        "y_c_acc": [],
        "temp_diff_x": [],
        "temp_diff_y": [],
        "temp_diff_x_median": [],
        "temp_diff_y_median": [],
        "lambda_G": [],
        "lambda_F": [],
        "lr_G": [],
        "lr_F": [],
        "lr_D_x": [],
        "lr_D_y": [],
    }
    base_lrs = {
        name: float(tf.keras.backend.get_value(opt.learning_rate))
        for name, opt in optimizer_map.items()
    }
    lr_scheduler_cfg = config['CycleGan'].get('LR_Schedule', {})
    lr_scheduler = WarmupDecayLRScheduler(optimizer_map, base_lrs, lr_scheduler_cfg)
    current_lrs = lr_scheduler.current_lrs.copy()
    global_step = 0

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

    pid_G.output_limits = (0, 20)
    pid_F.output_limits = (0, 20)

    print("Created PID controlers")

    ############################################################################################################

    ############################################## Training loop ###############################################

    meso_iter = iter(data['meso_train'])
    thermo_iter = iter(data['thermo_train'])
    steps_per_epoch = int(1e6 / config['CycleGan']['batch_size'])

    for epoch in range(config['CycleGan']['epochs']):
        start_time = TIME()
        for step in range(steps_per_epoch):
            
            batches_Meso = next(meso_iter)
            batches_Thermo = next(thermo_iter)
            if step%config['CycleGan']["Fractional_training"] == 0:
                model.train_step( batch_data = [batches_Meso, batches_Thermo]) # returns losses_, logits = 
            else:
                model.train_step_generator( batch_data = [batches_Meso, batches_Thermo])
            
            model.validate_step([batches_Meso, batches_Thermo])
            global_step += 1
            current_lrs = lr_scheduler(global_step)

            if step%int(1024/config['CycleGan']['batch_size']) == 0 and step>0: #2e5/config['CycleGan']['batch_size']
                print_stuff(model, current_lrs, global_step)   

                history_obj.update_training_history(model, global_step )
                history_obj.write_training_history()

                history_obj.update_validation_history(model, global_step )
                history_obj.write_validation_history()
                    
                lambda_self_G = pid_G(float(model.validation_metrics['acc_x'].result()), dt=1) # Important that Acc measured on Validation (Not same as Training)
                lambda_self_F = pid_F(float(model.validation_metrics['acc_y'].result()), dt=1) # Important that Acc measured on Validation (Not same as Training)
                tf.keras.backend.set_value(model.lambda_self_G,  lambda_self_G)
                tf.keras.backend.set_value(model.lambda_self_F,  lambda_self_F)

                reset_training_metrics(model)
                reset_validation_metrics(model)

                gms_g = model.G.get_layer("gumbel")
                gms_f = model.F.get_layer("gumbel")
                gms_f.set_temperature(max(0.1, np.exp(-0.0001*global_step)))
                gms_g.set_temperature(max(0.1, np.exp(-0.0001*global_step)))

        reset_training_metrics(model)
        reset_validation_metrics(model)

        stop_time = TIME()
        print(f"Time for trining steps: {stop_time - start_time}")

        ################################ Validation steps #################################
        start_time = TIME()
        val_x = data['meso_val'].batch(config['CycleGan']['batch_size'], drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_y = data['thermo_val'].batch(config['CycleGan']['batch_size'], drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
        for i, x in enumerate(zip(val_x, val_y)):
            model.validate_step(x)

        history_obj.update_test_history(model, epoch )
        history_obj.write_test_history()
        reset_validation_metrics(model)

        stop_time = TIME()
        print(f"Time for validation steps: {stop_time - start_time}")

        #####################################################################################

        ############################ Generating test sequences ##############################
        if epoch % 10 == 0:
            print("Generating test sequences")
            val_x = data['meso_val'].batch(config['CycleGan']['batch_size'], drop_remainder=True)
            val_y = data['thermo_val'].batch(config['CycleGan']['batch_size'], drop_remainder=True)

            for batch in zip(val_x,val_y):
                _, fake_y, _, _ = model.generate_step_bert(batch)
                
            if flag_blast:
                generated = callbacks.fastaFromList(fake_y, epoch, query_meso.toTempDir)
                in_queue_meso = Thread(target=query_meso, args=(generated, epoch, mutex, ))
                in_queue_meso.start()
                in_queue_target = Thread(target=query_target, args=(generated, epoch, mutex, ))
                in_queue_target.start()

            for seq in fake_y[:10]:
                print(seq[:200])

	####################################### Saving models ##########################################
        if epoch % 3 == 0 and epoch > 0:
            model.save_gan(os.path.join(result_dir,'weights',f"epoch_{epoch}"), pid_G, pid_F)
    ################################################################################################

    
    return 0 #history


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
        data_train_meso, data_val_meso = pre.load_compact_data_bert(config, "Data_meso",batch_size = config['CycleGan']['batch_size'])#["Data_meso"])
        data_train_thermo, data_val_thermo = pre.load_compact_data_bert(config, "Data_thermo",batch_size = config['CycleGan']['batch_size'])#["Data_thermo"])
    else:
        data_train_meso, data_val_meso = pre.load_data(config["Data_meso"],batch_size = config['CycleGan']['batch_size'], model="cycle_gan")
        data_train_thermo, data_val_thermo = pre.load_data(config["Data_thermo"],batch_size = config['CycleGan']['batch_size'], model="cycle_gan")
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

    # Save config_file

        
    return 0
    
if __name__ == "__main__":
    main()
