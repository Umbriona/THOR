import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils.preprocessing import prepare_dataset
from utils import losses
from utils.metrics import BERT_prob_acc

def load_data(config):
    """ Function to load all the data """
    # Parameters
    file_thermo = config['file_thermo']
    file_meso   = config['file_meso']
    seq_length  = config['seq_length']
    max_samples = config['max_samples']
    
    thermo_train, thermo_val, n_thermo_train, n_thermo_val = prepare_dataset(file_thermo, 
                                                                             seq_length = seq_length,
                                                                             max_samples = max_samples)
    
    meso_train, meso_val, n_meso_train, n_meso_val = prepare_dataset(file_meso,
                                                                     seq_length = seq_length,
                                                                     max_samples = max_samples)

    data = {'thermo_train': thermo_train,
            'meso_train': meso_train,
            'thermo_val': thermo_val,
            'meso_val': meso_val,
            'n_thermo_train': n_thermo_train,
            'n_meso_train': n_meso_train,
            'n_thermo_val': n_thermo_val,
            'n_meso_val': n_meso_val}
    
    return data


def load_losses(config):
    if config['loss'] == 'Non-Reducing':
        loss_obj = losses.NonReduceingLoss()
    elif config['loss'] == 'Wasserstein':
        loss_obj = losses.WassersteinLoss()
    elif config['loss'] == 'Hinge':
        loss_obj = losses.HingeLoss()
    elif config['loss'] == "Mse":
        loss_obj = losses.MSE()
    else:
        raise NotImplementedError
        
    return loss_obj

def load_optimizers(config):
    lr_D   = float(config['learning_rate_discriminator'])
    lr_G   = float(config['learning_rate_generator'])
    beta_D = float(config['beta_1_discriminator'])
    beta_G = float(config['beta_1_generator'])
    optimizers = {}
    if config['optimizer_discriminator'] == 'Adam':
        optimizers['opt_D_x'] = keras.optimizers.Adam(learning_rate = lr_D, beta_1 = beta_D,clipnorm=1.0) 
        optimizers['opt_D_y'] = keras.optimizers.Adam(learning_rate = lr_D, beta_1 = beta_D,clipnorm=1.0)
    else:
        optimizers['opt_D_x'] = keras.optimizers.SGD(learning_rate = lr_D, momentum = beta_D,clipnorm=1.0) 
        optimizers['opt_D_y'] = keras.optimizers.SGD(learning_rate = lr_D, momentum = beta_D,clipnorm=1.0)
        
    if config['optimizer_generator'] == 'Adam':
        optimizers['opt_G'] = keras.optimizers.Adam(learning_rate = lr_G, beta_1 = beta_G,clipnorm=1.0) 
        optimizers['opt_F'] = keras.optimizers.Adam(learning_rate = lr_G, beta_1 = beta_G,clipnorm=1.0)
    else: 
        optimizers['opt_G'] = keras.optimizers.SGD(learning_rate = lr_G, momentum = beta_G,clipnorm=1.0) 
        optimizers['opt_F'] = keras.optimizers.SGD(learning_rate = lr_G, momentum = beta_G,clipnorm=1.0)
        
    return optimizers

def load_training_metrics():
    metrics = {}
    metrics['loss_G']       = tf.keras.metrics.Mean('loss_G', dtype=tf.float32)
    metrics['loss_cycle_x'] = tf.keras.metrics.Mean('loss_cycle_x', dtype=tf.float32)
    metrics['loss_disc_y']  = tf.keras.metrics.Mean('loss_disc_y', dtype=tf.float32)
    metrics['loss_F']       = tf.keras.metrics.Mean('loss_F', dtype=tf.float32)
    metrics['loss_cycle_y'] = tf.keras.metrics.Mean('loss_cycle_y', dtype=tf.float32)
    metrics['loss_disc_x']  = tf.keras.metrics.Mean('loss_disc_x', dtype=tf.float32)
    metrics['loss_id_x']  = tf.keras.metrics.Mean('loss_id_x', dtype=tf.float32)
    metrics['loss_id_y']  = tf.keras.metrics.Mean('loss_id_y', dtype=tf.float32)

    metrics['temp_diff_x']  = tf.keras.metrics.Mean('temp_diff_x', dtype=tf.float32)
    metrics['temp_diff_y']  = tf.keras.metrics.Mean('temp_diff_y', dtype=tf.float32)

    metrics['acc_x']        = tf.keras.metrics.CategoricalAccuracy()
    metrics['cycled_x_acc'] = tf.keras.metrics.CategoricalAccuracy()
    metrics['acc_y']        = tf.keras.metrics.CategoricalAccuracy()
    metrics['cycled_y_acc'] = tf.keras.metrics.CategoricalAccuracy()
    metrics['id_acc_y'] = tf.keras.metrics.CategoricalAccuracy()
    metrics['id_acc_x'] = tf.keras.metrics.CategoricalAccuracy()
    
    metrics['bert_wt_mut_acc_x'] = BERT_prob_acc()
    metrics['bert_wt_mut_acc_y'] = BERT_prob_acc()
    
    return metrics

def load_validation_metrics():
    metrics = {}

    metrics['temp_diff_x']  = tf.keras.metrics.Mean('temp_diff_x', dtype=tf.float32)
    metrics['temp_diff_y']  = tf.keras.metrics.Mean('temp_diff_y', dtype=tf.float32)

    metrics['acc_x']        = tf.keras.metrics.CategoricalAccuracy()
    metrics['acc_y']        = tf.keras.metrics.CategoricalAccuracy()

    metrics['likelihood_diff_x']  = tf.keras.metrics.Mean('likelihood_diff_x', dtype=tf.float32)
    metrics['likelihood_diff_y']  = tf.keras.metrics.Mean('likelihood_diff_y', dtype=tf.float32)

    
    return metrics