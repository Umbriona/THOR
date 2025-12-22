import os
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Activation, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax, Dropout, Concatenate
from utils.layers_new import  SelfAttentionSN, GumbelSoftmax, SpectralNormalization
from utils.layers_residual import ResModPreAct
from utils import models_generator as models_gen
from utils import models_discriminator as models_dis
from utils import preprocessing as pre
from utils import models_gan_atte as gan
from utils.loaders import  load_training_metrics, load_validation_metrics

MAX_LEN=512
USE_XLA = True

class PIDWrapper(tf.Module):
    def __init__(self, pid):
        super().__init__()
        self.Kp = tf.Variable(pid.Kp, trainable=False)
        self.Ki = tf.Variable(pid.Ki, trainable=False)
        self.Kd = tf.Variable(pid.Kd, trainable=False)
        self.setpoint = tf.Variable(pid.setpoint, trainable=False)
        self.integral = tf.Variable(pid._integral, trainable=False)
        self.last_error = tf.Variable(pid._last_error, trainable=False)

    def update_pid(self, pid):
        self.Kp.assign(pid.Kp)
        self.Ki.assign(pid.Ki)
        self.Kd.assign(pid.Kd)
        self.setpoint.assign(pid.setpoint)
        self.integral.assign(pid._integral)
        self.last_error.assign(pid._last_error)

    def restore_pid(self, pid):
        pid.Kp = float(self.Kp)
        pid.Ki = float(self.Ki)
        pid.Kd = float(self.Kd)
        pid.setpoint = float(self.setpoint)
        pid._integral = float(self.integral)
        pid._last_error = float(self.last_error)

class CycleGan(tf.keras.Model):

    def __init__(self, config, callbacks=None, name = "gan", classifier=None):
        super(CycleGan, self).__init__(name = name)
        #self.G, self.F, self.D_x, self.D_y = self.load_models(config['CycleGan'])
        
        # Propagate input dimensionality based on how modalities are merged.
        # Allow different merge strategies for generators vs discriminators.
        self.concat_modalities_G = config['CycleGan'].get('concat_modalities_generator',
                                                         config['CycleGan'].get('concat_modalities', False))
        self.concat_modalities_D = config['CycleGan'].get('concat_modalities_discriminator',
                                                         self.concat_modalities_G)

        input_dim_G = 42 if self.concat_modalities_G else 21
        input_dim_D = 42 if self.concat_modalities_D else 21
        config['CycleGan']["Generator"]["input_dim"] = input_dim_G
        config['CycleGan']["Discriminator"]["input_dim"] = input_dim_D

        self.G = gan.Generator(config=config['CycleGan']["Generator"])
        self.F = gan.Generator(config=config['CycleGan']["Generator"])
        self.D_x = gan.Discriminator(config=config['CycleGan']["Discriminator"])
        self.D_y = gan.Discriminator(config=config['CycleGan']["Discriminator"])
        
        # Build models
       # inp = Input(shape=(512,21))
       # output_G = self.G(inp)
       # output_F = self.F(inp)
       # output_Dx = self.D_x(inp)
       # output_Dy = self.D_y(inp)
        
        # Build summary
        self.G.summary()
        self.F.summary()
        self.D_x.summary()
        self.D_y.summary()
        
        self.compute_dtypee = tf.keras.mixed_precision.global_policy().compute_dtype
        self.lambda_cycle_G = tf.Variable(config['CycleGan']['lambda_cycle'], dtype="float32", trainable=False)
        self.lambda_id_G    = tf.Variable(config['CycleGan']['lambda_id'], dtype="float32", trainable=False) 
        self.lambda_cycle_F = tf.Variable(config['CycleGan']['lambda_cycle'], dtype="float32", trainable=False)
        self.lambda_id_F    = tf.Variable(config['CycleGan']['lambda_id'], dtype="float32", trainable=False) 
        self.lambda_self_G  = tf.Variable(1, dtype="float32", trainable=False) 
        self.lambda_self_F  = tf.Variable(1, dtype="float32", trainable=False) 

        self.lambda_evo_G   = tf.Variable(config['CycleGan']['lambda_evo_G'], dtype="float32", trainable=False)
        self.lambda_evo_F   = tf.Variable(config['CycleGan']['lambda_evo_F'], dtype="float32", trainable=False)

        self.add  = tf.keras.layers.Add()
        
        self.classifier = classifier

        self.training_metrics = load_training_metrics()
        self.validation_metrics = load_validation_metrics()

    def _merge_modalities(self, seq, prob, concat_flag):
        """Merge one-hot sequence and pLLM probabilities either by concat or add."""
        if concat_flag:
            return tf.concat([seq, prob], axis=-1)
        return tf.math.add(seq, prob)
        
    def compile( self, loss_obj, optimizers):
        super(CycleGan, self).compile()
        
        #self.gen_G_optimizer = optimizers['opt_G']
        self.G.compile(optimizer = optimizers['opt_G'])
        #self.gen_F_optimizer = optimizers['opt_F']
        self.F.compile(optimizer = optimizers['opt_F'])
        #self.disc_X_optimizer = optimizers['opt_D_x']
        self.D_x.compile(optimizer = optimizers['opt_D_x'])
        #self.disc_Y_optimizer = optimizers['opt_D_y']
        self.D_y.compile(optimizer = optimizers['opt_D_y'])
        
        self.generator_loss_fn = loss_obj.generator_loss_fn
        self.discriminator_loss_fn = loss_obj.discriminator_loss_fn
        self.cycle_loss_fn = loss_obj.cycle_loss_fn
        self.id_loss_fn = loss_obj.identity_loss_fn
        self.self_loss_fn = loss_obj.self_loss_fn
        
    def load_models(self, config):
        """Create all models that is used in cycle gan""" 

        if config["Losses"]["loss"] == 'Non-Reducing':
            D_activation = 'sigmoid'
        else:
            D_activation = 'linear'

        vocab = config["Vocab_size"] 
        
        G    = models_gen.get_generator(config["Generator"], vocab)
        F    = models_gen.get_generator(config["Generator"], vocab)
        D_x  = models_dis.get_discriminator(config["Discriminator"], vocab, activation=D_activation)
        D_y  = models_dis.get_discriminator(config["Discriminator"], vocab, activation=D_activation)

        #G    = models_gen.Generator_res(config["Generator"], vocab, name = "Generator_thermo")
        #F    = models_gen.Generator_res(config["Generator"], vocab, name = "Generator_meso") 
        #D_x  = models_dis.Discriminator(config["Discriminator"], vocab, activation = D_activation, name = "Discriminator_thermo")
        #D_y  = models_dis.Discriminator(config["Discriminator"], vocab, activation = D_activation, name = "Discriminator_meso")

        return G, F, D_x, D_y
    
    @tf.function(jit_compile=USE_XLA)
    def train_step(self, batch_data):

        with tf.GradientTape(persistent=True) as tape:
            real_x, _, prob_x = batch_data[0]
            real_y, _, prob_y = batch_data[1]

            
            ## ONE HOT cast ##

            W_x = tf.cast(real_x >= 0, self.compute_dtypee)            
            W_x = tf.reshape(W_x,[-1,MAX_LEN,1]) # [L,1]

            W_y = tf.cast(real_y >= 0, self.compute_dtypee)           
            W_y = tf.reshape(W_y,[-1,MAX_LEN,1]) # [L,1]

            real_x = tf.one_hot(real_x, depth=21, dtype=self.compute_dtypee, off_value=0) # [L,21]
            real_y = tf.one_hot(real_y, depth=21, dtype=self.compute_dtypee, off_value=0) # [L,21]
 
            ## q_prob ###

            prob_x = tf.cast(prob_x, self.compute_dtypee) / 255.0                     # [20*MAX_LEN]
            prob_y = tf.cast(prob_y, self.compute_dtypee) / 255.0                     # [20*MAX_LEN]


            ############################

            # Calculate masks to perserve padding in fake sequences shape (None, 512, 1) -> (None, 512, 21)
            mask_x = W_x 
            mask_y = W_y
            real_x = real_x * mask_x 
            real_y = real_y * mask_y

            # Adding likelihoods to input 
            input_x_real_G = self._merge_modalities(real_x, prob_x, self.concat_modalities_G)
            input_y_real_G = self._merge_modalities(real_y, prob_y, self.concat_modalities_G)

            input_x_real_D = self._merge_modalities(real_x, prob_x, self.concat_modalities_D)
            input_y_real_D = self._merge_modalities(real_y, prob_y, self.concat_modalities_D)


            fake_y_tmp, fake_y_sm, fake_y_logits = self.G([input_x_real_G, mask_x], training=True)   # G:x -> y'

            fake_y = tf.math.multiply(fake_y_tmp, mask_x) # Preserve padding
            
            input_y_fake_G = self._merge_modalities(fake_y, prob_x, self.concat_modalities_G)
            _, cycled_x, _ = self.F([input_y_fake_G, mask_x], training=True) # Cycle: F:y' -> x

            fake_x_tmp, fake_x_sm, fake_x_logits = self.F([input_y_real_G, mask_y], training=True)   # F:y -> x'
            fake_x = tf.math.multiply(fake_x_tmp, mask_y) ##Apply mask
            input_x_fake_G = self._merge_modalities(fake_x, prob_y, self.concat_modalities_G)
            _, cycled_y, _ = self.G([input_x_fake_G, mask_y], training=True) # Cycle: G:x' -> y        

            # Identity mapping
            _, same_x, _ = self.F([input_x_real_G, mask_x], training=True)  #F:x -> x
            _, same_y, _ = self.G([input_y_real_G, mask_y], training=True)  #G:y -> y
            
            disc_real_x, final_mask_x = self.D_x([input_x_real_D, mask_x], training=True)
            disc_real_y, final_mask_y = self.D_y([input_y_real_D, mask_y], training=True)
            
            input_x_fake_D = self._merge_modalities(fake_x, prob_y, self.concat_modalities_D)
            input_y_fake_D = self._merge_modalities(fake_y, prob_x, self.concat_modalities_D)
            disc_fake_x, fake_mask_x = self.D_x([input_x_fake_D, mask_y], training=True)
            disc_fake_y, fake_mask_y = self.D_y([input_y_fake_D, mask_x], training=True)

            gen_G_loss = self.generator_loss_fn(disc_fake_y, final_mask_x)
            gen_F_loss = self.generator_loss_fn(disc_fake_x, final_mask_y)
            
            id_G_loss = self.id_loss_fn(real_y, same_y, W_y) * self.lambda_cycle_G * self.lambda_id_G
            id_F_loss = self.id_loss_fn(real_x, same_x, W_x) * self.lambda_cycle_F * self.lambda_id_F
            
            cycle_G_loss = self.cycle_loss_fn(real_y, cycled_y, W_y) * self.lambda_cycle_G 
            cycle_F_loss = self.cycle_loss_fn(real_x, cycled_x, W_x) * self.lambda_cycle_F 
            #cycle_tot_loss= cycle_G_loss + cycle_F_loss

            W_x_self = tf.math.reduce_sum(tf.math.multiply(real_x, prob_x), axis=-1)
            W_y_self = tf.math.reduce_sum(tf.math.multiply(real_y, prob_y), axis=-1)

            W_x_self = W_x_self + 1/255
            W_y_self = W_y_self + 1/255

            W_x_self = W_x_self * tf.reshape(W_x, [-1, MAX_LEN])
            W_y_self = W_y_self * tf.reshape(W_y, [-1, MAX_LEN])
            
            self_G_loss = self.self_loss_fn(real_x, fake_y_tmp, W_x_self) * self.lambda_self_G ## Need to be _tmp (avoids nan in smx)
            self_F_loss = self.self_loss_fn(real_y, fake_x_tmp, W_y_self) * self.lambda_self_F ## Need to be _tmp (avoids nan in smx)

            evo_G_loss = self.self_loss_fn(prob_x, fake_y_tmp, W_x) * self.lambda_evo_G ## Need to be _tmp (avoids nan in smx)
            evo_F_loss = self.self_loss_fn(prob_y, fake_x_tmp, W_y) * self.lambda_evo_F ## Need to be _tmp (avoids nan in smx)

            # Generator total loss
            tot_loss_G = gen_G_loss  + cycle_G_loss  + evo_G_loss + self_G_loss + id_G_loss
            tot_loss_F = gen_F_loss  + cycle_F_loss  + evo_F_loss + self_F_loss + id_F_loss

            loss_D_y = self.discriminator_loss_fn(disc_real_y, disc_fake_y, real_mask=final_mask_y, fake_mask=final_mask_x) 
            loss_D_x = self.discriminator_loss_fn(disc_real_x, disc_fake_x, real_mask=final_mask_x, fake_mask=final_mask_y) 

            eps = tf.keras.backend.epsilon()
            def _masked_disc_accuracy(logits, label, mask):
                mask = tf.cast(mask, logits.dtype)
                preds = tf.cast(logits > 0.0, logits.dtype)
                labels = tf.ones_like(preds) if label == 1.0 else tf.zeros_like(preds)
                correct = tf.cast(tf.equal(preds, labels), logits.dtype)
                correct = correct * mask
                return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + eps)

            real_acc_x = _masked_disc_accuracy(disc_real_x, 1.0, final_mask_x)
            real_acc_y = _masked_disc_accuracy(disc_real_y, 1.0, final_mask_y)
            fake_acc_x = _masked_disc_accuracy(disc_fake_x, 0.0, fake_mask_x)
            fake_acc_y = _masked_disc_accuracy(disc_fake_y, 0.0, fake_mask_y)
            disc_real_acc = 0.5 * (real_acc_x + real_acc_y)
            disc_fake_acc = 0.5 * (fake_acc_x + fake_acc_y)
            disc_avg_acc = 0.5 * (disc_real_acc + disc_fake_acc)

        grads_G_gen = tape.gradient(tot_loss_G, self.G.trainable_variables)
        grads_F_gen = tape.gradient(tot_loss_F, self.F.trainable_variables)

        # Get the gradients for the discriminators
        grads_disc_y = tape.gradient(loss_D_y, self.D_y.trainable_variables)
        grads_disc_x = tape.gradient(loss_D_x, self.D_x.trainable_variables)

        # Update the weights of the generators 
        self.G.optimizer.apply_gradients(zip(grads_G_gen, self.G.trainable_variables)) 
        self.F.optimizer.apply_gradients(zip(grads_F_gen, self.F.trainable_variables))
        
        # Update the weights of the discriminators
        self.D_y.optimizer.apply_gradients(zip(grads_disc_y, self.D_y.trainable_variables))
        self.D_x.optimizer.apply_gradients(zip(grads_disc_x, self.D_x.trainable_variables))

        self.training_metrics['loss_G'](gen_G_loss) 
        self.training_metrics['loss_cycle_x'](cycle_G_loss)
        self.training_metrics['loss_disc_y'](loss_D_x)
        self.training_metrics['loss_F'](gen_F_loss) 
        self.training_metrics['loss_cycle_y'](cycle_F_loss)
        self.training_metrics['loss_disc_x'](loss_D_y)
        self.training_metrics['loss_id_x'](id_G_loss)
        self.training_metrics['loss_id_y'](id_F_loss)
        self.training_metrics['disc_real_acc'](disc_real_acc)
        self.training_metrics['disc_fake_acc'](disc_fake_acc)
        self.training_metrics['disc_avg_acc'](disc_avg_acc)

        self.training_metrics['acc_x'](real_x, fake_y, W_x)
        self.training_metrics['acc_y'](real_y, fake_x, W_y)
        self.training_metrics['cycled_x_acc'](real_x, cycled_x, W_x)
        self.training_metrics['cycled_y_acc'](real_y, cycled_y, W_y)
        self.training_metrics['id_acc_x'](real_x, same_x, W_x)
        self.training_metrics['id_acc_y'](real_y, same_y, W_y)

    @tf.function(jit_compile=USE_XLA)
    def train_step_generator(self, batch_data):

        with tf.GradientTape(persistent=True) as tape:
            real_x, _, prob_x = batch_data[0]
            real_y, _, prob_y = batch_data[1]

            
            ## ONE HOT cast ##

            W_x = tf.cast(real_x >= 0, self.compute_dtypee)            
            W_x = tf.reshape(W_x,[-1,MAX_LEN,1]) # [L,1]

            W_y = tf.cast(real_y >= 0, self.compute_dtypee)           
            W_y = tf.reshape(W_y,[-1,MAX_LEN,1]) # [L,1]

            real_x = tf.one_hot(real_x, depth=21, dtype=self.compute_dtypee, off_value=0) # [L,21]
            real_y = tf.one_hot(real_y, depth=21, dtype=self.compute_dtypee, off_value=0) # [L,21]
 
            ## q_prob ###

            prob_x = tf.cast(prob_x, self.compute_dtypee) / 255.0                     # [20*MAX_LEN]
            prob_y = tf.cast(prob_y, self.compute_dtypee) / 255.0                     # [20*MAX_LEN]


            ############################

            # Calculate masks to perserve padding in fake sequences shape (None, 512, 1) -> (None, 512, 21)
            mask_x = W_x 
            mask_y = W_y
            real_x = real_x * mask_x 
            real_y = real_y * mask_y

            # Adding likelihoods to input 
            input_x_real_G = self._merge_modalities(real_x, prob_x, self.concat_modalities_G)
            input_y_real_G = self._merge_modalities(real_y, prob_y, self.concat_modalities_G)

            input_x_real_D = self._merge_modalities(real_x, prob_x, self.concat_modalities_D)
            input_y_real_D = self._merge_modalities(real_y, prob_y, self.concat_modalities_D)


            fake_y_tmp, fake_y_sm, fake_y_logits = self.G([input_x_real_G, mask_x], training=True)   # G:x -> y'

            fake_y = tf.math.multiply(fake_y_tmp, mask_x) # Preserve padding
            
            input_y_fake_G = self._merge_modalities(fake_y, prob_x, self.concat_modalities_G)
            _, cycled_x, _ = self.F([input_y_fake_G, mask_x], training=True) # Cycle: F:y' -> x

            fake_x_tmp, fake_x_sm, fake_x_logits = self.F([input_y_real_G, mask_y], training=True)   # F:y -> x'
            fake_x = tf.math.multiply(fake_x_tmp, mask_y) ##Apply mask
            input_x_fake_G = self._merge_modalities(fake_x, prob_y, self.concat_modalities_G)
            _, cycled_y, _ = self.G([input_x_fake_G, mask_y], training=True) # Cycle: G:x' -> y        

            # Identity mapping
            _, same_x, _ = self.F([input_x_real_G, mask_x], training=True)  #F:x -> x
            _, same_y, _ = self.G([input_y_real_G, mask_y], training=True)  #G:y -> y
            
            disc_real_x, final_mask_x = self.D_x([input_x_real_D, mask_x], training=True)
            disc_real_y, final_mask_y = self.D_y([input_y_real_D, mask_y], training=True)
            
            input_x_fake_D = self._merge_modalities(fake_x, prob_y, self.concat_modalities_D)
            input_y_fake_D = self._merge_modalities(fake_y, prob_x, self.concat_modalities_D)
            disc_fake_x, _ = self.D_x([input_x_fake_D, mask_y], training=True)
            disc_fake_y, _ = self.D_y([input_y_fake_D, mask_x], training=True)

            gen_G_loss = self.generator_loss_fn(disc_fake_y, final_mask_x)
            gen_F_loss = self.generator_loss_fn(disc_fake_x, final_mask_y)
            
            id_G_loss = self.id_loss_fn(real_y, same_y, W_y) * self.lambda_cycle_G * self.lambda_id_G
            id_F_loss = self.id_loss_fn(real_x, same_x, W_x) * self.lambda_cycle_F * self.lambda_id_F
            
            cycle_G_loss = self.cycle_loss_fn(real_y, cycled_y, W_y) * self.lambda_cycle_G 
            cycle_F_loss = self.cycle_loss_fn(real_x, cycled_x, W_x) * self.lambda_cycle_F 
            #cycle_tot_loss= cycle_G_loss + cycle_F_loss

            W_x_self = tf.math.reduce_sum(tf.math.multiply(real_x, prob_x), axis=-1)
            W_y_self = tf.math.reduce_sum(tf.math.multiply(real_y, prob_y), axis=-1)

            W_x_self = W_x_self + 1/255
            W_y_self = W_y_self + 1/255

            W_x_self = W_x_self * tf.reshape(W_x, [-1, MAX_LEN])
            W_y_self = W_y_self * tf.reshape(W_y, [-1, MAX_LEN])
            
            self_G_loss = self.self_loss_fn(real_x, fake_y_tmp, W_x_self) * self.lambda_self_G ## Need to be _tmp (avoids nan in smx)
            self_F_loss = self.self_loss_fn(real_y, fake_x_tmp, W_y_self) * self.lambda_self_F ## Need to be _tmp (avoids nan in smx)

            evo_G_loss = self.self_loss_fn(prob_x, fake_y_tmp, W_x) * self.lambda_evo_G ## Need to be _tmp (avoids nan in smx)
            evo_F_loss = self.self_loss_fn(prob_y, fake_x_tmp, W_y) * self.lambda_evo_F ## Need to be _tmp (avoids nan in smx)

            # Generator total loss
            tot_loss_G = gen_G_loss  + cycle_G_loss  + evo_G_loss + self_G_loss + id_G_loss
            tot_loss_F = gen_F_loss  + cycle_F_loss  + evo_F_loss + self_F_loss + id_F_loss

            loss_D_y = self.discriminator_loss_fn(disc_real_y, disc_fake_y, real_mask=final_mask_y, fake_mask=final_mask_x) 
            loss_D_x = self.discriminator_loss_fn(disc_real_x, disc_fake_x, real_mask=final_mask_x, fake_mask=final_mask_y) 

        grads_G_gen = tape.gradient(tot_loss_G, self.G.trainable_variables)
        grads_F_gen = tape.gradient(tot_loss_F, self.F.trainable_variables)

        # Update the weights of the generators 
        self.G.optimizer.apply_gradients(zip(grads_G_gen, self.G.trainable_variables)) 
        self.F.optimizer.apply_gradients(zip(grads_F_gen, self.F.trainable_variables))

        self.training_metrics['loss_G'](gen_G_loss) 
        self.training_metrics['loss_cycle_x'](cycle_G_loss)
        self.training_metrics['loss_disc_y'](loss_D_x)
        self.training_metrics['loss_F'](gen_F_loss) 
        self.training_metrics['loss_cycle_y'](cycle_F_loss)
        self.training_metrics['loss_disc_x'](loss_D_y)
        self.training_metrics['loss_id_x'](id_G_loss)
        self.training_metrics['loss_id_y'](id_F_loss)

        self.training_metrics['acc_x'](real_x, fake_y, W_x)
        self.training_metrics['acc_y'](real_y, fake_x, W_y)
        self.training_metrics['cycled_x_acc'](real_x, cycled_x, W_x)
        self.training_metrics['cycled_y_acc'](real_y, cycled_y, W_y)
        self.training_metrics['id_acc_x'](real_x, same_x, W_x)
        self.training_metrics['id_acc_y'](real_y, same_y, W_y)

    @tf.function(jit_compile=USE_XLA)
    def validate_step(self, batch_data):#batch_data):
        
        real_x_int, _, prob_x = batch_data[0]
        real_y_int, _, prob_y = batch_data[1]


        ## ONE HOT cast ##

        W_x = tf.cast(real_x_int >= 0, self.compute_dtypee)            # [L,1]
        W_x = tf.reshape(W_x,[-1,MAX_LEN,1]) # [L,1]

        W_y = tf.cast(real_y_int >= 0, self.compute_dtypee)           # [L,1]
        W_y = tf.reshape(W_y,[-1,MAX_LEN,1]) # [L,1]

        real_x = tf.one_hot(real_x_int, depth=21, dtype=self.compute_dtypee, off_value=0) # [L,21]
        real_y = tf.one_hot(real_y_int, depth=21, dtype=self.compute_dtypee, off_value=0) # [L,21]


        ## q_prob ###

        prob_x = tf.cast(prob_x, self.compute_dtypee) / 255.0                     # [20*MAX_LEN]
        prob_y = tf.cast(prob_y, self.compute_dtypee) / 255.0                     # [20*MAX_LEN]



        ############################

        input_x_real = self._merge_modalities(real_x, prob_x, self.concat_modalities_G)
        input_y_real = self._merge_modalities(real_y, prob_y, self.concat_modalities_G)
        
        fake_y, fake_y_sm, fake_y_logits = self.G([input_x_real, W_x], training=False)
        fake_x, fake_x_sm, fake_x_logits = self.F([input_y_real, W_y], training=False)

        # mask fakes
        mask_x = W_x #tf.repeat(W_x, 21, axis=-1)
        mask_y = W_y #tf.repeat(W_y, 21, axis=-1)

        fake_y = tf.math.multiply(fake_y, mask_x)
        fake_x = tf.math.multiply(fake_x, mask_y)
        # Ensure classifier receives one-hot encoded generated sequences
        fake_y_int = tf.argmax(fake_y, axis=-1)
        fake_y_oh = tf.one_hot(fake_y_int, depth=21, dtype=self.compute_dtypee, off_value=0)
        fake_y_oh = tf.math.multiply(fake_y_oh, mask_x)

        fake_x_int = tf.argmax(fake_x, axis=-1)
        fake_x_oh = tf.one_hot(fake_x_int, depth=21, dtype=self.compute_dtypee, off_value=0)
        fake_x_oh = tf.math.multiply(fake_x_oh, mask_y)

        temp_real_x = self.classifier(real_x,training=False)
        temp_fake_y = self.classifier(fake_y_oh,training=False)
        temp_diff_x = tf.math.subtract(temp_fake_y,temp_real_x)

        temp_real_y = self.classifier(real_y ,training=False)
        temp_fake_x = self.classifier(fake_x_oh,training=False)
        temp_diff_y = tf.math.subtract(temp_fake_x,temp_real_y)

        self.validation_metrics['temp_diff_x'](temp_diff_x)
        self.validation_metrics['temp_diff_y'](temp_diff_y)

        self.validation_metrics['acc_x'](real_x, fake_y, W_x)
        self.validation_metrics['acc_y'](real_y, fake_x, W_y)

        fake_y_int = tf.argmax( fake_y, axis=-1, output_type=tf.int32)
        fake_x_int = tf.argmax( fake_x, axis=-1, output_type=tf.int32)

        likelihood_real_x = tf.math.reduce_mean(tf.gather(prob_x, indices=real_x_int , axis=2, batch_dims=2) * tf.reshape(W_x, (-1, MAX_LEN)))
        likelihood_fake_y = tf.math.reduce_mean(tf.gather(prob_x, indices=fake_y_int , axis=2, batch_dims=2) * tf.reshape(W_x, (-1, MAX_LEN)))

        likelihood_real_y = tf.math.reduce_mean(tf.gather(prob_y, indices=real_y_int , axis=2, batch_dims=2) * tf.reshape(W_y, (-1, MAX_LEN)))
        likelihood_fake_x = tf.math.reduce_mean(tf.gather(prob_y, indices=fake_x_int , axis=2, batch_dims=2) * tf.reshape(W_y, (-1, MAX_LEN)))

        self.validation_metrics['likelihood_diff_x'](likelihood_fake_y - likelihood_real_x)
        self.validation_metrics['likelihood_diff_y'](likelihood_fake_x - likelihood_real_y)

    def generate_step_bert(self, batch_data):
        real_x, _, prob_x = batch_data[0]
        real_y_int, _, prob_y = batch_data[1]
        
        #real_x, ids_batch, prob_x = batch_data
        ## ONE HOT cast ##
        W_x = tf.cast(real_x >= 0, self.compute_dtypee)            # [L,1]
        W_x = tf.reshape(W_x,[-1,MAX_LEN,1])

        real_x = tf.one_hot(real_x, depth=21, dtype=self.compute_dtypee, off_value=0) # [L,21]
        
        ## q_prob ###
        prob_x = tf.cast(prob_x, self.compute_dtypee) / 255.0                     # [20*MAX_LEN]

        ############################

        # Adding likelihoods to input 
        input_x_real = self._merge_modalities(real_x, prob_x, self.concat_modalities_G)
        fake_y, fake_y_sm, fake_y_logits = self.G([input_x_real, W_x], training=False)

        #################### Predict temperature ##################
        mask_x = W_x #tf.repeat(W_x, 21, axis=-1)
        fake_y = tf.math.multiply(fake_y, mask_x)
        # Ensure classifier receives one-hot encoded generated sequences
        fake_y_int = tf.argmax(fake_y, axis=-1)
        fake_y_oh = tf.one_hot(fake_y_int, depth=21, dtype=self.compute_dtypee, off_value=0)
        fake_y_oh = tf.math.multiply(fake_y_oh, mask_x)

        temp_real_x = self.classifier(real_x ,training=False)
        temp_fake_y = self.classifier(fake_y_oh,training=False)

        seqs_fake = []
        temps_fake = []
        seqs_real = []
        temps_real = []
        ids = []

        for  seq_fake, w in zip(list(tf.math.argmax(fake_y,axis=-1).numpy()), list(W_x.numpy())):
            seqs_fake.append(pre.convert_table(seq_fake, tf.reshape(w, shape=(512,))))
        for  seq_true, w in zip(list(tf.math.argmax(real_x,axis=-1).numpy()), list(W_x.numpy())):
            seqs_real.append(pre.convert_table(seq_true, tf.reshape(w, shape=(512,))))  
        #for id_ in ids_batch:
        #    ids.append(id_.numpy().decode('utf-8'))
        for temp_real, temp_fake in zip(temp_real_x, temp_fake_y):
            temps_fake.append(temp_fake[0])
            temps_real.append(temp_real[0])
        
        return  seqs_real, seqs_fake, temps_real, temps_fake 

    def generate_step_bert_inference(self, batch_data):

        real_x, ids_batch, prob_x = batch_data[0]
        ## ONE HOT cast ##
        W_x = tf.cast(real_x >= 0, self.compute_dtypee)            # [L,1]
        W_x = tf.reshape(W_x,[-1,MAX_LEN,1])

        real_x = tf.one_hot(real_x, depth=21, dtype=self.compute_dtypee, off_value=0) # [L,21]
        
        ## q_prob ###
        prob_x = tf.cast(prob_x, self.compute_dtypee) / 255.0                     # [20*MAX_LEN]

        ############################

        # Adding likelihoods to input 
        input_x_real = self._merge_modalities(real_x, prob_x, self.concat_modalities_G)
        fake_y, fake_y_sm, fake_y_logits = self.G([input_x_real, W_x], training=False)

        #################### Predict temperature ##################
        mask_x = W_x #tf.repeat(W_x, 21, axis=-1)
        fake_y = tf.math.multiply(fake_y, mask_x)
        fake_y_int = tf.argmax(fake_y, axis=-1)
        fake_y_oh = tf.one_hot(fake_y_int, depth=21, dtype=self.compute_dtypee, off_value=0)
        fake_y_oh = tf.math.multiply(fake_y_oh, mask_x)

        temp_real_x = self.classifier(real_x ,training=False)
        temp_fake_y = self.classifier(fake_y_oh,training=False)

        seqs_fake = []
        temps_fake = []
        seqs_real = []
        temps_real = []
        ids = []

        for  seq_fake, w in zip(list(tf.math.argmax(fake_y,axis=-1).numpy()), list(W_x.numpy())):
            seqs_fake.append(pre.convert_table(seq_fake, tf.reshape(w, shape=(512,))))
        for  seq_true, w in zip(list(tf.math.argmax(real_x,axis=-1).numpy()), list(W_x.numpy())):
            seqs_real.append(pre.convert_table(seq_true, tf.reshape(w, shape=(512,))))  
        for id_ in ids_batch:
            ids.append(id_.numpy().decode('utf-8'))
        for temp_real, temp_fake in zip(temp_real_x, temp_fake_y):
            temps_fake.append(temp_fake[0])
            temps_real.append(temp_real[0])
        
        return  ids, seqs_real, seqs_fake, temps_real, temps_fake 

    def generate_MAX_likelihood_step_bert_inference(self, batch_data, return_probs=False):

        real_x, ids_batch, prob_x = batch_data[0]
        ## ONE HOT cast ##
        W_x = tf.cast(real_x >= 0, self.compute_dtypee)            # [L,1]
        W_x = tf.reshape(W_x,[-1,MAX_LEN,1])

        real_x = tf.one_hot(real_x, depth=21, dtype=self.compute_dtypee, off_value=0) # [L,21]
        
        ## q_prob ###
        prob_x = tf.cast(prob_x, self.compute_dtypee) / 255.0                     # [20*MAX_LEN]

        ############################

        # Adding likelihoods to input 
        input_x_real = self._merge_modalities(real_x, prob_x, self.concat_modalities_G)
        # Use the softmax output (second return) from G and pick max-likelihood tokens.
        _, fake_y, fake_y_logits = self.G([input_x_real, W_x], training=False)

        #################### Predict temperature ##################
        mask_x = W_x #tf.repeat(W_x, 21, axis=-1)
        fake_y = tf.math.multiply(fake_y, mask_x)
        fake_y_int = tf.argmax(fake_y, axis=-1)
        fake_y_oh = tf.one_hot(fake_y_int, depth=21, dtype=self.compute_dtypee, off_value=0)
        fake_y_oh = tf.math.multiply(fake_y_oh, mask_x)

        temp_real_x = self.classifier(real_x ,training=False)
        temp_fake_y = self.classifier(fake_y_oh,training=False)

        seqs_fake = []
        temps_fake = []
        seqs_real = []
        temps_real = []
        ids = []
        raw_probs = [] if return_probs else None

        for  seq_fake, w in zip(list(tf.math.argmax(fake_y,axis=-1).numpy()), list(W_x.numpy())):
            seqs_fake.append(pre.convert_table(seq_fake, tf.reshape(w, shape=(512,))))
        if return_probs:
            # Collect per-position softmax probabilities (trimmed to non-padded tokens).
            for probs, w in zip(list(fake_y.numpy()), list(W_x.numpy())):
                length = int(np.sum(w))  # w is shape (512,1) with 1s on real tokens
                raw_probs.append(probs[:length].tolist())
        for  seq_true, w in zip(list(tf.math.argmax(real_x,axis=-1).numpy()), list(W_x.numpy())):
            seqs_real.append(pre.convert_table(seq_true, tf.reshape(w, shape=(512,))))  
        for id_ in ids_batch:
            ids.append(id_.numpy().decode('utf-8'))
        for temp_real, temp_fake in zip(temp_real_x, temp_fake_y):
            temps_fake.append(temp_fake[0])
            temps_real.append(temp_real[0])
        
        if return_probs:
            return ids, seqs_real, seqs_fake, temps_real, temps_fake, raw_probs
        return  ids, seqs_real, seqs_fake, temps_real, temps_fake 


    
    def save_gan(self, path, pid_G, pid_F):
        self.G.save(os.path.join(path,"generator_G.h5"))
        self.D_x.save(os.path.join(path,"discriminator_x.h5"))
        self.D_y.save(os.path.join(path,"discriminator_y.h5"))
        self.F.save(os.path.join(path,"generator_F.h5"))

        with open(os.path.join(path, "optimizer_G.pkl"), "wb") as f:
            pickle.dump([v.numpy() for v in self.G.optimizer.variables], f)

        with open(os.path.join(path, "optimizer_Dx.pkl"), "wb") as f:
            pickle.dump([v.numpy() for v in self.D_x.optimizer.variables], f)

        with open(os.path.join(path, "optimizer_Dy.pkl"), "wb") as f:
            pickle.dump([v.numpy() for v in self.D_y.optimizer.variables], f)

        with open(os.path.join(path, "optimizer_F.pkl"), "wb") as f:
            pickle.dump([v.numpy() for v in self.F.optimizer.variables], f)

        with open(os.path.join(path,"PID_G.pkl"), "wb") as f:
            pickle.dump(pid_G.__dict__, f)
        with open(os.path.join(path,"PID_F.pkl"), "wb") as f:
            pickle.dump(pid_F.__dict__, f)

        print(f"Saved gan model at {path}")
    
    def load_gan(self, path, pid_G, pid_F):
        print(f"Load gan model at {path}")
        self.D_x.load_weights(os.path.join(path,"discriminator_x.h5"))
        self.D_y.load_weights(os.path.join(path,"discriminator_y.h5"))
        self.G.load_weights(os.path.join(path,"generator_G.h5"))
        self.F.load_weights(os.path.join(path,"generator_F.h5"))

      #  with open(os.path.join(path, "optimizer_G.pkl"), "rb") as f:
      #      self.G.optimizer.set_weights(pickle.load(f))
      #  with open(os.path.join(path, "optimizer_F.pkl"), "rb") as f:
      #      self.F.optimizer.set_weights(pickle.load(f))
      #  with open(os.path.join(path, "optimizer_Dx.pkl"), "rb") as f:
      #      self.D_x.optimizer.set_weights(pickle.load(f))
      #  with open(os.path.join(path, "optimizer_Dy.pkl"), "rb") as f:
      #      self.D_y.optimizer.set_weights(pickle.load(f))

      #  with open(os.path.join(path,"PID_G.pkl"), "rb") as f:
      #      state = pickle.load(f)
      #  pid_G.__dict__.update(state)

       # with open(os.path.join(path,"PID_F.pkl"), "rb") as f:
       #     state = pickle.load(f)
       # pid_F.__dict__.update(state)
