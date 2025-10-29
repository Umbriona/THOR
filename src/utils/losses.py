import tensorflow as tf
from tensorflow.keras.losses import Loss, CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.backend import softplus
from tensorflow.keras import backend as K

class WassersteinLoss(Loss):
    def __init__(self, ):
        super(WassersteinLoss, self).__init__()
        self.cross = CategoricalCrossentropy(from_logits=False, label_smoothing=0.0)
        
    def cycle_loss_fn(self, real, cycled, w=None):
        #return tf.reduce_mean(tf.abs(real - cycled))
        return self.cross( real, cycled, w)
    
    def identity_loss_fn(self, real, same, w = None):
        #loss = tf.reduce_mean(tf.abs(real - same))
        loss = self.cross( real, same, w)
        return loss
    
    # Define the loss function for the generators
    def generator_loss_fn(self, fake):
        return -tf.reduce_mean(fake)

    # Define the loss function for the discriminators
    def discriminator_loss_fn(self, real, fake):
        real_loss = tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)
        return fake_loss - real_loss
    
class NonReduceingLoss(Loss):
    def __init__(self, ):
        super(NonReduceingLoss, self).__init__()
        self.cross = CategoricalCrossentropy(from_logits=False, label_smoothing=0.0)#, reduction=tf.keras.losses.Reduction.NONE)
        self.bin = tf.keras.losses.BinaryCrossentropy(from_logits=True, axis=-1)#, reduction=tf.keras.losses.Reduction.NONE)
        self.bin_d=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.0,axis=-1)#, reduction=tf.keras.losses.Reduction.NONE)

    def cycle_loss_fn(self, real, cycled, w=None):
        #return tf.reduce_mean(tf.abs(real - cycled))
        return self.cross( real, cycled, w)
    
    def identity_loss_fn(self, real, same, w = None):
        #loss = tf.reduce_mean(tf.abs(real - same))
        loss = self.cross( real, same, w)
        return loss
    def self_loss_fn(self, real, fake, w = None):
        #loss = tf.reduce_mean(tf.abs(real - same))
        loss = self.cross( real, fake, w)
        return loss
    
    def generator_loss_fn(self, fake):
        #return self.bin(tf.ones_like(fake), fake)
        return K.mean(K.softplus(-fake))#, axis=0)
    
    def discriminator_loss_fn(self, real, fake):
        #L1 = tf.reduce_mean(tf.math.log(real))
        #L2 = tf.reduce_mean(tf.math.log(tf.ones_like(fake)-fake))
        L1 = K.mean(K.softplus(-real))#, axis=0)
        L2 = K.mean(K.softplus(fake))#, axis=0)
        total_disc_loss = L1+L2
        #real_loss = self.bin_d(tf.ones_like(real), real)
        #generated_loss = self.bin_d(tf.zeros_like(fake), fake)
        #total_disc_loss = real_loss + generated_loss
        return total_disc_loss #* 0.5
    
class HingeLoss(Loss):
    def __init__(self ):
        super(HingeLoss,self).__init__()
        self.cross = CategoricalCrossentropy(from_logits=False, label_smoothing=0.0)

    def cycle_loss_fn(self, real, cycled, w):
        return self.cross( real, cycled, w)
    
    # Define the loss function for the generators
    def generator_loss_fn(self, fake):
        return -1 * K.mean(fake, axis=0)

    # Define the loss function for the discriminators
    def discriminator_loss_fn(self, real, fake):
        loss = K.mean(K.relu(1. - real),axis=0)
        loss += K.mean(K.relu(1. + fake),axis=0)
        return loss
    
class MSE(Loss):
    def __init__(self ):
        super(MSE, self).__init__()
        self.cross = CategoricalCrossentropy(from_logits=False, label_smoothing=0.0)
        self.mse   = tf.keras.losses.MeanSquaredError()
        
    def cycle_loss_fn(self, real, cycled, w):
        return self.cross( real, cycled, w)
    
    # Define the loss function for the generators
    def generator_loss_fn(self, fake):
        fake_loss = self.mse(tf.ones_like(fake), fake)
        return fake_loss

    # Define the loss function for the discriminators
    def discriminator_loss_fn(self, real, fake):
        real_loss = self.mse(tf.ones_like(real), real)
        fake_loss = self.mse(tf.zeros_like(fake), fake)
        return (real_loss + fake_loss) * 0.5


