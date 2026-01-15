import tensorflow as tf
from tensorflow.keras.losses import Loss, CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.backend import softplus
from tensorflow.keras import backend as K

def _reduce_weighted_loss(loss, weight=None, eps=1e-7, num_replicas=1):
    if weight is None:
        reduced = tf.reduce_mean(loss)
    else:
        weight = tf.cast(weight, loss.dtype)
        if weight.shape.rank is not None and loss.shape.rank is not None:
            if weight.shape.rank == loss.shape.rank + 1:
                weight = tf.squeeze(weight, axis=-1)
        reduced = tf.reduce_sum(loss * weight) / (tf.reduce_sum(weight) + eps)
    return reduced / tf.cast(num_replicas, loss.dtype)

class WassersteinLoss(Loss):
    def __init__(self, ):
        super(WassersteinLoss, self).__init__()
        self.cross = CategoricalCrossentropy(from_logits=False, label_smoothing=0.0,
                                             reduction=tf.keras.losses.Reduction.NONE)
        self.num_replicas = 1
        
    def cycle_loss_fn(self, real, cycled, w=None):
        #return tf.reduce_mean(tf.abs(real - cycled))
        loss = self.cross(real, cycled, w)
        return _reduce_weighted_loss(loss, w, num_replicas=self.num_replicas)
    
    def identity_loss_fn(self, real, same, w = None):
        #loss = tf.reduce_mean(tf.abs(real - same))
        loss = self.cross(real, same, w)
        return _reduce_weighted_loss(loss, w, num_replicas=self.num_replicas)
    
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
        self.cross = CategoricalCrossentropy(from_logits=False, label_smoothing=0.0,
                                             reduction=tf.keras.losses.Reduction.NONE)
        self.bin = tf.keras.losses.BinaryCrossentropy(from_logits=True, axis=-1,
                                                      reduction=tf.keras.losses.Reduction.NONE)
        self.bin_d=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.0,axis=-1,
                                                      reduction=tf.keras.losses.Reduction.NONE)
        self.eps = tf.keras.backend.epsilon()
        self.num_replicas = 1

    def cycle_loss_fn(self, real, cycled, w=None):
        #return tf.reduce_mean(tf.abs(real - cycled))
        real = tf.cast(real, dtype="float32")
        cycled = tf.cast(cycled, dtype="float32")
        w = tf.cast(w, dtype="float32")
        loss = self.cross(real, cycled, w)
        return _reduce_weighted_loss(loss, w, eps=self.eps, num_replicas=self.num_replicas)
    
    def identity_loss_fn(self, real, same, w = None):
        #loss = tf.reduce_mean(tf.abs(real - same))
        real = tf.cast(real, dtype="float32")
        same = tf.cast(same, dtype="float32")
        w = tf.cast(w, dtype="float32")
        loss = self.cross(real, same, w)
        return _reduce_weighted_loss(loss, w, eps=self.eps, num_replicas=self.num_replicas)
    def self_loss_fn(self, real, fake, w = None):
        #loss = tf.reduce_mean(tf.abs(real - same))
        real = tf.cast(real, dtype="float32")
        fake = tf.cast(fake, dtype="float32")
        w = tf.cast(w, dtype="float32")
        loss = self.cross(real, fake, w)
        return _reduce_weighted_loss(loss, w, eps=self.eps, num_replicas=self.num_replicas)

    def _masked_mean(self, values, mask=None):
        if mask is None:
            reduced = K.mean(values)
            return reduced / tf.cast(self.num_replicas, values.dtype)
        mask = tf.cast(mask, values.dtype)
        masked_values = values * mask
        denominator = tf.reduce_sum(mask) + self.eps
        reduced = tf.reduce_sum(masked_values) / denominator
        return reduced / tf.cast(self.num_replicas, values.dtype)
    
    def generator_loss_fn(self, fake, mask=None):
        #return self.bin(tf.ones_like(fake), fake)
        fake = tf.cast(fake, dtype="float32")
        mask = tf.cast(mask, dtype="float32")
        loss = K.softplus(-fake)
        return self._masked_mean(loss, mask)#, axis=0)
    
    def discriminator_loss_fn(self, real, fake, real_mask=None, fake_mask=None):
        real = tf.cast(real, dtype="float32")
        fake = tf.cast(fake, dtype="float32")
        real_mask = tf.cast(real_mask, dtype="float32")
        fake_mask = tf.cast(fake_mask, dtype="float32")
        #L2 = tf.reduce_mean(tf.math.log(tf.ones_like(fake)-fake))
        L1 = self._masked_mean(K.softplus(-real), real_mask)#, axis=0)
        L2 = self._masked_mean(K.softplus(fake), fake_mask)#, axis=0)
        total_disc_loss = L1+L2
        #real_loss = self.bin_d(tf.ones_like(real), real)
        #generated_loss = self.bin_d(tf.zeros_like(fake), fake)
        #total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5
    
class HingeLoss(Loss):
    def __init__(self ):
        super(HingeLoss,self).__init__()
        self.cross = CategoricalCrossentropy(from_logits=False, label_smoothing=0.0,
                                             reduction=tf.keras.losses.Reduction.NONE)
        self.num_replicas = 1

    def cycle_loss_fn(self, real, cycled, w):
        loss = self.cross(real, cycled, w)
        return _reduce_weighted_loss(loss, w, num_replicas=self.num_replicas)
    
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
        self.cross = CategoricalCrossentropy(from_logits=False, label_smoothing=0.0,
                                             reduction=tf.keras.losses.Reduction.NONE)
        self.mse   = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.num_replicas = 1
        
    def cycle_loss_fn(self, real, cycled, w):
        loss = self.cross(real, cycled, w)
        return _reduce_weighted_loss(loss, w, num_replicas=self.num_replicas)
    
    # Define the loss function for the generators
    def generator_loss_fn(self, fake):
        fake_loss = self.mse(tf.ones_like(fake), fake)
        return _reduce_weighted_loss(fake_loss, num_replicas=self.num_replicas)

    # Define the loss function for the discriminators
    def discriminator_loss_fn(self, real, fake):
        real_loss = self.mse(tf.ones_like(real), real)
        fake_loss = self.mse(tf.zeros_like(fake), fake)
        total = _reduce_weighted_loss(real_loss + fake_loss, num_replicas=self.num_replicas)
        return total * 0.5
