import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, LayerNormalization, Add, Concatenate, LeakyReLU, Softmax, Dropout
from tensorflow.keras.regularizers import L1L2
from typeguard import typechecked
import numpy as np

POWER_ITERATIONS = 5


class SpectralNormalization(tf.keras.layers.Wrapper):
    """ From tensorflow addons https://github.com/tensorflow/addons/blob/v0.14.0/tensorflow_addons/layers/spectral_normalization.py
    Performs spectral normalization on weights.
    This wrapper controls the Lipschitz constant of the layer by
    constraining its spectral norm, which can stabilize the training of GANs.
    See [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957).
    Wrap `tf.keras.layers.Conv2D`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> conv2d = SpectralNormalization(tf.keras.layers.Conv2D(2, 2))
    >>> y = conv2d(x)
    >>> y.shape
    TensorShape([1, 9, 9, 2])
    Wrap `tf.keras.layers.Dense`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> dense = SpectralNormalization(tf.keras.layers.Dense(10))
    >>> y = dense(x)
    >>> y.shape
    TensorShape([1, 10, 10, 10])
    Args:
      layer: A `tf.keras.layers.Layer` instance that
        has either `kernel` or `embeddings` attribute.
      power_iterations: `int`, the number of iterations during normalization.
    Raises:
      AssertionError: If not initialized with a `Layer` instance.
      ValueError: If initialized with negative `power_iterations`.
      AttributeError: If `layer` does not has `kernel` or `embeddings` attribute.
    """

    @typechecked
    def __init__(self, layer: tf.keras.layers, power_iterations: int = 1, **kwargs):
        super().__init__(layer,  **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero, got "
                "`power_iterations={}`".format(power_iterations)
            )
        self.power_iterations = power_iterations
        self._initialized = False
        
    def build(self, input_shape):
        """Build `Layer`"""
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(
                "{} object has no attribute 'kernel' nor "
                "'embeddings'".format(type(self.layer).__name__)
            )

        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name=self.name + "_sn_u",
            dtype=self.w.dtype,
        )

    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            self.normalize_weights()

        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    @tf.function
    def normalize_weights(self):
        """Generate spectral normalized weights.
        This method will update the value of `self.w` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """

        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        with tf.name_scope("spectral_normalize"):
            for _ in range(self.power_iterations):
                v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
                u = tf.math.l2_normalize(tf.matmul(v, w))

            sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

            self.w.assign(tf.cast(self.w / sigma, dtype=tf.float32)) # Cast to float32 because mixed presission
            self.u.assign(tf.cast(u, dtype=tf.float32)) # Cast to float32 because mixed presission

    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}


    
class GumbelSoftmax(Layer):
    def __init__(self,temperature = 0.5, name = "gumbel", *args, **kwargs):
        super(GumbelSoftmax,self).__init__(name = name)
        
        self.temperature=temperature
        # Temperature
      #  self.tau = self.add_weight(shape=(1,),name=f"{self.name}tau", initializer=tf.keras.initializers.Constant(value=1),dtype=tf.float32,trainable=False) #temperature #tf.constant(temperature, dtype=tf.float32, name = self.name + "tau")
        
        
    def build(self, input_shape):
        self.tau = self.add_weight(shape=(),name=f"{self.name}tau", initializer=tf.keras.initializers.Constant(value=self.temperature),dtype=tf.float32,trainable=False)

    def get_config(self):   
        return  {f"temperature": self.temperature}

   # def call(self, logits):
   #     U = tf.random.uniform(tf.shape(logits), minval=0, maxval=1, dtype=tf.dtypes.float32)
   #     g = -tf.math.log(-tf.math.log(U+1e-20))
   #     nom = tf.keras.activations.softmax((g + logits)/self.tau, axis=-1)
   #     return nom

    def call(self, logits):
        logits = tf.convert_to_tensor(logits)
        compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        logits_fp32 = tf.cast(logits, compute_dtype)

        tau = tf.cast(self.tau, compute_dtype)
        u = tf.random.uniform(tf.shape(logits), minval=0.0, maxval=1.0, dtype=compute_dtype)
        g = -tf.math.log(-tf.math.log(tf.maximum(u, 1e-20)))
        y = tf.nn.softmax((logits_fp32 + g) / tau, axis=-1)

        return tf.cast(y, logits.dtype)
    
class Conv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=1, activation = None, name=None,use_bias=False, *args, **kwargs):
        self._filters = filters
        self._kernel_size =  kernel_size
        self._strides =  strides
        self.activation = activation
        self._args, self._kwargs = args, kwargs
        self.use_bias = use_bias
        self.conv = Conv1D(self._filters,
                           kernel_size=self._kernel_size,
                           strides=1,
                           activation = self.activation,
                           padding='same',
                           use_bias=self.use_bias)
        
        super(Conv1DTranspose, self).__init__(name=name)
        
    def call(self, x):
        shape = tf.shape(x)
        b_size = shape[0]
        s_size = shape[1]
        f_size = shape[2]
        n = tf.zeros_like(x)
        x = tf.stack([x,n],axis = 1)
        x = tf.transpose(x, perm=[0,2,1,3])
        x = tf.reshape(x, shape=(b_size,s_size*2,f_size,))
        x = self.conv(x)
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            '_filters': self._filters,
            '_kernel_size': self._kernel_size,
            '_strides': self._strides,
            'activation': self.activations,
            '_args': self._args,
            '_kwargs': self._kwargs,
            'use_bias': use_bias
            
        })
        return config
    
class Linear(Layer):
    def __init__(self, obj, name):
        super(Linear, self).__init__(name=name)
        self.obj = obj
    def call(x, training=True):
        x = self.obj(x)
        return x

class SelfAttentionSN(Layer):
    def __init__(self,  filters, name = "attention"):
        super(SelfAttentionSN, self).__init__(name = name)

            
        
        self.kernel_querry = SpectralNormalization(tf.keras.layers.Dense(filters, name = self.name + "_query"), name = self.name + "_atte_sn_0")
        self.kernel_key    = SpectralNormalization(tf.keras.layers.Dense(filters, name = self.name + "_key"), name = self.name + "_atte_sn_1")
        self.kernel_value  = SpectralNormalization(tf.keras.layers.Dense(filters, name = self.name + "_value"), name = self.name + "_atte_sn_2")
        self.out           = SpectralNormalization(tf.keras.layers.Dense(filters, name = self.name + "_atte_out"), name = self.name + "_atte_sn_3")
        self.num_heads = 8
        self.filters = filters
        self.depth = filters // self.num_heads
        self.gamma = self.add_weight( initializer=tf.keras.initializers.Constant(value=1), trainable=True, name = self.name + "_attention_gamma")
        self.dout = Dropout(0.3, name = self.name + "_attention_dout")
        
            
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])        
    
    def call(self, x, mask=None, training = True):
        batch_size = tf.shape(x)[0]
        
        querry = self.kernel_querry(x)
        key    = self.kernel_key(x)
        value  = self.kernel_value(x)
        
        querry = self.split_heads( querry, batch_size)
        key    = self.split_heads( key, batch_size)
        value  = self.split_heads( value, batch_size)
         
        
        attention_logits  = tf.matmul(querry, key, transpose_b = True)
        attention_weights = tf.math.softmax(attention_logits, axis=-1)
        
        attention_feature_map = tf.matmul(attention_weights, value)
        if mask is not None:
            attention_feature_map = tf.math.multiply(attention_feature_map, mask)
            
        attention_feature_map = tf.transpose(attention_feature_map, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention_feature_map, (batch_size, -1, self.filters))
        concat_attention = self.dout(concat_attention, training = training)
        out = x + self.out(concat_attention)*self.gamma
        return out, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters     
        })
        return config
    
class SelfAttention(Layer):
    def __init__(self,  filters, name = "attention"):
        super(SelfAttention, self).__init__(name = name)
      
        self.kernel_querry = tf.keras.layers.Dense(filters, name = self.name + "_query")
        self.kernel_key    = tf.keras.layers.Dense(filters, name = self.name + "_key")
        self.kernel_value  = tf.keras.layers.Dense(filters, name = self.name + "_value")
        self.out           = tf.keras.layers.Dense(filters, name = self.name + "_atte_out")
        self.num_heads = 8
        self.filters = filters
        self.depth = filters // self.num_heads
        self.gamma = self.add_weight( initializer=tf.keras.initializers.Constant(value=1), trainable=True, name = self.name + "_attention_gamma")
        self.dout = Dropout(0.3, name = self.name + "_attention_dout")
        

            
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])        
    
    def call(self, x, mask=None, training = True):
        batch_size = tf.shape(x)[0]
        
        querry = self.kernel_querry(x)
        key    = self.kernel_key(x)
        value  = self.kernel_value(x)
        
        querry = self.split_heads( querry, batch_size)
        key    = self.split_heads( key, batch_size)
        value  = self.split_heads( value, batch_size)
         
        
        attention_logits  = tf.matmul(querry, key, transpose_b = True)
        attention_weights = tf.math.softmax(attention_logits, axis=-1)
        
        attention_feature_map = tf.matmul(attention_weights, value)
        if mask is not None:
            attention_feature_map = tf.math.multiply(attention_feature_map, mask)
            
        attention_feature_map = tf.transpose(attention_feature_map, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention_feature_map, (batch_size, -1, self.filters))
        concat_attention = self.dout(concat_attention, training = training)
        out = x + self.out(concat_attention)*self.gamma
        return out, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters     
        })
        return config
    
class ResMod_old(Layer):
    def __init__(self, filters, size, strides=1, dilation=1, constrains = None):
        super(ResMod_old, self).__init__()
        
        self.conv1 = tfa.layers.SpectralNormalization(Conv1D(filters,
                                                             size,
                                                             dilation_rate = dilation,
                                                             padding = 'same',
                                                             use_bias = False))
        
        self.conv2 = tfa.layers.SpectralNormalization(Conv1D(filters,
                                                             size,
                                                             dilation_rate = dilation,
                                                             padding = 'same',
                                                             use_bias = False))
        
        self.conv3 = tfa.layers.SpectralNormalization(Conv1D(filters,
                                                             size,
                                                             dilation_rate = dilation,
                                                             padding = 'same',
                                                             use_bias = False))
        self.strides = False
        
        if strides > 1:
            self.strides = True
            self.conv4 = tfa.layers.SpectralNormalization(Conv1D(filters,
                                                                 size,
                                                                 dilation_rate = 1,
                                                                 strides = strides,
                                                                 padding ='same',
                                                                 use_bias = True))
            
            
        self.conv  = tfa.layers.SpectralNormalization(Conv1D(filters, 1,
                                                             padding = 'same',
                                                             use_bias = False))   
        self.add = Add()
        self.act = LeakyReLU(0.2)
        
    def call(self, x):
        x_in = self.conv(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.add([x, x_in])
        if self.strides:
            x = self.act(self.conv4(x)) 
        x = self.act(x)
        return x

class ResModSN(Layer):
    def __init__(self, filters, size, strides=1, dilation=1, constrains = None, l1=0.0, l2=0.0, rate = 0.2):
        super(ResModSN, self).__init__()
        self.filters = filters
        self.kernel  = size
        self.strides = strides
        self.dilation= dilation
        self.constrains = constrains
        self.l1 = l1
        self.l2 = l2
        self.rate = rate
        self.conv1 = tfa.layers.SpectralNormalization(Conv1D(self.filters, 
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        
        self.conv2 = tfa.layers.SpectralNormalization(Conv1D(self.filters,
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        
        self.conv3 = tfa.layers.SpectralNormalization(Conv1D(self.filters,
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))

        
        if self.strides > 1:
            self.conv4 = tfa.layers.SpectralNormalization(Conv1D(self.filters,
                                                                 self.kernel,
                                                                 dilation_rate = 1,
                                                                 strides = self.strides,
                                                                 padding = 'same',
                                                                 use_bias = True,
                                                                 kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
            
            
        self.conv  = tfa.layers.SpectralNormalization(Conv1D(self.filters, 1,
                                                             padding = 'same',
                                                             use_bias = False,
                                                             kernel_constraint = self.constrains,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))   
        self.add  = Add()
        self.dout = Dropout(self.rate)
        self.act  = LeakyReLU(0.1)
        self.norm1 = LayerNormalization(axis = -1)


        
    def call(self, x, training=True):
        x_in = self.conv(x)
        x = self.conv1(self.act(x))
        x = self.conv2(self.act(x))
        x = self.conv3(self.act(x))
        x = self.add([x, x_in])
        x = self.dout(x, training = training)
        if self.strides > 1:
            x = self.act(self.conv4(x)) 
        return self.norm1(x)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'dilation': self.dilation,
            'constrains': self.constrains,
            'l1': self.l1,
            'l2': self.l2,
            'rate': self.rate
            
        })
        return config
    
class ResMod(Layer):
    def __init__(self, filters, size, strides=1, dilation=1, l1=0.01, l2=0.01, rate = 0.2):
        super(ResMod, self).__init__()
        self.filters = filters
        self.kernel  = size
        self.strides = strides
        self.dilation= dilation
        self.l1 = l1
        self.l2 = l2
        self.rate = rate
        
        self.conv1 = tfa.layers.SpectralNormalization(Conv1D(self.filters, 
                            self.kernel,
                            dilation_rate = self.dilation,
                            padding = 'same',
                            use_bias = True,
                            kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        self.conv2 = tfa.layers.SpectralNormalization(Conv1D(self.filters,
                            self.kernel,
                            dilation_rate = self.dilation,
                            padding = 'same',
                            use_bias = True,
                            kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
                    
            
        self.conv  = tfa.layers.SpectralNormalization(Conv1D(self.filters, 1,
                            padding = 'same',
                            use_bias = False,
                            kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        if self.strides > 1:
            self.conv3 = tfa.layers.SpectralNormalization(Conv1D(self.filters,
                                self.kernel,
                                dilation_rate = 1,
                                strides = self.strides,
                                padding = 'same',
                                use_bias = True,
                                kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        self.add  = Add()
        self.dout = Dropout(self.rate)
        self.act  = LeakyReLU(0.2)
        self.norm1 = BatchNormalization()
        self.norm2 = BatchNormalization()


        
    def call(self, x, training=True):
        x_in = self.conv(x)

        x = self.conv1(self.act(self.norm1(x, training = training)))
        x = self.conv2(self.act(self.norm2(x, training = training)))
        
        x = self.dout(x, training = training)
        x = self.add([x, x_in])
        
        if self.strides > 1:
            x = self.act(self.conv3(x)) 
        
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'dilation': self.dilation,
            'constrains': self.constrains,
            'l1': self.l1,
            'l2': self.l2,
            'rate': self.rate
            
        })
        return config
        
    
class ResModPreActSN(Layer):
    def __init__(self, filters, size, strides=1, dilation=1, constrains = None, l1=0.0, l2=0.0, rate = 0.2):
        super(ResModPreActSN, self).__init__()
        self.filters = filters
        self.kernel  = size
        self.strides = strides
        self.dilation= dilation
        self.constrains = constrains
        self.l1 = l1
        self.l2 = l2
        self.rate = rate
        self.conv1 = SpectralNormalization(Conv1D(self.filters, 
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        
        self.conv2 = SpectralNormalization(Conv1D(self.filters,
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
                    
            
        self.conv  = SpectralNormalization(Conv1D(self.filters, 1,
                                                             padding = 'same',
                                                             use_bias = False,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        if self.strides > 1:
            self.conv3 = SpectralNormalization(Conv1D(self.filters,
                                                                 self.kernel,
                                                                 dilation_rate = 1,
                                                                 strides = self.strides,
                                                                 padding = 'same',
                                                                 use_bias = True,
                                                                 kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        self.add  = Add()
        self.dout = Dropout(self.rate)
        self.act  = LeakyReLU(0.2)



        
    def call(self, x, training=True):
        x_in = self.conv(x)

        x = self.conv1(self.act(x))
        x = self.conv2(self.act(x))
        
        x = self.dout(x, training = training)
        x = self.add([x, x_in])
        
        if self.strides > 1:
            x = self.act(self.conv3(x)) 
        
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'dilation': self.dilation,
            'l1': self.l1,
            'l2': self.l2,
            'rate': self.rate
            
        })
        return config
    
class ResModPreAct(Layer):
    def __init__(self, filters, size, strides=1, dilation=1, constrains = None, l1=0.0, l2=0.0, rate = 0.2):
        super(ResModPreAct, self).__init__()
        self.filters = filters
        self.kernel  = size
        self.strides = strides
        self.dilation= dilation
        self.constrains = constrains
        self.l1 = l1
        self.l2 = l2
        self.rate = rate
        self.conv1 = Conv1D(self.filters, 
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2))
        
        self.conv2 = Conv1D(self.filters,
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2))
                    
            
        self.conv  = Conv1D(self.filters, 1,
                                                             padding = 'same',
                                                             use_bias = False,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2))
        if self.strides > 1:
            self.conv3 = Conv1D(self.filters,
                                                                 self.kernel,
                                                                 dilation_rate = 1,
                                                                 strides = self.strides,
                                                                 padding = 'same',
                                                                 use_bias = True,
                                                                 kernel_regularizer = L1L2(l1=self.l1, l2=self.l2))
        self.add  = Add()
        self.dout = Dropout(self.rate)
        self.act  = LeakyReLU(0.2)



        
    def call(self, x, training=True):
        x_in = self.conv(x)

        x = self.conv1(self.act(x))
        x = self.conv2(self.act(x))
        
        x = self.dout(x, training = training)
        x = self.add([x, x_in])
        
        if self.strides > 1:
            x = self.act(self.conv3(x)) 
        
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'dilation': self.dilation,
            'l1': self.l1,
            'l2': self.l2,
            'rate': self.rate
            
        })
        return config
        