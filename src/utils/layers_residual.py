import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, LayerNormalization, Add, Concatenate, LeakyReLU, ReLU, Softmax, Dropout
from tensorflow.keras.regularizers import L1L2
from typeguard import typechecked

from utils.layers_new import SpectralNormalization

def linear(x):
    return x

class Linear(Layer):
    def __init__(self):
        super(Linear, self).__init__()
    def call(self, x):
        return x

class ResMod_old(Layer):
    """
    Residual layer as originaly constructed
    No Normalization is applied to the output     
    
    """
    def __init__(self, filters, size, strides=1, dilation=1, use_bias = False, spectral_norm = False):
        """
        Args: 
        
            filter: filters in residual block
            size: kernel size in residual bloc
            dilation: dilation rate in block
            stride: wether or not to use strides at the end of the block if 1< 
            use_bias: weather to use bias in the convolutions
            spectral_norm: weather or not to use spectral normalization
        
        """

        super(ResMod_old, self).__init__()
        if spectral_norm:
            self.norm = SpectralNormalization 
        else:
            self.norm = linear
            
        self.use_bias = use_bias
        
        self.conv1 = self.norm(Conv1D(filters,
                                size,
                                dilation_rate = dilation,
                                padding = 'same',
                                use_bias = self.use_bias))
        
        self.conv2 = self.norm(Conv1D(filters,
                                size,
                                dilation_rate = dilation,
                                padding = 'same',
                                use_bias = self.use_bias))
        
        self.conv3 = self.norm(Conv1D(filters,
                                size,
                                dilation_rate = dilation,
                                padding = 'same',
                                use_bias = self.use_bias))
        self.strides = False
        
        if strides > 1:
            self.strides = True
            self.conv4 = self.norm(Conv1D(filters,
                            size,
                            dilation_rate = 1,
                            strides = strides,
                            padding ='same',
                            use_bias = self.use_bias))
            
            
        self.conv  = self.norm(Conv1D(filters, 1,
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
    
class ResModPreAct(Layer):
    """
    Residual layer with full preactivation schema
    No Normalization is applied to the output     
    
    """
    def __init__(self, filters,
                 size,
                 strides=1,
                 dilation=1,
                 constrains = None,
                 l1=0.0, l2=0.0,
                 rate = 0.2,
                 use_bias=False,
                 
                 norm=False,
                 act="LReLU",
                 name = "res"):
        
        super(ResModPreAct, self).__init__(name = name)
        self.filters = filters
        self.kernel  = size
        self.strides = strides
        self.dilation= dilation
        self.constrains = constrains
        self.l1 = l1
        self.l2 = l2
        self.rate = rate
        self.use_bias = use_bias
        self.norm = norm
        


          
        if self.norm == "Layer":
            self.norm1 = LayerNormalization(axis = -1, epsilon = 1e-6, name = self.name + "_conv_1_norm")
            self.norm2 = LayerNormalization(axis = -1, epsilon = 1e-6, name = self.name + "_conv_2_norm")
        elif self.norm == "Batch":
            self.norm1 = BatchNormalization(name = self.name + "_conv_1_norm")
            self.norm2 = BatchNormalization(name = self.name + "_conv_2_norm")
        elif self.norm == "Instance":
            self.norm1 = tfa.layers.InstanceNormalization(name = self.name + "_conv_1_norm")
            self.norm2 = tfa.layers.InstanceNormalization(name = self.name + "_conv_2_norm")
        else:
            self.norm1 = Linear
            self.norm2 = Linear
            
        self.conv1 = Conv1D(self.filters, 
                                    self.kernel,
                                    dilation_rate = self.dilation,
                                    padding = 'same',
                                    use_bias = self.use_bias,
                                    kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                    name = self.name + "_conv_1")
        
        self.conv2 = Conv1D(self.filters,
                                    self.kernel,
                                    dilation_rate = self.dilation,
                                    padding = 'same',
                                    use_bias = self.use_bias,
                                    kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                    name = self.name + "_conv_2")
                    
            
        self.conv  = Conv1D(self.filters, 1,
                                    padding = 'same',
                                    use_bias = False,
                                    kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                    name = self.name + "_conv_3")
        if self.strides > 1:
            self.conv3 = Conv1D(self.filters,
                                        self.kernel,
                                        dilation_rate = 1,
                                        strides = self.strides,
                                        padding = 'same',
                                        use_bias = self.use_bias,
                                        kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                        name = self.name + "_conv_4")
    
        self.add  = Add()
        self.dout = Dropout(self.rate, name = self.name + "_dropout")
        
        if act=="LReLU":
            self.act  = LeakyReLU(0.2, name = self.name + "_activation")
        else:
            self.act = ReLU(name = self.name + "_activation")

        
    def call(self, x, training=True):
        x_in = self.conv(x)
        if self.norm == "Batch":
            x = self.conv1(self.act(self.norm1(x, training=training)))
            x = self.conv2(self.act(self.norm2(x, training=training)))
        else:
            x = self.conv1(self.act(self.norm1(x)))
            x = self.conv2(self.act(self.norm2(x)))
        
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
            'rate': self.rate,
            'norm': self.norm,
            'use_bias': self.use_bias,
            'use_spectral_norm': self.use_spectral_norm,
            'norm': self.norm
            
        })
        return config
    
class ResModPreActSN(Layer):
    """
    Residual layer with full preactivation schema
    No Normalization is applied to the output     
    
    """
    def __init__(self, filters,
                 size,
                 strides=1,
                 dilation=1,
                 constrains = None,
                 l1=0.0, l2=0.0,
                 rate = 0.2,
                 use_bias=False,
                 norm=False,
                 act="LReLU",
                 name = "res"):
        
        super(ResModPreActSN, self).__init__(name = name)
        self.filters = filters
        self.kernel  = size
        self.strides = strides
        self.dilation= dilation
        self.constrains = constrains
        self.l1 = l1
        self.l2 = l2
        self.rate = rate
        self.use_bias = use_bias
        self.norm = norm
        
        if self.norm == "Layer":
            self.norm1 = LayerNormalization(axis = -1, epsilon = 1e-6, name = self.name + "_conv_1_norm")
            self.norm2 = LayerNormalization(axis = -1, epsilon = 1e-6, name = self.name + "_conv_2_norm")
        elif self.norm == "Batch":
            self.norm1 = BatchNormalization(name = self.name + "_conv_1_norm")
            self.norm2 = BatchNormalization(name = self.name + "_conv_2_norm")
        elif self.norm == "Instance":
            self.norm1 = tfa.layers.InstanceNormalization(name = self.name + "_conv_1_norm")
            self.norm2 = tfa.layers.InstanceNormalization(name = self.name + "_conv_2_norm")
        else:
            self.norm1 = Linear
            self.norm2 = Linear
            
        self.conv1 = SpectralNormalization(Conv1D(self.filters, 
                                    self.kernel,
                                    dilation_rate = self.dilation,
                                    padding = 'same',
                                    use_bias = self.use_bias,
                                    kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                    name = self.name + "_conv_1"), name = self.name + "_conv_1_sn")
        
        self.conv2 = SpectralNormalization(Conv1D(self.filters,
                                    self.kernel,
                                    dilation_rate = self.dilation,
                                    padding = 'same',
                                    use_bias = self.use_bias,
                                    kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                    name = self.name + "_conv_2"), name = self.name + "_conv_2_sn")
                    
            
        self.conv  = SpectralNormalization(Conv1D(self.filters, 1,
                                    padding = 'same',
                                    use_bias = False,
                                    kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                    name = self.name + "_conv_3"), name = self.name + "_conv_3_sn")
        if self.strides > 1:
            self.conv3 = SpectralNormalization(Conv1D(self.filters,
                                        self.kernel,
                                        dilation_rate = 1,
                                        strides = self.strides,
                                        padding = 'same',
                                        use_bias = self.use_bias,
                                        kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                        name = self.name + "_conv_4"), name = self.name + "_conv_4_sn")
    
        self.add  = Add()
        self.dout = Dropout(self.rate, name = self.name + "_dropout")
        
        if act=="LReLU":
            self.act  = LeakyReLU(0.2, name = self.name + "_activation")
        else:
            self.act = ReLU(name = self.name + "_activation")

        
    def call(self, x, training=True):
        x_in = self.conv(x)
        if self.norm == "Batch":
            x = self.conv1(self.act(self.norm1(x, training=training)))
            x = self.conv2(self.act(self.norm2(x, training=training)))
        else:
            x = self.conv1(self.act(self.norm1(x)))
            x = self.conv2(self.act(self.norm2(x)))
        
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
            'rate': self.rate,
            'norm': self.norm,
            'use_bias': self.use_bias,
            'use_spectral_norm': self.use_spectral_norm,
            'norm': self.norm
            
        })
        return config
def residual_mod(filters,
                 size,
                 strides=1,
                 dilation=1,
                 l1=0.0, l2=0.0,
                 rate = 0.2,
                 use_bias=False,
                 norm="Instance",
                 sn = False,
                 use_dout=False,
                 act="LReLU", name = "res"):
    
    result = tf.keras.Sequential()
    
    # first normalisation
    if norm == "Batch":
        result.add(BatchNormalization())
    elif norm == "Instance":
        result.add(tfa.layers.InstanceNormalization())
    elif norm == "Layer":
        result.add(LayerNormalization())
    else:
        result.add(Linear())
    # first activation
    if act=="LReLU":
        result.add(LeakyReLU())
    else:
        result.add(ReLU())
        
    # first conv
    if use_dout:
        result.add(Dropout(0.2))
    if sn:
        result.add(SpectralNormalization(Conv1D(filters,
                      size,
                      padding = 'same',
                      use_bias = True,
                      kernel_regularizer = L1L2(l1=l1, l2=l2))))
    else:
        result.add(Conv1D(filters,
                      size,
                      padding = 'same',
                      use_bias = True,
                      kernel_regularizer = L1L2(l1=l1, l2=l2)))
    #second norm   
    if norm == "Batch":
        result.add(BatchNormalization())
    elif norm == "Instance":
        result.add(tfa.layers.InstanceNormalization())
    elif norm == "Layer":
        result.add(LayerNormalization())
    else:
        result.add(Linear())
    # second activation
    if act=="LReLU":
        result.add(LeakyReLU())
    else:
        result.add(ReLU())
        
    # second conv
    if use_dout:
        result.add(Dropout(0.2))
    if sn:
        result.add(SpectralNormalization(Conv1D(filters,
                      size,
                      padding = 'same',
                      use_bias = True,
                      kernel_regularizer = L1L2(l1=l1, l2=l2))))
    else:
        result.add(Conv1D(filters,
                      size,
                      padding = 'same',
                      use_bias = True,
                      kernel_regularizer = L1L2(l1=l1, l2=l2)))
    return result
    
    
def residual_mod_new(filters,
                 size,
                 strides=1,
                 dilation=1,
                 l1=0.0, l2=0.0,
                 rate = 0.2,
                 use_bias=False,
                 norm="Instance",
                 sn = False,
                 use_dout=False,
                 act="LReLU", name = "res"):
    
    result = tf.keras.Sequential()
    
    # first normalisation
    if norm == "Batch":
        result.add(BatchNormalization())
    elif norm == "Instance":
        result.add(tfa.layers.InstanceNormalization())
    elif norm == "Layer":
        result.add(LayerNormalization())
    
    # first activation
    if act=="LReLU":
        result.add(LeakyReLU())
    else:
        result.add(ReLU())
        
    # first conv
    if use_dout:
        result.add(Dropout(0.2))
    if sn:
        result.add(SpectralNormalization(Conv1D(filters,
                      size,
                      padding = 'same',
                      use_bias = True,
                      kernel_regularizer = L1L2(l1=l1, l2=l2))))
    else:
        result.add(Conv1D(filters,
                      size,
                      padding = 'same',
                      use_bias = True,
                      kernel_regularizer = L1L2(l1=l1, l2=l2)))
    #second norm   
    if norm == "Batch":
        result.add(BatchNormalization())
    elif norm == "Instance":
        result.add(tfa.layers.InstanceNormalization())
    elif norm == "Layer":
        result.add(LayerNormalization())
    
    # second activation
    if act=="LReLU":
        result.add(LeakyReLU())
    else:
        result.add(ReLU())
        
    # second conv
    if use_dout:
        result.add(Dropout(0.2))
    if sn:
        result.add(SpectralNormalization(Conv1D(filters,
                      size,
                      padding = 'same',
                      use_bias = True,
                      kernel_regularizer = L1L2(l1=l1, l2=l2))))
    else:
        result.add(Conv1D(filters,
                      size,
                      padding = 'same',
                      use_bias = True,
                      kernel_regularizer = L1L2(l1=l1, l2=l2)))
    return result
