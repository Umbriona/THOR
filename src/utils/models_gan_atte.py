import os
import numpy as np
import math
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Activation, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax, Dropout, Concatenate, Layer, Add, ReLU
from tensorflow.keras.regularizers import L1L2
from utils.layers_new import GumbelSoftmax, SpectralNormalization


def downsample(filters, size,  norm="instance", act= "Relu"):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv1D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    if act=="LReLU":
        result.add(LeakyReLU(0.2))
    else:
        result.add(ReLU())

    if norm=="batch":
        result.add(tf.keras.layers.BatchNormalization())
    #elif norm=="instance":
    #    result.add(tfa.layers.InstanceNormalization())
    elif norm == "layer":
        result.add(LayerNormalization(axis = -1, epsilon = 1e-6))


    return result

def downsampleSN(filters, size,  norm="instance", act= "Relu"):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(SpectralNormalization(tf.keras.layers.Conv1D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)))

    if act=="LReLU":
        result.add(LeakyReLU(0.2))
    else:
        result.add(ReLU())

    if norm=="batch":
        result.add(tf.keras.layers.BatchNormalization())
    #elif norm=="instance":
    #    result.add(tfa.layers.InstanceNormalization())
    elif norm == "layer":
        result.add(LayerNormalization(axis = -1, epsilon = 1e-6))


    return result

class UAttention(Layer):
    def __init__(self,  filters, heads=4, name = "attention",  act = "Relu"):
        super(UAttention, self).__init__(name = name)
      
        self.kernel_querry = tf.keras.layers.Dense(filters, name = self.name + "_query")
        self.kernel_key    = tf.keras.layers.Dense(filters, name = self.name + "_key")
        self.kernel_value  = tf.keras.layers.Dense(filters, name = self.name + "_value")
        self.out           = tf.keras.layers.Dense(filters, name = self.name + "_atte_out")
        self.num_heads = heads
        self.filters = filters
        self.depth = filters // self.num_heads
        self.scale = math.sqrt(float(self.depth))
        self.gamma = self.add_weight( initializer=tf.keras.initializers.Constant(value=1), trainable=True, name = self.name + "_attention_gamma")
        self.dout = Dropout(0.3, name = self.name + "_attention_dout")
        if act=="LReLU":
            self.act  = LeakyReLU(0.2, name = self.name + "_activation")
        else:
            self.act = ReLU(name = self.name + "_activation")

            
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])        
    
    def call(self, x, k, mask=None, training = True):
        batch_size = tf.shape(x)[0]
        
        querry = self.kernel_querry(x)
        querry = self.act(querry)
        key    = self.kernel_key(k)
        key    = self.act(key)
        value  = self.kernel_value(k)
        value  = self.act(value)

        querry = self.split_heads( querry, batch_size)
        key    = self.split_heads( key, batch_size)
        value  = self.split_heads( value, batch_size)
         
        
        attention_logits  = tf.matmul(querry, key, transpose_b = True) / self.scale
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

def upsample(filters, size, apply_dropout=False,act="ReLU",  norm="instance"):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv1DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    if act=="LReLU":
        result.add(LeakyReLU(0.2))
    else:
        result.add(ReLU())

    if norm=="batch":
        result.add(tf.keras.layers.BatchNormalization())
    #elif norm=="instance":
    #    result.add(tfa.layers.InstanceNormalization())
    elif norm == "layer":
        result.add(LayerNormalization(axis = -1, epsilon = 1e-6))

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    return result




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
        


          
        if self.norm == "layer":
            self.norm1 = LayerNormalization(axis = -1, epsilon = 1e-6, name = self.name + "_conv_1_norm")
            self.norm2 = LayerNormalization(axis = -1, epsilon = 1e-6, name = self.name + "_conv_2_norm")
        elif self.norm == "batch":
            self.norm1 = BatchNormalization(name = self.name + "_conv_1_norm")
            self.norm2 = BatchNormalization(name = self.name + "_conv_2_norm")
        #elif self.norm == "instance":
        #    self.norm1 = tfa.layers.InstanceNormalization(name = self.name + "_conv_1_norm")
        #    self.norm2 = tfa.layers.InstanceNormalization(name = self.name + "_conv_2_norm")

            
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
        if self.norm == "batch":
            x = self.conv1(self.act(self.norm1(x, training=training)))
            x = self.conv2(self.act(self.norm2(x, training=training)))
        elif self.norm == "layer":
            x = self.conv1(self.act(self.norm1(x)))
            x = self.conv2(self.act(self.norm2(x)))
        else:
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
            'rate': self.rate,
            'norm': self.norm,
            'use_bias': self.use_bias,
            'act': self.act
            
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
        


          
        if self.norm == "layer":
            self.norm1 = LayerNormalization(axis = -1, epsilon = 1e-6, name = self.name + "_conv_1_norm")
            self.norm2 = LayerNormalization(axis = -1, epsilon = 1e-6, name = self.name + "_conv_2_norm")
        elif self.norm == "batch":
            self.norm1 = BatchNormalization(name = self.name + "_conv_1_norm")
            self.norm2 = BatchNormalization(name = self.name + "_conv_2_norm")
        #elif self.norm == "instance":
        #    self.norm1 = tfa.layers.InstanceNormalization(name = self.name + "_conv_1_norm")
        #    self.norm2 = tfa.layers.InstanceNormalization(name = self.name + "_conv_2_norm")

            
        self.conv1 = SpectralNormalization(Conv1D(self.filters, 
                                    self.kernel,
                                    dilation_rate = self.dilation,
                                    padding = 'same',
                                    use_bias = self.use_bias,
                                    kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                    name = self.name + "_conv_1"))
        
        self.conv2 = SpectralNormalization(Conv1D(self.filters,
                                    self.kernel,
                                    dilation_rate = self.dilation,
                                    padding = 'same',
                                    use_bias = self.use_bias,
                                    kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                    name = self.name + "_conv_2"))
                    
            
        self.conv  = SpectralNormalization(Conv1D(self.filters, 1,
                                    padding = 'same',
                                    use_bias = False,
                                    kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                    name = self.name + "_conv_3"))
        if self.strides > 1:
            self.conv3 = SpectralNormalization(Conv1D(self.filters,
                                        self.kernel,
                                        dilation_rate = 1,
                                        strides = self.strides,
                                        padding = 'same',
                                        use_bias = self.use_bias,
                                        kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                        name = self.name + "_conv_4"))
    
        self.add  = Add()
        self.dout = Dropout(self.rate, name = self.name + "_dropout")
        
        if act=="LReLU":
            self.act  = LeakyReLU(0.2, name = self.name + "_activation")
        else:
            self.act = ReLU(name = self.name + "_activation")

        
    def call(self, x, training=True):
        x_in = self.conv(x)
        if self.norm == "batch":
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
            'norm': self.norm
            
        })
        return config    
    
    
class SelfAttention(Layer):
    def __init__(self,  filters, name = "attention", act = "Relu"):
        super(SelfAttention, self).__init__(name = name)
      
        self.kernel_querry = tf.keras.layers.Dense(filters, name = self.name + "_query")
        self.kernel_key    = tf.keras.layers.Dense(filters, name = self.name + "_key")
        self.kernel_value  = tf.keras.layers.Dense(filters, name = self.name + "_value")
        self.out           = tf.keras.layers.Dense(filters, name = self.name + "_atte_out")
        self.num_heads = 4
        
        self.filters = filters
        self.depth = filters // self.num_heads
        self.scale = math.sqrt(float(self.depth))
        self.gamma = self.add_weight( initializer=tf.keras.initializers.Constant(value=1), trainable=True, name = self.name + "_attention_gamma")
        self.dout = Dropout(0.3, name = self.name + "_attention_dout")

        if act=="LReLU":
            self.act = LeakyReLU(0.2, name = self.name + "_activation")
        else:
            self.act = ReLU(name = self.name + "_activation")

            
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])        
    
    def call(self, x, mask=None, training = True):
        batch_size = tf.shape(x)[0]
        
        querry = self.kernel_querry(x)
        querry = self.act(querry)
        key    = self.kernel_key(x)
        key    = self.act(key)
        value  = self.kernel_value(x)
        value  = self.act(value)

        querry = self.split_heads( querry, batch_size)
        key    = self.split_heads( key, batch_size)
        value  = self.split_heads( value, batch_size)
         
        
        attention_logits  = tf.matmul(querry, key, transpose_b = True) / self.scale
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
    

    
class SelfAttentionSN(Layer):
    def __init__(self,  filters, heads= 4, name = "attention", act = "Relu"):
        super(SelfAttentionSN, self).__init__(name = name)
      
        self.kernel_querry = SpectralNormalization(tf.keras.layers.Dense(filters, name = self.name + "_query"))
        self.kernel_key    = SpectralNormalization(tf.keras.layers.Dense(filters, name = self.name + "_key"))
        self.kernel_value  = SpectralNormalization(tf.keras.layers.Dense(filters, name = self.name + "_value"))
        self.out           = SpectralNormalization(tf.keras.layers.Dense(filters, name = self.name + "_atte_out"))
        self.num_heads = heads
        
        self.filters = filters
        self.depth = filters // self.num_heads
        self.scale = math.sqrt(float(self.depth))
        self.gamma = self.add_weight( initializer=tf.keras.initializers.Constant(value=1), trainable=True, name = self.name + "_attention_gamma")
        self.dout = Dropout(0.3, name = self.name + "_attention_dout")
        
        if act=="LReLU":
            self.act = LeakyReLU(0.2, name = self.name + "_activation")
        else:
            self.act = ReLU(name = self.name + "_activation")

            
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])        
    
    def call(self, x, mask=None, training = True):
        batch_size = tf.shape(x)[0]
        
        querry = self.kernel_querry(x)
        querry = self.act(querry)
        key    = self.kernel_key(x)
        key    = self.act(key)
        value  = self.kernel_value(x)
        value  = self.act(value)
        
        querry = self.split_heads( querry, batch_size)
        key    = self.split_heads( key, batch_size)
        value  = self.split_heads( value, batch_size)
         
        
        attention_logits  = tf.matmul(querry, key, transpose_b = True) / self.scale
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

class AbsolutePositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_length, embedding_dim, initializer="truncated_normal", **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, _):
        self.positional_table = self.add_weight(
            name="positional_table",
            shape=(1, self.max_length, self.embedding_dim),
            initializer=self.initializer,
            trainable=True,
        )

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.positional_table[:, :seq_len, :]

def Generator(norm="layer"):
    inputs = tf.keras.layers.Input(shape=[512, 21])

    down_stack = [
    downsample(512, 4, norm=None),  # (batch_size, 128, 128, 64)
    downsample(512, 4, norm=norm),  # (batch_size, 64, 64, 128)
    downsample(512, 4, norm=norm),  # (batch_size, 32, 32, 256)
    downsample(512, 4, norm=norm),  # (batch_size, 16, 16, 512)
    downsample(1024, 4, norm=norm),  # (batch_size, 8, 8, 512)
    #downsample(512, 4, norm=norm),  # (batch_size, 4, 4, 512)
    #downsample(512, 4, norm=norm),  # (batch_size, 2, 2, 512)
    #downsample(512, 4, norm=norm),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
    #upsample(512, 4, apply_dropout=True, norm=norm),  # (batch_size, 2, 2, 1024)
    #upsample(512, 4, apply_dropout=True, norm=norm),  # (batch_size, 4, 4, 1024)
    #upsample(512, 4, apply_dropout=True, norm=norm),  # (batch_size, 8, 8, 1024)
    upsample(512, 4, norm=norm),  # (batch_size, 16, 16, 1024)
    upsample(512, 4, norm=norm),  # (batch_size, 32, 32, 512)
    upsample(512, 4, norm=norm),  # (batch_size, 64, 64, 256)
    upsample(512, 4, norm=norm),  # (batch_size, 128, 128, 128)
    ]

    down_res_stack = [
        ResModPreAct(512,4, norm = False, name="res_down0"),
        ResModPreAct(512,4, norm = "layer", name="res_down1"),
        ResModPreAct(512,4, norm = "layer", name="res_down2"),
        ResModPreAct(512,4, norm = "layer", name="res_down3"),
        ResModPreAct(1024,4, norm = "layer", name="res_down4"),
        ]
    


    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv1DTranspose(21, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='linear')  # (batch_size, 256, 256, 3)
    pos_emb = AbsolutePositionalEmbedding(max_length=512, embedding_dim=inputs.shape[-1], name="gen_pos_emb")
    gsm_out = GumbelSoftmax()
    sm_out = Softmax()

    x = inputs
    x = pos_emb(x)
    # Downsampling through the model
    skips = []
    for down, res in zip(down_stack, down_res_stack):
        x = res(x)
        skips.append(x)
        x = down(x)
        

    skips = reversed(skips)#[:-1])

    # Upsampling and establishing the skip connections
    filt = [512, 512, 512, 512]
    heads = [8, 8, 8, 8]
    nn = 0
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x, _ = UAttention(filt[nn], heads[nn], name=f"U_atte{nn}")(x, skip)
        x = ResModPreAct(filt[nn],4, norm = "layer", name=f"res_up{nn}")(x)
        nn += 1
        #x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    sm  = sm_out(x)
    gsm = gsm_out(x)
    
    return tf.keras.Model(inputs=inputs, outputs=[gsm, sm])

def Discriminator(norm="layer"):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[ 512, 21], name='input_image')
    x = tf.keras.layers.GaussianNoise(stddev = 0.1)(inp)
    x = AbsolutePositionalEmbedding(max_length=512, embedding_dim=inp.shape[-1], name="disc_pos_emb")(x)
    down1 = downsampleSN(512, 4, False, act = "LReLU")(x)  # (batch_size, 128, 128, 64)

    x = ResModPreActSN(512, 4, norm=norm, act = "LReLU", name="res0")(down1)
    
    
    down2 = downsampleSN(512, 4, norm = norm, act = "LReLU")(x)  # (batch_size, 64, 64, 128)

    x = ResModPreActSN(512, 4, act = "LReLU", norm=norm, name="res1")(down2)
    x = ResModPreActSN(512, 4, act = "LReLU", norm=norm, name="res11")(x)
    x, _ = SelfAttentionSN(512, heads=4, act = "LReLU", name = "attention_0")(x)
    
    down3 = downsampleSN(512, 4, norm=norm, act = "LReLU" )(x)  # (batch_size, 32, 32, 256)


    x = ResModPreActSN(512, 4, norm=norm, act = "LReLU",  name="res2")(down3)
    x = ResModPreActSN(512, 4, norm=norm, act = "LReLU",  name="res22")(x)
    x, _ = SelfAttentionSN(512, heads = 8, act = "LReLU", name = "attention_1")(x)
   

    

    last = tf.keras.layers.Conv1D(1, 4, strides=1,
                                kernel_initializer=initializer,
                                 activation=None)(x) # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)
