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

def compute_padding_mask(inputs, pad_value=0.0, pad_threshold=1e-6):
    mask = tf.reduce_any(tf.abs(inputs - pad_value) > pad_threshold, axis=-1, keepdims=True)
    return tf.cast(mask, inputs.dtype)

def pool_padding_mask(mask, pool_size, strides, padding="SAME"):
    mask_float = tf.cast(mask, tf.float32)
    pooled = tf.nn.max_pool1d(mask_float, ksize=pool_size, strides=strides, padding=padding)
    pooled = tf.where(pooled > 0.0, tf.ones_like(pooled), tf.zeros_like(pooled))
    return tf.cast(pooled, mask.dtype)

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
    def __init__(self, max_length, embedding_dim, initializer="truncated_normal", pad_value=0.0, pad_threshold=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.initializer = tf.keras.initializers.get(initializer)
        self.pad_value = pad_value
        self.pad_threshold = pad_threshold

    def build(self, _):
        self.positional_table = self.add_weight(
            name="positional_table",
            shape=(1, self.max_length, self.embedding_dim),
            initializer=self.initializer,
            trainable=True,
        )

    def call(self, inputs, mask=None):
        seq_len = tf.shape(inputs)[1]
        positional = self.positional_table[:, :seq_len, :]

        if mask is None:
            abs_diff = tf.abs(inputs - self.pad_value)
            padding_positions = tf.reduce_all(abs_diff <= self.pad_threshold, axis=-1, keepdims=True)
            mask = tf.logical_not(padding_positions)

        mask = tf.cast(mask, positional.dtype)
        if mask.shape.rank is None or mask.shape.rank == 2:
            mask = tf.expand_dims(mask, axis=-1)

        positional = positional * mask
        return inputs + positional

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_length": self.max_length,
                "embedding_dim": self.embedding_dim,
                "initializer": tf.keras.initializers.serialize(self.initializer),
                "pad_value": self.pad_value,
                "pad_threshold": self.pad_threshold,
            }
        )
        return config

def Generator(norm="layer", config=None):
    input_dim = config.get("input_dim", 21)
    inputs = tf.keras.layers.Input(shape=[512, input_dim])
    mask = tf.keras.layers.Input(shape=[512, 1])

    filters_down_stack = config["filters_down_stack"] 

    down_stack = [
    downsample(filters_down_stack[0], 4, norm=None),  # (batch_size, 128, 128, 64)
    downsample(filters_down_stack[1], 4, norm=norm),  # (batch_size, 64, 64, 128)
    downsample(filters_down_stack[2], 4, norm=norm),  # (batch_size, 32, 32, 256)
    downsample(filters_down_stack[3], 4, norm=norm),  # (batch_size, 16, 16, 512)
    downsample(filters_down_stack[4], 4, norm=norm),  # (batch_size, 8, 8, 512)
    #downsample(512, 4, norm=norm),  # (batch_size, 4, 4, 512)
    #downsample(512, 4, norm=norm),  # (batch_size, 2, 2, 512)
    #downsample(512, 4, norm=norm),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
    #upsample(512, 4, apply_dropout=True, norm=norm),  # (batch_size, 2, 2, 1024)
    #upsample(512, 4, apply_dropout=True, norm=norm),  # (batch_size, 4, 4, 1024)
    #upsample(512, 4, apply_dropout=True, norm=norm),  # (batch_size, 8, 8, 1024)
    upsample(filters_down_stack[4], 4, norm=norm),  # (batch_size, 16, 16, 1024)
    upsample(filters_down_stack[3], 4, norm=norm),  # (batch_size, 16, 16, 1024)
    upsample(filters_down_stack[2], 4, norm=norm),  # (batch_size, 32, 32, 512)
    upsample(filters_down_stack[1], 4, norm=norm),  # (batch_size, 64, 64, 256)
    upsample(filters_down_stack[0], 4, norm=norm),  # (batch_size, 128, 128, 128)
    ]

    down_res_stack = [
        ResModPreAct(filters_down_stack[0],12, norm = False, name="res_down0"),
        ResModPreAct(filters_down_stack[1],4, norm = "layer", name="res_down1"),
        ResModPreAct(filters_down_stack[2],4, norm = "layer", name="res_down2"),
        ResModPreAct(filters_down_stack[3],4, norm = "layer", name="res_down3"),
        ResModPreAct(filters_down_stack[4],4, norm = "layer", name="res_down4"),
        ]
    


    initializer = tf.random_normal_initializer(0., 0.02)
    bottom = ResModPreAct(filters_down_stack[0],4, norm = "layer", name="bottom")
    last = tf.keras.layers.Conv1D(21, 4,
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='linear', dtype="float32")  # (batch_size, 256, 256, 3)
    #pos_emb = AbsolutePositionalEmbedding(max_length=512, embedding_dim=inputs.shape[-1], name="gen_pos_emb")
    gsm_out = GumbelSoftmax()
    sm_out = Softmax(dtype="float32")

    x = inputs
    if "input_projection" in config:
        x = tf.keras.layers.Conv1D(filters_down_stack[0], 1,strides=1, padding='same', activation='linear')(x)
    #x = pos_emb(x, mask)
    x = AbsolutePositionalEmbedding(max_length=512, embedding_dim=x.shape[-1], name="gen_pos_emb")(x)
    # Downsampling through the model
    skips = []
    for down, res in zip(down_stack, down_res_stack):
        x = res(x)
        skips.append(x)
        x = down(x)
        
    x = bottom(x)
    skips = reversed(skips)#[:-1])
    filters_down_stack = reversed(filters_down_stack)
    # Upsampling and establishing the skip connections
   # filt = [512, 256, 128, 64]
   # heads = [8, 4, 2, 1]
    nn = 0
    for up, skip, filt in zip(up_stack, skips, filters_down_stack):
        x = up(x)
        x, _ = UAttention(filt, filt//64, name=f"U_atte{nn}")(x, skip)
        x = ResModPreAct(filt,4, norm = "layer", name=f"res_up{nn}")(x)
        nn += 1
        #x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    sm  = sm_out(x)
    gsm = gsm_out(x)
    
    return tf.keras.Model(inputs=[inputs, mask], outputs=[gsm, sm, x])

def Discriminator(norm="layer", config = None):
    filters = config["filters"] 
    initializer = tf.random_normal_initializer(0., 0.02)

    input_dim = config.get("input_dim", 21)
    inp = tf.keras.layers.Input(shape=[512, input_dim], name='input_image')
    x = inp
    mask_input = tf.keras.layers.Input(shape=[512, 1], name='input_mask')
    mask = tf.cast(mask_input, x.dtype)
    mask = tf.stop_gradient(mask)
    if "input_projection" in config:
        x = tf.keras.layers.Conv1D(filters[0], 1,strides=1, padding='same', activation='linear')(x)
    x = AbsolutePositionalEmbedding(max_length=512, embedding_dim=x.shape[-1], name="disc_pos_emb")(x, mask)
    mask = tf.cast(mask, x.dtype)
    x = x * mask
    x = tf.keras.layers.GaussianNoise(stddev = 0.01)(x)
    x = ResModPreActSN(filters[0], 6, norm=norm, act = "LReLU", name="res_proj")(x)
    down1 = downsampleSN(filters[0], 4, False, act="LReLU")(x)  # (batch_size, 256, 512)
    mask = tf.stop_gradient(pool_padding_mask(mask, pool_size=2, strides=2))
    mask = tf.cast(mask, down1.dtype)
    down1 = down1 * mask

    x = ResModPreActSN(filters[0], 4, norm=norm, act = "LReLU", name="res0")(down1)
    x = x * mask
    
    
    down2 = downsampleSN(filters[1], 4, norm = norm, act = "LReLU")(x)  # (batch_size, 64, 64, 128)
    mask = tf.stop_gradient(pool_padding_mask(mask, pool_size=2, strides=2))
    mask = tf.cast(mask, down2.dtype)
    down2 = down2 * mask

    x = ResModPreActSN(filters[1], 4, act = "LReLU", norm=norm, name="res1")(down2)
    x = ResModPreActSN(filters[1], 4, act = "LReLU", norm=norm, name="res11")(x)
    x = x * mask
    x, _ = SelfAttentionSN(filters[1], heads=filters[1]//64, act = "LReLU", name = "attention_0")(x)
    x = x * mask
    
    down3 = downsampleSN(filters[2], 4, norm=norm, act = "LReLU" )(x)  # (batch_size, 32, 32, 256)
    mask = tf.stop_gradient(pool_padding_mask(mask, pool_size=2, strides=2))
    mask = tf.cast(mask, down3.dtype)
    down3 = down3 * mask


    x = ResModPreActSN(filters[2], 4, norm=norm, act = "LReLU",  name="res2")(down3)
    x = ResModPreActSN(filters[2], 4, norm=norm, act = "LReLU",  name="res22")(x)
    x = x * mask
    x, _ = SelfAttentionSN(filters[2], heads = filters[2]//64, act = "LReLU", name = "attention_1")(x)
    x = x * mask
   

    

    last = tf.keras.layers.Conv1D(1, 4, strides=1,
                                kernel_initializer=initializer,
                                 activation=None, dtype="float32")(x) # (batch_size, 30, 30, 1)
    final_mask = pool_padding_mask(mask, pool_size=4, strides=1, padding="VALID")
    final_mask = tf.cast(final_mask, last.dtype)
    last = last * final_mask

    return tf.keras.Model(inputs=[inp, mask_input], outputs=[last, final_mask])
