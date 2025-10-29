import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv1D, Dense, Flatten, Activation, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax, Dropout, Concatenate
from tensorflow.keras.regularizers import L1L2
from utils.layers_new import  SelfAttentionSN, SelfAttention, GumbelSoftmax, SpectralNormalization
from utils.layers_residual import ResModPreAct, ResModPreActSN, residual_mod
from utils import preprocessing as pre

def linear(x):
    return x

class Linear(Layer):
    def __init__(self, obj, name):
        super(Linear, self).__init__(name=name)
        self.obj = obj
    def call(x, training=True):
        x = self.obj(x)
        return x
def first_layers_sn(name, l1, l2, patch, activation):
    projection = SpectralNormalization(Dense(64,
                                        activation = None,
                                        use_bias = False,
                                        kernel_regularizer = L1L2(l1=l1, l2=l2),
                                        name = name+"_proj"),
                                  name = name+"_proj_sn")
        
    input_conv = SpectralNormalization(Conv1D(64,
                                         9,
                                         padding = "same",
                                         activation = LeakyReLU(0.2),
                                         kernel_regularizer = L1L2(l1=l1, l2=l2),
                                         name = name+"_input_conv"),
                                  name = name+"_input_conv_sn")
    if patch:
            out = SpectralNormalization(Conv1D(1,
                                      9,
                                      padding= "same",
                                      activation = activation,
                                      kernel_regularizer = L1L2(l1=l1, l2=l2),
                                      name =name + "_conv_out"))
    else:
        out = SpectralNormalization(Dense(1,
                       activation= activation,
                       use_bias = False,
                        kernel_regularizer = L1L2(l1=l1, l2=l2),
                        name = name + "_dense_out"), name = name + "_dense_out_norm")
    return projection, input_conv, out

def first_layers(name, l1, l2, patch, activation):
    projection = Dense(64,
                                        activation = None,
                                        use_bias = False,
                                        kernel_regularizer = L1L2(l1=l1, l2=l2),
                                        name = name+"_proj")
        
    input_conv = Conv1D(64,
                                         9,
                                         padding = "same",
                                         activation = LeakyReLU(0.2),
                                         kernel_regularizer = L1L2(l1=l1, l2=l2),
                                         name = name+"_input_conv")
    if patch:
        out = Conv1D(1,
                                  9,
                                  padding= "same",
                                  activation = activation,
                                  kernel_regularizer = L1L2(l1=l1, l2=l2),
                                  name = name + "_conv_out")
    else:
        out = Dense(1,
                       activation= activation,
                       use_bias = False,
                        kernel_regularizer = L1L2(l1=l1, l2=l2),
                        name = name + "_dense_out")
    return projection, input_conv, out

class Discriminator(Model):
    def __init__(self, config, vocab, activation = 'sigmoid' , name = "dis"):
        super(Discriminator, self).__init__(name = name)
        self.conv=[]
        self.type = config["type"]
        self.n_layers = config['n_layers']
        self.patch = config["patch"]
        self.l1 = config['l1']
        self.l2 = config['l2']
        self.activation = activation
        
        if config['use_spectral_norm'] == True:
            self.projection, self.input_conv, self.out = first_layers_sn(self.name, self.l1, self.l2, self.patch, self.activation)
            res = ResModPreActSN
            att = SelfAttentionSN
        else:
            self.projection, self.input_conv, self.out = first_layers(self.name, self.l1, self.l2, self.patch, self.activation)
            res = ResModPreAct
            att = SelfAttention

        self.conv = [res(config['filters'][i],
                           config['kernels'][i],
                           strides=config['strides'][i],
                           dilation = config['dilations'][i],
                           l1=config['l1'],
                           l2=config['l2'],
                           rate = config['rate'],
                           norm=config['norm'],
                           name = self.name + "res_{}".format(i)) for i in range(self.n_layers)]  

            
        self.use_atte = config['use_attention']
        self.atte_loc = config['attention_loc'] 
        self.atte = att(config['filters'][self.atte_loc], name = self.name + "_attention")
        
        if not self.patch:
            self.flat = Flatten(name = self.name + "_flat")


    @classmethod
    def from_config(cls, config):
        return cls(**config)  
    
    def call(self, x, training= True):
        x = self.projection(x)
        x = self.input_conv(x)
        for i in range(self.n_layers):
            x = self.conv[i](x)
            if self.atte_loc == i and self.use_atte:
                x, self.a_w = self.atte(x, training=training)
        if not self.patch:
            x = self.flat(x)
        x = self.out(x)
        return x
    

def discriminator(config):
    
    
    return model

def get_discriminator(config, vocab, activation = 'sigmoid'):
    # Input layer
    model_input = tf.keras.layers.Input(shape=(512,21))
    
    n_layers = config['n_layers']
    
    l1 = config['l1']
    l2 = config['l2']
    atte_loc = config['attention_loc']
    use_atte = config['use_attention']
    filters = config['filters']
    kernels = config['kernels']
    dilation = config['dilations']
    use_spectral_norm = config['use_spectral_norm']
    strides = config['strides']
    patch = config['patch']
    norm=config['norm']

    
    # Parameters


    
    

    filters_down = [64,128,128,256,256]
    
    if use_spectral_norm:
        projection = SpectralNormalization(Conv1D(64, 9, padding = "same", activation = LeakyReLU(), use_bias = True,kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)))
        att = SelfAttentionSN(filters[atte_loc])
        down = [SpectralNormalization(Conv1D(filters_down[i], 9,
             strides=2,
             padding="same",
             activation = LeakyReLU(),
             kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2))) for i in range(len(filters_down))]
    
        conv1 = SpectralNormalization(Conv1D(filters_down[0], 9, padding = 'same', activation = LeakyReLU()))
        outconv1 = SpectralNormalization(Conv1D(filters_down[-1], 9, padding = 'same', activation = LeakyReLU()))
        pro = [SpectralNormalization(Conv1D(filters[i], 1, use_bias=True)) for i in range(n_layers)]
        if patch:
            out = SpectralNormalization(Conv1D(1, 9, padding= "same", activation = activation, kernel_regularizer = L1L2(l1=l1, l2=l2)))      
        else:
            flatt = Flatten()
            out = SpectralNormalization(Dense(1, activation= activation, use_bias = False, kernel_regularizer = L1L2(l1=l1, l2=l2)))
    else:
        projection = Conv1D(64, 9, padding = "same", activation = LeakyReLU(), use_bias = True,kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2))
        att = SelfAttention(filters[atte_loc])
        down = [Conv1D(filters_down[i], 9,
             strides=2,
             padding="same",
             activation = LeakyReLU(),
             kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)) for i in range(len(filters_down))]
    
        conv1 = Conv1D(filters_down[0], 9, padding = 'same', activation = LeakyReLU())
        outconv1 = Conv1D(filters_down[-1], 9, padding = 'same', activation = LeakyReLU())
        
        pro = [Conv1D(filters[i], 1, use_bias=True) for i in range(n_layers)]
        if patch:
            out = Conv1D(1, 9, padding= "same", activation = activation, kernel_regularizer = L1L2(l1=l1, l2=l2))      
        else:
            flatt = Flatten()
            out = Dense(1, activation= activation, use_bias = True, kernel_regularizer = L1L2(l1=l1, l2=l2))
        
      
    res = [residual_mod(filters[i],
                        kernels[i],
                        dilation=dilation[i],
                        l1=l1,
                        l2=l2,
                        use_dout = False,
                        use_bias=False,
                        norm=norm,
                        sn = use_spectral_norm,
                        act="LReLU") for i in range(n_layers)]
    

    # Normalisations
    if norm == "Layer":
        norm_up_down = [LayerNormalization(axis = -1, epsilon = 1e-6) for i in range(len(filters_down)+1)]
    elif norm == "Batch":
        norm_up_down = [BatchNormalization() for i in range(len(filters_down)+1)]
    elif norm == "Instance":
        norm_up_down = [tfa.layers.InstanceNormalization() for i in range(len(filters_down)+1)]
    else:
        norm_up_down = [linear for i in range(len(filters_down)+1)]
    
    


    x = model_input
    x = projection(x)      
    x = conv1(x)

    down_count=0       
    for i in range(n_layers):
        x_out  = res[i](x)
        x_pro  = pro[i](x)

        x = tf.keras.layers.Add()([x_out, x_pro])
        if i == atte_loc and use_atte:
            x = att(x)[0]
        if 2 == strides[i]:
            x = down[down_count](x)
            x = norm_up_down[down_count](x)
            down_count+=1


    x = outconv1(x)
    x = norm_up_down[down_count](x)
    if not patch:
        x = flatt(x)
    x = out(x)
    model = tf.keras.Model(inputs=model_input, outputs=x)
    return model