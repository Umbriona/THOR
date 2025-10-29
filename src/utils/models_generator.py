import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D,Conv1DTranspose, Dense, Flatten, Activation, Embedding, Input, LeakyReLU, ReLU, LayerNormalization, BatchNormalization, Softmax, Dropout, Concatenate, Layer
from utils.layers_new import SelfAttention, SelfAttentionSN, GumbelSoftmax, SpectralNormalization
from utils.layers_residual import ResModPreAct, ResModPreActSN, residual_mod
from utils import preprocessing as pre

ATTENTION_FEATURES = 512

def linear(x):
    return x

def sn_up_down(config, name,down_sample, l1, l2, gms, filters_up, filters_down, vocab):
    projection = SpectralNormalization(Dense(filters_down[0],
                                     activation = None,
                                     use_bias = False,
                                     kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                     name=name+"_in_proj"), 
                               name=name+"_sn_in_proj")
    
    conv_down = [SpectralNormalization(Conv1D(filters_down[i], 9,
                 strides=2,
                 padding="same",
                 activation = "relu",
                 kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                 name = name + "_down_{}".format(i)), name =name + "_down_sn_{}".format(i)) for i in range(down_sample)]

    conv_up = [SpectralNormalization(Conv1DTranspose(filters_up[i], 9,
                                strides=2,
                                padding="same",
                                activation = "relu",
                                kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                name = name + "up_{}".format(i)), name=name + "up_sn_{}".format(i)) for i in range(down_sample)]
        
    outconv1 = SpectralNormalization(Conv1D(filters_up[-1],
                                            9,
                                            padding = 'same',
                                            activation = "relu",
                                            name = name + "_output1"),
                                     name=name + "_output1_sn")
    outconv2 = SpectralNormalization(Conv1D(vocab, 9, padding = 'same', activation = gms, name = name + "_output2"), name = name + "_output2_sn")
    return projection, conv_down, conv_up, outconv1, outconv2

def up_down(config, name,down_sample, l1, l2, gms, filters_up, filters_down, vocab):
    projection = Dense(filters_down[0],
                                     activation = None,
                                     use_bias = False,
                                     kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                     name=name+"_in_proj")
    
    conv_down = [Conv1D(filters_down[i], 9,
                 strides=2,
                 padding="same",
                 activation = "relu",
                 kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                 name = name + "_down_{}".format(i)) for i in range(down_sample)]

    conv_up = [Conv1DTranspose(filters_up[i], 9,
                                strides=2,
                                padding="same",
                                activation = "relu",
                                kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                name = name + "up_{}".format(i)) for i in range(down_sample)]

    outconv1 = Conv1D(filters_up[-1],
                    9,
                    padding = 'same',
                    activation = "relu",
                    name = name + "_output1")
    outconv2 = SpectralNormalization(Conv1D(vocab, 9, padding = 'same', activation = gms, name = name + "_output2"), name = name + "_output2_sn")
    return projection, conv_down, conv_up, outconv1, outconv2


class Generator_res(Model):
    def __init__(self, config, vocab, name):
        super(Generator_res, self).__init__(name = name)

        self.n_layers = config['n_layers']
        self.down_sample = config['down_sample']
        self.l1 = config['l1']
        self.l2 = config['l2']
        self.atte_loc = config['attention_loc']
        self.use_atte = config['use_attention']
        
        assert len(config['filters']) >= self.n_layers, "not enough filters specified"
        assert len(config['kernels']) >= self.n_layers, "not enough kernels specified"
        assert len(config['dilations']) >= self.n_layers, "not enough dilations specified"
        
        # Set output distribution (Activation)
        self.use_gumbel = config['use_gumbel']
        if self.use_gumbel:
            self.gms = GumbelSoftmax(temperature = 1, name = self.name + "_gumbel")
        else:
            self.gms = Softmax(name=self.name + "_softmax")
        
        filters_down = [64,128]
        filters_up = [128,64]
        
        if config['use_spectral_norm']:
            self.projection, self.conv_down, self.conv_up, self.outconv1, self.outconv2 = sn_up_down(config,
                                                                                                     self.name,
                                                                                                     self.down_sample,
                                                                                                     self.l1,
                                                                                                     self.l2, 
                                                                                                     self.gms,
                                                                                                     filters_up,
                                                                                                     filters_down,
                                                                                                    vocab)
            self.atte = SelfAttentionSN(config['filters'][self.atte_loc],
                                    name = self.name + "_attention")
            
            self.res = [ResModPreActSN(config['filters'][i],
                           config['kernels'][i],
                           strides=1,
                           dilation = config['dilations'][i],
                           l1=config['l1'],
                           l2=config['l2'],
                           rate = config['rate'],
                           norm=config['norm'],
                           act="ReLU",
                            name = self.name + "_res_{}".format(i)) for i in range(self.n_layers)]
        else:
            self.projection, self.conv_down, self.conv_up, self.outconv1, self.outconv2 = up_down(config,
                                                                                                     self.name,
                                                                                                     self.down_sample,
                                                                                                     self.l1,
                                                                                                     self.l2, 
                                                                                                     self.gms,
                                                                                                     filters_up,
                                                                                                     filters_down,
                                                                                                 vocab)
        
            self.atte = SelfAttention(config['filters'][self.atte_loc],
                                    name = self.name + "_attention")
            
            self.res = [ResModPreAct(config['filters'][i],
                           config['kernels'][i],
                           strides=1,
                           dilation = config['dilations'][i],
                           l1=config['l1'],
                           l2=config['l2'],
                           rate = config['rate'],
                           norm=config['norm'],
                           act="ReLU",
                           name = self.name + "_res_{}".format(i)) for i in range(self.n_layers)]
        # Set up down sample

        
        
        self.norm = config['norm']
        if self.norm == "Layer":
            self.norm_up_down = [LayerNormalization(axis = -1, epsilon = 1e-6, name = self.name + "_up_down_{}_norm".format(i)) for i in range(self.down_sample*2)]
        elif self.norm == "Batch":
            self.norm_up_down = [BatchNormalization(name = self.name + "_up_down_{}_norm".format(i)) for i in range(self.down_sample*2)]
        elif self.norm == "Instance":
            self.norm_up_down = [tfa.layers.InstanceNormalization(name = self.name + "_up_down_{}_norm".format(i)) for i in range(self.down_sample*2)]
        else:
            self.norm_up_down = [Linear(name = self.name + "_up_down_{}_norm".format(i)) for i in range(self.down_sample*2)]

        
    def call(self, x, training = True):
        # Down sample
        print(training)
        print(x)
        x = self.projection(x)
        skip = []
        for i in range(self.down_sample):
            if self.norm == "Batch":
                x = self.norm_up_down[i](self.conv_down[i](x), training = training)
            else:
                x = self.norm_up_down[i](self.conv_down[i](x))
            skip.append(x)
        # Ress modules
        for i in range(self.n_layers):
            x = self.res[i](x, training = training)
            # Attention
            if self.atte_loc == i and self.use_atte:
                x, self.a_w = self.atte(x, training = training)
        # Up sample
        print(skip)
        for i in range(self.down_sample):
            x = tf.concat([x, skip[self.down_sample-1-i]], axis = -1)
            if self.norm == "Batch":
                x = self.norm_up_down[i+self.down_sample](self.conv_up[i](x) , training = training)
            else:
                x = self.norm_up_down[i+self.down_sample](self.conv_up[i](x))
            
            
        # Output dist
        x = self.outconv1(x)
        x = self.outconv2(x)
        return x
    
    
def get_generator(config, vocab):
    # Input layer
    model_input = tf.keras.layers.Input(shape=(512,21))
    
    # Parameters
    n_layers = config['n_layers']
    down_sample = config['down_sample']
    l1 = config['l1']
    l2 = config['l2']
    atte_loc = config['attention_loc']
    use_atte = config['use_attention']
    filters = config['filters']
    kernels = config['kernels']
    dilation = config['dilations']
    use_spectral_norm = config['use_spectral_norm']
    norm=config['norm']
    use_gumbel = config['use_gumbel']
    
    filters_down = [128, 128]
    filters_up = [128, 128]
    
    print(filters)
    print(atte_loc)
    
    # Sampling
    if use_gumbel:
        gms = GumbelSoftmax(temperature = 1)
    else:
        gms = Softmax()
    

    if use_spectral_norm:
        projection = SpectralNormalization(Conv1D(64,9, padding='same', activation = 'relu', use_bias = True ,kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)))
        att = SelfAttentionSN(filters[atte_loc])
        down = [SpectralNormalization(Conv1D(filters_down[i], 9,
                 strides=2,
                 padding="same",
                 activation = "relu",
                 kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2))) for i in range(down_sample)]
        up = [SpectralNormalization(Conv1DTranspose(filters_up[i], 9,
                                strides=2,
                                padding="same",
                                activation = "relu",
                                kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2))) for i in range(down_sample)]
        
        outconv1 = SpectralNormalization(Conv1D(filters_up[-1],9,padding = 'same', activation = "relu"))
        outconv2 = SpectralNormalization(Conv1D(vocab, 9, padding = 'same', activation = gms))
    else:
        projection = Conv1D(64,9, padding='same', activation = 'relu', use_bias = True ,kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2))
        att = SelfAttention(filters[atte_loc])
        down = [Conv1D(filters_down[i], 9,
                 strides=2,
                 padding="same",
                 activation = "relu",
                 kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)) for i in range(down_sample)]
        up = [Conv1DTranspose(filters_up[i], 9,
                                strides=2,
                                padding="same",
                                activation = "relu",
                                kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)) for i in range(down_sample)]

        outconv1 = Conv1D(filters_up[-1],9,padding = 'same', activation = "relu")
        outconv2 = Conv1D(vocab, 9, padding = 'same', activation = None)
      
    res = [residual_mod(filters[i],
                        kernels[i],
                        dilation=dilation[i],
                        l1=l1,
                        l2=l2,
                        use_dout = False,
                        use_bias=True,
                        norm=norm,
                        sn = use_spectral_norm,
                        act="ReLU") for i in range(n_layers)]

    # Normalisations
    if norm == "Layer":
        norm_up_down = [LayerNormalization(axis = -1, epsilon = 1e-6) for i in range(down_sample*2+1)]
    elif norm == "Batch":
        norm_up_down = [BatchNormalization() for i in range(down_sample*2+1)]
    elif norm == "Instance":
        norm_up_down = [tfa.layers.InstanceNormalization() for i in range(down_sample*2+1)]
    else:
        norm_up_down = [linear for i in range(down_sample*2+1)]

            

    skip = []
    x = model_input
    x = projection(x)       
    x = norm_up_down[0](x)
    
    x = down[0](x)
    x = norm_up_down[1](x) 
    skip.append(x)
    
    x = down[1](x)
    x = norm_up_down[2](x)
    skip.append(x)

           
    for i in range(n_layers):
        x_out  = res[i](x)
        x = tf.keras.layers.Add()([x_out, x])
        if i == atte_loc and use_atte:
            x = att(x)[0]
    x = tf.keras.layers.Concatenate()([skip[1], x])
    x = up[0](x)
    x = norm_up_down[3](x)
    x = tf.keras.layers.Concatenate()([skip[0], x])
    x = up[1](x)
    x = norm_up_down[4](x)  
    x = outconv1(x)
    x = outconv2(x)
    x = gms(x) 
    model = tf.keras.Model(inputs=model_input, outputs=x)
    return model
    
