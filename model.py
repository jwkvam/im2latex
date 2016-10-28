#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras.layers import Input, Reshape
from keras.layers.convolutional import Convolution2D
from attention import Attention

IMAGE_SIZE = 28
N_DIGITS = 3

inputs = Input(shape=(1, IMAGE_SIZE, N_DIGITS * IMAGE_SIZE))

def convolutional_features(inp, filters, size):
    conv = Convolution2D(filters, size, size, border_mode='same')(inp)
    conv = Convolution2D(filters, size, size, border_mode='same')(conv)
    conv = Convolution2D(filters, size, size, border_mode='same')(conv)
    conv = Convolution2D(filters, size, size, border_mode='same')(conv)
    return conv

def cost():
    pass

FILTERS = 16
FILTER_SIZE = 3

cfeatures = convolutional_features(inputs, FILTERS, FILTER_SIZE)

cflat = Reshape((FILTERS, IMAGE_SIZE * N_DIGITS * IMAGE_SIZE))(cfeatures)

# Repeat
# import IPython
# IPython.embed()

output = Attention(10)(cflat)
