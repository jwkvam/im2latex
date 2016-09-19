#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras.layers import Input, Convolutional2D
from attention import Attention

image_len = 28
n_digits = 3

inputs = Input(shape=(image_len, n_digits * image_len))

def convolutional_features(inp):
    conv = Convolutional2D(16, 3, 3)(inp)
    conv = Convolutional2D(16, 3, 3)(inp)
    conv = Convolutional2D(16, 3, 3)(inp)
    conv = Convolutional2D(16, 3, 3)(inp)
    return conv

cfeatures = convolutional_features(inputs)

output = attention()(cfeatures)
