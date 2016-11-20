#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras.models import Model
from keras.layers import Input, Reshape, LSTM, RepeatVector, Dense
from keras.layers.convolutional import Convolution2D
from attention import Attention

IMAGE_SIZE = 28
N_DIGITS = 3


def convolutional_features(inp, filters, size):
    conv = Convolution2D(filters, size, size, border_mode='same')(inp)
    conv = Convolution2D(filters, size, size, border_mode='same')(conv)
    conv = Convolution2D(filters, 2, 2, border_mode='same', subsample=(2, 2))(conv)
    conv = Convolution2D(filters, size, size, border_mode='same')(conv)
    conv = Convolution2D(filters, size, size, border_mode='same')(conv)
    conv = Convolution2D(filters, 2, 2, border_mode='same', subsample=(2, 2))(conv)
    conv = Convolution2D(filters, size, size, border_mode='same')(conv)
    conv = Convolution2D(filters, size, size, border_mode='same')(conv)
    return conv

def cost():
    pass

FILTERS = 16
FILTER_SIZE = 3

def model():
    inputs = Input(shape=(1, IMAGE_SIZE, N_DIGITS * IMAGE_SIZE))
    cfeatures = convolutional_features(inputs, FILTERS, FILTER_SIZE)

    cfeatures = Reshape((FILTERS * IMAGE_SIZE * N_DIGITS * IMAGE_SIZE // 16, ))(cfeatures)

    cfeatures = RepeatVector(3)(cfeatures)

    output = LSTM(10, return_sequences=True)(cfeatures)

    # output = Dense(10)(output)

    model = Model(input=inputs, output=output)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

if __name__ == "__main__":
    model = model()
