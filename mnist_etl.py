#!/usr/bin/env python
# -*- coding: utf-8 -*-

import joblib
import numpy.random as rng

data = joblib.load('./mnist.jbl')

IMAGE_SIZE = 28
N_DIGITS = 3
SAMPLES = 10

images = data['data'].reshape((-1, IMAGE_SIZE, IMAGE_SIZE))
target = data['target']

N = len(target)

ix = rng.randint(0, N+1, (SAMPLES, N_DIGITS))

features = images[ix]
features = features.transpose((0, 2, 3, 1))

features = features.reshape((SAMPLES, IMAGE_SIZE, -1), order='F')
labels = target[ix]
