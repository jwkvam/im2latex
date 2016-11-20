#!/usr/bin/env python
# -*- coding: utf-8 -*-

import network
import joblib
import numpy as np

model = network.model()


features = joblib.load('features.jbl')
labels = joblib.load('labels.jbl').astype(int)

features = features.reshape(features.shape[0], 1,
                            features.shape[1],
                            features.shape[2])

print('feature shape = {}'.format(features.shape))
print('labels shape = {}'.format(labels.shape))

exp_labels = np.zeros((labels.shape[0], 10, 3))
exp_labels[np.arange(exp_labels.shape[0])[:, None], labels, np.arange(3)[None, :]] = 1

exp_labels = exp_labels.transpose((0, 2, 1))

print(features.shape)
print(exp_labels.shape)
#
model.fit(features, exp_labels, batch_size=2, verbose=True)
