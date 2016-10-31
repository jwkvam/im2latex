#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networks
import joblib

model = networks.model()


features = joblib.load('features.jbl')
labels = joblib.load('labels.jbl')

features = features.reshape(features.shape[0], 1,
                            features.shape[1],
                            features.shape[2])

model.summary()
print('fitting')
model.fit(features, labels)
