#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networks
import joblib

model = networks.model()


features = joblib.load('features.jbl')
labels = joblib.load('labels.jbl')

model.fit(features, labels)
