#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_mldata
import joblib

mnist = fetch_mldata('MNIST original', data_home='.')
joblib.dump(mnist, 'mnist.jbl', compress=3)
