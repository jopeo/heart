#!/usr/bin/env python

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers





if __name__ ==  "__main__":
	model = keras.Sequential([
			# the hidden ReLU layers
			layers.Dense(units=4, activation='relu', input_shape=[2]),
			layers.Dense(units=3, activation='relu'),
			# the linear output layer
			layers.Dense(units=1),
	])
	
	
	pass