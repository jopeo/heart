#!/usr/bin/env python

# conda activate tf

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import load_model
full_file = "LLCP2020.XPT"
model_name = "model1.h5"

def load_data(sas_file_name):
	data = pd.read_sas('./' + full_file)
	data = data.dropna(subset=["_MICHD"], axis=0)
	data = data.dropna(axis=1)
	
	data = data.drop([
			'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'DISPCODE',
			'SEQNO', '_PSU', 'QSTVER', '_STSTR', '_STRWT','_RAWRAKE',
			'_WT2RAKE', '_DUALUSE', '_LLCPWT2', '_LLCPWT', '_AGEG5YR',
			'_AGE65YR', '_AGE_G'
			], axis=1)
	
	data = data[data.SLEPTIM1 < 25]    #  sleep time cannot be > 24hr
	data = data[data.HLTHPLN1 < 3]    #  responded yes or no to health plan (excluded refusal or don't know)
	data = data[data.PERSDOC2 < 4]    #  responded yes or no to doctor (excluded refusal or don't know)
	data = data[data.MEDCOST < 3]    #  responded yes or no to medcost (excluded refusal or don't know)
	data = data[data.EXERANY2 < 3]    #  responded yes or no to exercise (excluded refusal or don't know)
	data = data[data.ASTHMA3 < 3]    #  responded yes or no to asthma (excluded refusal or don't know)
	data = data[data._LTASTH1 < 3]    #  responded yes or no to ever asthma (excluded refusal or don't know)
	data = data[data._CASTHM1 < 3]    #  responded yes or no to current asthma (excluded refusal or don't know)
	data = data[data._ASTHMS1 < 4]    #  current former or never asthma
	data = data[data.CHCSCNCR < 3]    #  responded yes or no to skin cancer (excluded refusal or don't know)
	data = data[data.CHCOCNCR < 3]    #  responded yes or no to other cancer (excluded refusal or don't know)
	data = data[data.QSTLANG < 3]    #  responded english or spanish to language (only 1 respondent said other)
	data = data[data._RFHLTH < 3]    #  responded knew their health (excluded refusal or don't know)
	data = data[data._PHYS14D < 4]    #  responded knew their health (excluded refusal or don't know)
	data = data[data._MENT14D < 4]    #  responded knew their health (excluded refusal or don't know)
	data = data[data._TOTINDA < 3]    #  responded yes or no to exercise (excluded refusal or don't know)
	data = data[data._EXTETH3 < 3]    #  responded yes or no to teeth extraction (excluded refusal or don't know)
	data = data[data._DENVST3 < 3]    #  responded yes or no to dentist in last year (excluded refusal or don't know)
	data = data[data._RFBMI5 < 3]    #  responded yes or no to BMI > 25 (excluded refusal or don't know)
	data = data[data._CHLDCNT < 7]    #  responded knew how many children (excluded refusal or don't know)
	data = data[data._EDUCAG < 5]    #  responded knew schooling (excluded refusal or don't know)
	data = data[data.DROCDY3_ < 900]    #  responded knew drink occasions (excluded refusal or don't know)
	data = data[data._DRNKWK1 < 99900]    #  responded knew drinks per week (excluded refusal or don't know)
	X = data.drop(['_MICHD', 'CVDCRHD4', 'CVDSTRK3',
	               ], axis=1)
	y = abs(data._MICHD - 2)
	
	
	train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
	return data, y, X, train_X, val_X, train_y, val_y

#  todo: visit https://www.cdc.gov/brfss/annual_data/annual_2020.html to download the data,
#   the SAS Transport Format is used here:

print(tf.config.list_physical_devices('GPU'))
print(tf.reduce_sum(tf.random.normal([1000, 1000])))


if __name__ ==  "__main__":
	data, y, X, train_X, val_X, train_y, val_y = load_data(full_file)
	train_y.head()
	data.shape
	X.shape
	X.columns
	train_X.shape
	X.head()
	
	input_shape = [train_X.shape[1]]
	input_shape
	
	features_num = ['SLEPTIM1', '_AGE80', '_CHLDCNT', 'DROCDY3_', '_DRNKWK1',
	                ]
	
	features_cat = ['_STATE', 'SEXVAR', 'HLTHPLN1', 'PERSDOC2', 'MEDCOST',
	                'EXERANY2', 'ASTHMA3', 'CHCSCNCR', 'CHCOCNCR', 'QSTLANG',
	                '_IMPRACE', '_RFHLTH', '_PHYS14D', '_MENT14D', '_HCVU651',
	                '_TOTINDA', '_LTASTH1', '_CASTHM1', '_ASTHMS1', '_EXTETH3',
	                '_DENVST3', '_HISPANC', '_SEX',    '_RFBMI5',  '_EDUCAG',
	                '_INCOMG', '_SMOKER3', '_RFSMOK3', 'DRNKANY5',  '_RFBING5',
	                '_RFDRHV7', '_RFSEAT2', '_RFSEAT3', '_DRNKDRV']
	
	preprocessor = make_column_transformer(
			(StandardScaler(), features_num),
			(OneHotEncoder(), features_cat),
	)
	
	early_stopping = EarlyStopping(
			min_delta=0.001,  # minimium amount of change to count as an improvement
			patience=5,  # how many epochs to wait before stopping
			restore_best_weights=True,
	)
	
	#  'relu' activation -- 'elu', 'selu', and 'swish'
	# layers.Dense(32, input_shape=[8]),
	# layers.Activation('relu'),
	m = 2
	
	model = keras.Sequential([
			layers.BatchNormalization(input_shape=input_shape),
			# the hidden ReLU layers
			layers.Dense(units=64*m, activation='relu'),  # , input_shape=input_shape),
			layers.BatchNormalization(),
			layers.Dropout(rate=0.3),  # apply 30% dropout to the next layer
			layers.Dense(units=64*m, activation='relu'),
			layers.BatchNormalization(),
			layers.Dropout(rate=0.3),  # apply 30% dropout to the next layer
			layers.Dense(units=64*m, activation='relu'),
			layers.BatchNormalization(),
			layers.Dropout(rate=0.3),  # apply 30% dropout to the next layer
			# the linear output layer
			# layers.Dense(units=1),
			layers.Dense(1, activation='sigmoid')
	])
	
	model.compile(
			optimizer="adam",
			# loss="mae",
			loss='binary_crossentropy',
			metrics=['binary_accuracy'],
	)
	# model = keras.Sequential([
	# 		layers.Dense(16, activation='relu'),
	# 		layers.Dense(1),
	# ])
	#
	# wider = keras.Sequential([
	# 		layers.Dense(32, activation='relu'),
	# 		layers.Dense(1),
	# ])
	#
	# deeper = keras.Sequential([
	# 		layers.Dense(16, activation='relu'),
	# 		layers.Dense(16, activation='relu'),
	# 		layers.Dense(1),
	# ])
	

	
	# outs = []
	# for array in train_X, train_y, val_X, val_y:
	# 	array = np.asarray(array).astype('float32')
	# 	outs.append(array)
	
	history = model.fit(
			X, y,
			# validation_data=(val_X, val_y),
			batch_size=256*2*m,
			epochs=10,
			callbacks=[early_stopping],  # put your callbacks in a list
			# verbose=0,  # turn off training log
	)
	
	model.save(model_name)
	
	
	# # convert the training history to a dataframe
	# history_df = pd.DataFrame(history.history)
	# history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
	# history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
	# # use Pandas native plot method
	# # history_df['loss'].plot()
	#
	# plt.show()
	
	
	z = pd.DataFrame(0, index=range(2), columns=X.columns)
	z.shape
	z.iloc[0] = X.mean().astype(int).transpose()
	z.iloc[1] = pd.DataFrame(0, index=range(1), columns=X.columns)
	z = z.astype(float)
	z.shape
	z
	z.columns
	
	# X_new = [[...], [...]]
	y_new = model.predict(z)
	print(y_new)
	
	pass







