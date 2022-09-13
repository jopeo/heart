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
model_name = "model2.h5"

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
	preprocessor
	
	train_X = preprocessor.fit_transform(train_X)
	val_X = preprocessor.fit_transform(val_X)
	train_X.shape
	val_X.shape
	
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
			train_X, train_y,  # X, y,
			validation_data=(val_X, val_y),
			batch_size=256*2*m,
			epochs=50,
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
	
	model = load_model(model_name)
	model
	
	z = pd.DataFrame(0, index=range(4), columns=X.columns)
	z.shape
	z.iloc[0] = X.mean().astype(int).transpose()
	z.iloc[1] = pd.DataFrame(0, index=range(1), columns=X.columns)
	z = z.astype(float)
	z.shape
	z
	z.columns
	
	n = 2
	z.iloc[n] = pd.DataFrame(0, index=range(1), columns=X.columns)
	z.iloc[n].SLEPTIM1 = 6  # sleep time in hours
	z.iloc[n]._AGE80 = 32  # age
	z.iloc[n]._CHLDCNT = 2  # number of children +1 (1 = no children)
	z.iloc[n].DROCDY3_ = 0  # drink occasions per day
	z.iloc[n]._DRNKWK1 = 0  # drinks per week
	z.iloc[n]._STATE = 6  # state https://www.cdc.gov/brfss/annual_data/2020/pdf/codebook20_llcp-v2-508.pdf
	z.iloc[n].SEXVAR = 1  # sex male = 1, female = 2
	z.iloc[n].HLTHPLN1 = 1  # health plan yes = 1, no = 2
	z.iloc[n].PERSDOC2 = 3  # personal doctor yes = 1, more = 2, no = 3 Do you have one person you think of as your personal doctor or health care provider? (If ´No´ ask ´Is there more than one or is there no person who you think of as your personal doctor or health care provider?´.)
	z.iloc[n].MEDCOST = 2  # Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?
	z.iloc[n].EXERANY2 = 1  # During the past month, other than your regular job, did you participate in any physical activities or exercises such as running, calisthenics, golf, gardening, or walking for exercise?
	z.iloc[n].ASTHMA3 = 2  #  (Ever told) (you had) asthma?
	z.iloc[n].CHCSCNCR = 2  #  (Ever told) (you had) skin cancer?
	z.iloc[n].CHCOCNCR = 2  # (Ever told) (you had) any other types of cancer?
	z.iloc[n].QSTLANG = 1  # Language identifier 1 english 2 spanish 3 other
	z.iloc[n]._IMPRACE = 1  # Imputed race/ethnicity value (This value is the reported race/ethnicity or an imputed race/ethnicity, if the respondent refused to give a race/ethnicity. The value of the imputed race/ethnicity will be the most common race/ethnicity response for that region of the state)
	z.iloc[n]._RFHLTH = 1  #  Adults with good or better health (good or better = 1, fair or poor = 2)
	z.iloc[n]._PHYS14D = 2  #  3 level not good physical health status: 0 days, 1-13 days, 14-30 days (1 Zero days when physical health not good, 2 for 1-13 days when physical health not good, 3 for 14+ days when physical health not good )
	z.iloc[n]._MENT14D = 2  #  3 level not good mental health status: 0 days, 1-13 days, 14-30 days (1 Zero days when physical health not good, 2 for 1-13 days when physical health not good, 3 for 14+ days when physical health not good )
	z.iloc[n]._HCVU651 = 1  #  Respondents aged 18-64 who have any form of health care coverage
	z.iloc[n]._TOTINDA = 1  #  Adults who reported doing physical activity or exercise during the past 30 days other than their regular job
	z.iloc[n]._LTASTH1 = 1  #   Adults who have ever been told they have asthma 1 = no, 2 = yes
	z.iloc[n]._CASTHM1 = 1  #   Adults who have been told they currently have asthma  1 = no, 2 = yes
	z.iloc[n]._ASTHMS1 = 3  #   Adults who have been told they currently have asthma  3 = never, 2 = former, 1 = current
	z.iloc[n]._EXTETH3 = 1  #   Adults aged 18+ who have had permanent teeth extracted 1 = no, 2 = yes
	z.iloc[n]._DENVST3 = 2  #   Adults who have visited a dentist, dental hygenist or dental clinic within the past year 1=yes, 2=no
	z.iloc[n]._HISPANC = 2  #   Hispanic, Latino/a, or Spanish origin calculated variable 1=yes, 2=no
	z.iloc[n]._SEX = 1  #   Calculated sex variable
	z.iloc[n]._RFBMI5 = 1  #   Adults who have a body mass index greater than 25.00 (Overweight or Obese) 1=no, 2=yes
	z.iloc[n]._EDUCAG = 4  #   Level of education completed (1 Did not graduate High School, 2 Graduated High School, 3Attended College or Technical School, 4 Graduated from College or Technical School, 9 don't know
	z.iloc[n]._INCOMG = 5  #  Income categories (1 Less than $15,000, 2 $15,000 to less than $25,000, 3 $25,000 to less than $35,000, 4 $35,000 to less than $50,000, 5 $50,000 or more, 9 dont know
	z.iloc[n]._SMOKER3 = 4  # Four-level smoker status: Everyday smoker, Someday smoker, Former smoker, Non-smoker
	z.iloc[n]._RFSMOK3 = 1  # Adults who are current smokers 1=no, 2=yes
	z.iloc[n].DRNKANY5 = 2  # Adults who reported having had at least one drink of alcohol in the past 30 days. 1=yes, 2=no
	z.iloc[n].DRNKANY5 = 1  # Binge drinkers (males having five or more drinks on one occasion, females having four or more drinks on one occasion) 1=no, 2=yes
	z.iloc[n]._RFDRHV7 = 1  # Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)  1=no, 2=yes
	z.iloc[n]._RFSEAT2 = 1  #  Always or Nearly Always Wear Seat Belts Calculated Variable  1=always or almost, 2=sometimes, seldon, or never
	z.iloc[n]._RFSEAT3 = 1  #   Always Wear Seat Belts Calculated Variable  1=always, 2=not always
	z.iloc[n]._DRNKDRV = 2  #   Drinking and Driving (Reported having driven at least once when perhaps had too much to drink)   1=yes, 2=no
	
	n = 3
	z.iloc[n] = pd.DataFrame(0, index=range(1), columns=X.columns)
	z.iloc[n].SLEPTIM1 = 8  # sleep time in hours
	z.iloc[n]._AGE80 = 70  # age
	z.iloc[n]._CHLDCNT = 1  # number of children +1 (1 = no children)
	z.iloc[n].DROCDY3_ = 2  # drink occasions per day
	z.iloc[n]._DRNKWK1 = 24  # drinks per week
	z.iloc[n]._STATE = 1  # state https://www.cdc.gov/brfss/annual_data/2020/pdf/codebook20_llcp-v2-508.pdf
	z.iloc[n].SEXVAR = 1  # sex male = 1, female = 2
	z.iloc[n].HLTHPLN1 = 2  # health plan yes = 1, no = 2
	z.iloc[n].PERSDOC2 = 3  # personal doctor yes = 1, more = 2, no = 3 Do you have one person you think of as your personal doctor or health care provider? (If ´No´ ask ´Is there more than one or is there no person who you think of as your personal doctor or health care provider?´.)
	z.iloc[n].MEDCOST = 1  # Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 1 =yes, 2=no
	z.iloc[n].EXERANY2 = 2  # During the past month, other than your regular job, did you participate in any physical activities or exercises such as running, calisthenics, golf, gardening, or walking for exercise? 1=yes, 2=no
	z.iloc[n].ASTHMA3 = 1  #  (Ever told) (you had) asthma? 1=yes, 2=no
	z.iloc[n].CHCSCNCR = 1  #  (Ever told) (you had) skin cancer? 1=yes, 2=no
	z.iloc[n].CHCOCNCR = 1  # (Ever told) (you had) any other types of cancer? 1=yes, 2=no
	z.iloc[n].QSTLANG = 2  # Language identifier 1 english 2 spanish 3 other
	z.iloc[n]._IMPRACE = 2  # Imputed race/ethnicity value (This value is the reported race/ethnicity or an imputed race/ethnicity, if the respondent refused to give a race/ethnicity. The value of the imputed race/ethnicity will be the most common race/ethnicity response for that region of the state)
	z.iloc[n]._RFHLTH = 2  #  Adults with good or better health (good or better = 1, fair or poor = 2)
	z.iloc[n]._PHYS14D = 3  #  3 level not good physical health status: 0 days, 1-13 days, 14-30 days (1 Zero days when physical health not good, 2 for 1-13 days when physical health not good, 3 for 14+ days when physical health not good )
	z.iloc[n]._MENT14D = 3  #  3 level not good mental health status: 0 days, 1-13 days, 14-30 days (1 Zero days when physical health not good, 2 for 1-13 days when physical health not good, 3 for 14+ days when physical health not good )
	z.iloc[n]._HCVU651 = 2  #  Respondents aged 18-64 who have any form of health care coverage
	z.iloc[n]._TOTINDA = 2  #  Adults who reported doing physical activity or exercise during the past 30 days other than their regular job
	z.iloc[n]._LTASTH1 = 2  #   Adults who have ever been told they have asthma 1 = no, 2 = yes
	z.iloc[n]._CASTHM1 = 2  #   Adults who have been told they currently have asthma  1 = no, 2 = yes
	z.iloc[n]._ASTHMS1 = 3  #   Adults who have been told they currently have asthma  3 = never, 2 = former, 1 = current
	z.iloc[n]._EXTETH3 = 2  #   Adults aged 18+ who have had permanent teeth extracted 1 = no, 2 = yes
	z.iloc[n]._DENVST3 = 2  #   Adults who have visited a dentist, dental hygenist or dental clinic within the past year 1=yes, 2=no
	z.iloc[n]._HISPANC = 1  #   Hispanic, Latino/a, or Spanish origin calculated variable 1=yes, 2=no
	z.iloc[n]._SEX = 1  #   Calculated sex variable
	z.iloc[n]._RFBMI5 = 2  #   Adults who have a body mass index greater than 25.00 (Overweight or Obese) 1=no, 2=yes
	z.iloc[n]._EDUCAG = 1  #   Level of education completed (1 Did not graduate High School, 2 Graduated High School, 3Attended College or Technical School, 4 Graduated from College or Technical School, 9 don't know
	z.iloc[n]._INCOMG = 2  #  Income categories (1 Less than $15,000, 2 $15,000 to less than $25,000, 3 $25,000 to less than $35,000, 4 $35,000 to less than $50,000, 5 $50,000 or more, 9 dont know
	z.iloc[n]._SMOKER3 = 1  # Four-level smoker status: Everyday smoker, Someday smoker, Former smoker, Non-smoker
	z.iloc[n]._RFSMOK3 = 2  # Adults who are current smokers 1=no, 2=yes
	z.iloc[n].DRNKANY5 = 1  # Adults who reported having had at least one drink of alcohol in the past 30 days. 1=yes, 2=no
	z.iloc[n].DRNKANY5 = 2  # Binge drinkers (males having five or more drinks on one occasion, females having four or more drinks on one occasion) 1=no, 2=yes
	z.iloc[n]._RFDRHV7 = 2 # Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)  1=no, 2=yes
	z.iloc[n]._RFSEAT2 = 2  #  Always or Nearly Always Wear Seat Belts Calculated Variable  1=always or almost, 2=sometimes, seldon, or never
	z.iloc[n]._RFSEAT3 = 2  #   Always Wear Seat Belts Calculated Variable  1=always, 2=not always
	z.iloc[n]._DRNKDRV = 1  #   Drinking and Driving (Reported having driven at least once when perhaps had too much to drink)   1=yes, 2=no
	
	# X_new = [[...], [...]]
	y_new = model.predict(z)
	print(y_new)
	
	pass







