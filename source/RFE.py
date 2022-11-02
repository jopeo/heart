#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from joblib import dump, load

outcome = "_MICHD"  # Reported having MI or CHD
raw_file = "./source/" + "raw.h5"
cleaned_file = "heart_cleaned.h5"
RFE_selector_name = "./source/" + "RFE_selector_heart_2.joblib"
features_txt = "features_heart_2.txt"
step = 0.05
n_features = 0.15
random_state = 1

if __name__ == "__main__":
	data = pd.read_hdf(raw_file)  # to read cleaned data
	data.shape
	data = data.dropna(subset=[outcome], axis=0)
	data.shape
	data = data[data.DISPCODE != 1200]  # == 1200    final disposition (1100 completed or not 1200)
	data.shape
	
	y = abs(data[outcome] - 2)
	X = data.copy().drop([outcome], axis=1)
	
	headers = X.columns.values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	empty_train_columns = []
	for col in X_train.columns.values:
		# all the values for this feature are null
		if sum(X_train[col].isnull()) == X_train.shape[0]:
			empty_train_columns.append(col)
	X = X.drop(empty_train_columns, axis=1)  # ['TOLDCFS', 'HAVECFS', 'WORKCFS']
	X.shape
	
	imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
	X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
	X.isnull().values.any()
	X.shape
	# X.to_hdf(imputed_cleaned_file, "X", complevel=2)
	# X = pd.read_hdf("./source/" + cleaned_file)  # to read cleaned data

	
	selector = RFE(RandomForestClassifier(), n_features_to_select=n_features, step=step, verbose=2)
	selector.fit(X, y)
	
	dump(selector, RFE_selector_name, compress=3)
	selector_2 = load(RFE_selector_name)
	
	features_to_keep = selector_2.get_feature_names_out(X.columns.values)
	features_to_keep = list(features_to_keep)
	len(features_to_keep)
	print(features_to_keep)
	
	with open(features_txt, 'w+') as f:
		f.write(str(features_to_keep))
	
	pass
