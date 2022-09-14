#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.stats as st
# from scipy.stats import chi2_contingency
#
# import statsmodels.api
#
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.feature_selection import SelectKBest, f_classif
#
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.metrics import classification_report_imbalanced

#  todo: visit https://www.cdc.gov/brfss/annual_data/annual_2020.html to download the data,
#   the SAS Transport Format is used here:

#filename = "heart_2020_cleaned.csv"
full_file = "LLCP2020.XPT"


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


if __name__ ==  "__main__":
	# data = pd.read_csv('./' + filename)
	
	data = pd.read_sas('./' + full_file)
	data.shape
	data.describe()
	data.head()
	data.columns
	
	data = data.dropna(subset=["_MICHD"], axis=0)
	data = data.dropna(axis=1)
	data.shape
	data.describe()
	data.head()
	data.columns

	# small = data.sample(frac=0.01, random_state=1)
	# small.shape
	# small.describe()
	# small.head()
	
	# z = small.dropna(subset=["_MICHD"], axis=0)
	# z.shape
	
	# keep_cols = z.dropna(axis=1)
	# keep_cols.describe
	
	y = data._MICHD
	
	bool("_MICHD" in data.columns)
	len(data.columns)
	y.shape
	
	
	X = data.drop("_MICHD", axis=1)
	len(X.columns)
	X.head()
	X.shape
	
	train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
	
	train_X.shape
	
	# first, a decicision tree model
	model = DecisionTreeRegressor(random_state=1)
	model.fit(train_X, train_y)
	
	val_predictions = model.predict(val_X)
	print(val_predictions[:5])
	bool(1 in val_predictions)
	val_predictions.shape
	print(val_y.head())

	val_mae = mean_absolute_error(val_y, val_predictions)
	val_mae
	
	candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
	my_mae = []
	for node in candidate_max_leaf_nodes:
		my_mae.append(get_mae(node, train_X, val_X, train_y, val_y))
	my_mae
	best_tree_size = candidate_max_leaf_nodes[my_mae.index(min(my_mae))]
	best_tree_size
	
	# now, a random forest model
	forest_model = RandomForestRegressor(random_state=1)
	forest_model.fit(train_X, train_y)
	melb_preds = forest_model.predict(val_X)
	print(mean_absolute_error(val_y, melb_preds))
	
	
	
	
	# categoricals = data.select_dtypes(include=[np.object])
	# categoricals.columns
	# numericals = data.select_dtypes(include=[np.number])
	# numericals.columns
	#
	# sns.pairplot(data, hue="HeartDisease")
	# sns.countplot(x="HeartDisease", data=data)
	#
	# data['HeartDisease'] = data['HeartDisease'].apply(lambda x: 0 if x == 'No' else 1)
	#
	# plt.figure(figsize=(15, 8))
	# ax = sns.kdeplot(data["BMI"][data.HeartDisease == 1], shade=True)  # color="darkturquoise",
	# sns.kdeplot(data["BMI"][data.HeartDisease == 0], shade=True)  # color="lightcoral",
	# plt.legend(['HeartDisease', 'non-HeartDisease'])
	# plt.title('Density Plot of HeartDisease for BMI')
	# ax.set(xlabel='BMI')
	# plt.xlim(10, 50)
	# plt.show()
	#
	# plt.figure(figsize=(15, 8))
	# ax = sns.kdeplot(data["SleepTime"][data.HeartDisease == 1], shade=True)
	# sns.kdeplot(data["SleepTime"][data.HeartDisease == 0], shade=True)
	# plt.legend(['HeartDisease', 'non-HeartDisease'])
	# plt.title('Density Plot of HeartDisease for SleepTime')
	# ax.set(xlabel='SleepTime')
	# plt.xlim(2, 15)
	# plt.show()
	#
	# plt.figure(figsize=(5, 3))
	# sns.barplot('AgeCategory', 'HeartDisease', data=data, )
	# plt.xticks(fontsize=12, rotation=90)
	# plt.yticks(fontsize=12)
	# plt.title('Density Plot of HeartDisease for Age')
	# plt.xlabel('AgeCategory', fontsize=11)
	# plt.ylabel('HeartDisease', fontsize=11)
	# plt.show()
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for feature in numericals.columns:
	# 	plt.subplot(6, 3, n)
	# 	sns.displot(data[feature], kde=True)
	# 	plt.xlabel(feature)
	# 	plt.ylabel("Count")
	# 	n += 1
	#
	# for column in data.columns:
	# 	if data[column].dtypes == "object":
	# 		data[column] = data[column].fillna(data[column].mode().iloc[0])
	# 		uniques = len(data[column].unique())
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for feature in categoricals.columns:
	# 	plt.subplot(6, 3, n)
	# 	sns.countplot(x=feature, hue="HeartDisease", data=data)
	# 	n += 1
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for col in categoricals:
	# 	plt.subplot(6, 3, n)
	# 	sns.countplot(x='Sex', hue=categoricals[col], data=data)
	# 	n += 1
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for feature in numericals:
	# 	plt.subplot(6, 3, n)
	# 	sns.boxplot(y=data[feature], x=data['AlcoholDrinking'])
	# 	n += 1
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for feature in numericals:
	# 	plt.subplot(6, 3, n)
	# 	sns.boxplot(y=data[feature], x=data['Diabetic'])
	# 	n += 1
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for feature in numericals:
	# 	plt.subplot(6, 3, n)
	# 	sns.boxplot(y=data[feature], x=data['PhysicalActivity'])
	# 	n += 1
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for feature in numericals:
	# 	plt.subplot(6, 3, n)
	# 	sns.boxplot(y=data[feature], x=data['Race'])
	# 	n += 1
	
	
	
	
	pass