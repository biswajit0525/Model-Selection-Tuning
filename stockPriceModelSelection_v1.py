# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 06:33:31 2017

@author: Biswajit
"""

import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader) # skipping column names
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

def predict_price(dates, prices, x):
	dates = np.reshape(dates, (len(dates),1)) # converting to matrix of n X 1
	prices = np.reshape(prices, (len(prices),1))
	
	linear_mod = linear_model.LinearRegression() # defining the linear regression model
	linear_mod.fit(dates, prices) # fitting the data points in the model
	
	plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
	plt.plot(dates, linear_mod.predict(dates), color= 'red', label= 'Linear model') # plotting the line made by linear regression
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Linear Regression')
	plt.legend()
	plt.show()
	
	return linear_mod.predict(x)[0][0], linear_mod.coef_[0][0], linear_mod.intercept_[0]

get_data('D:/Analytics/goog.csv') # calling get_data method by passing the csv file to it
print(dates)
print(prices)

predicted_price, coefficient, constant = predict_price(dates, prices, 29)  
