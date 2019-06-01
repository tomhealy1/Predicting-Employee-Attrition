#Predicting Employee Attrition
#01/06/2019

#First lets Import the Pandas Module and do some analysis

import pandas as pd 
import numpy as np 

#Next we will pass the csv file to variable hr
hr = pd.read_csv('HR.csv')
#We take the columns in the HR file and create a list called col_names
col_names = hr.columns.tolist()
print("Colunm Names:")
print(col_names)
print("\nSample data:")
print(hr.head)

#We dont need to update the names of the columns
#Lets take a look at the data types in our dataset
print(hr.dtypes)

#Next lets check for missing values
print(hr.isnull().any())

#Looks good :+1:

#Lets check the shape
print(hr.shape)

