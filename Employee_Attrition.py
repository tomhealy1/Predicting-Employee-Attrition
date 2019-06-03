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

#We have an issue here with the Sales Department being the wrong word. We will change this now
hr=hr.rename(columns = {'sales':'department'})

#Lets take a look at the data types in our dataset
print(hr.dtypes)

#Next lets check for missing values
print(hr.isnull().any())

#Looks good :+1:

#Lets check the shape
print(hr.shape)

#We have a lot of dimensionality due to many departments, we need to reduce this by combining the department that can be combined
print(hr['department'].unique())

#We are going to combine tech, IT and supoort in to one called technical

hr['department']=np.where(hr['department'] == 'support', 'technical', hr['department'])

hr['department']=np.where(hr['department'] == 'IT', 'technical', hr['department'])

#We have our new super department :-)
print(hr['department'].unique())

#Data Exploration 