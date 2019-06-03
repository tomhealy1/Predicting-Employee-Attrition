#Predicting Employee Attrition
#01/06/2019

#First lets Import the Pandas Module and do some analysis

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score


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
#Lets get the counts of how mant left
print(hr['left'].value_counts())

#Next we will display the mean values of the features of all the people who have left
print(hr.groupby('left').mean())

#Next we group by department and salary
print(hr.groupby('department').mean())

print(hr.groupby('salary').mean())

#Data Visualisation
#We create a bar chart showing the turnover frequency - Title = Turnover Frequency.... , x axis = Department, y axis = Frequency of Turnover
pd.crosstab(hr.department,hr.left).plot(kind='bar')
plt.title('Turnover Frequency for Department')
plt.xlabel('Department')
plt.ylabel('Frequency of Turnover')
plt.savefig('department_bar_chart')
plt.show()

#Next we create a stacked barchart: x axis = Salary Level, y axis = Proportion of Employees, 
table=pd.crosstab(hr.salary, hr.left)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Salary Level vs Turnover')
plt.xlabel('Salary Level')
plt.ylabel('Proportion of Employees')
plt.savefig('salary_bar_chart')
plt.show()

num_bins = 10
hr.hist(bins=num_bins, figsize=(20,15))
plt.savefig("hr_histogram_plots")
plt.show()

#Next we need create a numerical   variable to represent our categorical one for salary and department

cat_var=['department', 'salary']
for var in cat_var:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(hr[var], prefix=var)
    hr1 = hr.join(cat_list)
    hr = hr1

#We need to drop the original columns
hr.drop(hr.columns[[8,9]], axis=1, inplace=True)
print(hr.columns.values)

#Make a new list with the 
hr_vars=hr.columns.values.tolist()
y=['left']
X=[i for i in hr_vars if i not in y]


model = LogisticRegression()
rfe = RFE(model, 10)
rfe = rfe.fit(hr[X], hr[y])
print(rfe.support_)
print(rfe.ranking_)

cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'] 
X=hr[cols]
y=hr['left']

#Logistic regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(X_test))))

#Random forest

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_test, rf.predict(X_test))))