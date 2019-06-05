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
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


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

#We have a lot of dimensionality due to many departments, we need to reduce this by combining the departments that can be combined
print(hr['department'].unique())

#We are going to combine tech, IT and support in to one called technical

hr['department']=np.where(hr['department'] == 'support', 'technical', hr['department'])

hr['department']=np.where(hr['department'] == 'IT', 'technical', hr['department'])

#We have our new super department :-) Lets see what are our new unique departments
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

#Make a new list with the values
hr_vars=hr.columns.values.tolist()
y=['left']
X=[i for i in hr_vars if i not in y]

print(X)

#Time for Models - we are using Recursive Feature Elimination to do the hard work of feature selection n = 10
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


#Here we are using the K Fold cross validation resampling method to check the skill of the model on unseen data (How well does it generalize on unseen data) 
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = RandomForestClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

#We print the result
print(classification_report(y_test, rf.predict(X_test)))

#Next we want a confusion matrix to be created to better understand the Recall and Precision of our models (Recall - when a person leaves does the model predict that)
#(Precision - When we predict someone will leave how often does that occur)

y_pred = rf.predict(X_test)
forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest')
plt.savefig('random_forest')
plt.show()


logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])

plt.figure()
#plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()

feature_labels = np.array(['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'])
importance = rf.feature_importances_
feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))