#linear algebra
import numpy as np

# data processing, csv
import matplotlib as plt

#plotting
import pandas as pd

#visualization
import seaborn as sns

#load into notebook
#import plotly.offline as py
#py.init_notebook_mode(connected=True)

#plotting
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')
print('-'*25)

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
#from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#reading the csv files
gender = pd.read_csv("/Users/ervinli117/Downloads/titanic/gender_submission.csv")
train = pd.read_csv("/Users/ervinli117/Downloads/titanic/train.csv")
test = pd.read_csv("/Users/ervinli117/Downloads/titanic/test.csv")

#prints the matrix of the training dataset
print("Shape of the training set: " ,train.shape)
#prints column headers
print (train.columns.values)

#prints the first 5 rows for training dataset
train.head()

#prints bottom 5 of the training dataset
train.tail()

#number of data in each category for training dataset
train.count()


#for test set
#prints shape of the test dataset
print ("Shape of the test set: ", test.shape)
#prints the first 5 rows for test dataset
test.head()

#finds type of data in the training dataset
print(train.info())

#creating a copy of the data to change
data1 = train.copy(deep = True)

#cleaning both datasets
cleandata = [data1, test]

#getting a sample of 10 in the training dataset
print(train.sample(10))

#identifying null values in each columns
print("Training columns with null values:\n", data1.isnull().sum())
print("-" * 10)

print("Test/Validation columns with null values:\n", test.isnull().sum())
print("-" * 10)

#views some basic statistical details, percentile, mean, etc
train.describe(include = "all")

#complete or delete missing data
for dataset in cleandata:
    #fillna is to fill nonvalues as NA/NaN
    #inplace: if true it modifys the value
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace = True)
    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

#remove specified columns
drop_column = ['PassengerId', 'Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)

print(data1.isnull().sum())
print("-" * 10)
print(test.isnull().sum())

#changing the dataset
for dataset in cleandata:
    #combining the two cateogries
    dataset['FamilySize'] = dataset['SibSp'] + ['Parch'] + 1
    #1 = alone
    dataset['IsAlone'] = 1
    #loc is used to access a group of rows and columns by Label
    #family size greater than one has the isalone column changed to 0
    #0 = not alone
    dataset['IsAlone'].loc[[dataset['FamilySize'] > 1]] = 0
    #splits each comma  or period into a separate portion of a list
    dataset['Title'] = dataset['Name'].str.split(", ", expand = True[1].str.split(".", expand = True)[0]
    #splits the farebin category into 4 quantiles
    #low price, middle lower, middle upper, upper
    #qcut makes into quantized discrete function with 4 being 4 quantiles
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    #bins values into 5 discrete intervals
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
#clean up rare title names
stat_min = 10
#value_counts() returns the count of each unique value and making sure that it is less than stat_min:10
title_names = (data1['Title'].value_counts() < stat_min)

data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
print("-" * 10)
