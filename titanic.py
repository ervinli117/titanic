#linear algebra
import numpy as np

# data processing, csv
import matplotlib as plt

#plotting
import pandas as pd

#visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


#load into notebook
#import plotly.offline as py
#py.init_notebook_mode(connected=True)

#plotting
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')
print('-'*25)

#machine learning algorithms
import sklearn

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#misc libraries
import random
import time


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
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    #1 = alone
    dataset['IsAlone'] = 1
    #loc is used to access a group of rows and columns by Label
    #family size greater than one has the isalone column changed to 0
    #0 = not alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
    #splits each comma  or period into a separate portion of a list
    dataset['Title'] = dataset['Name'].str.split(", ", expand = True)[1].str.split(".", expand = True)[0]
    #splits the farebin category into 4 quantiles
    #low price, middle lower, middle upper, upper
    #qcut makes into quantized discrete function with 4 being 4 quantiles
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    #bins values into 5 discrete intervals
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

#clean up rare title names
stat_min = 10
#value_counts() returns the count of each unique value and seeing whether it has 10 or more values
title_names = (data1['Title'].value_counts() < stat_min)
#lambda function is an expression
#.apply() passes a function and applies on every value
#changes all the titles that are <10 and groups them together in misc
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
print("-" * 10)

#previews data again
data1.info()
test.info()
data1.sample(10)



#converting objects to category using label encoder for the datasets

#encoode target labels with value between 0 and n_classes - 1
label = LabelEncoder()
for dataset in cleandata:
    #fit_transform fits label encoder and returns encoded labels
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

#defines the y variable or the target you want to measure
Target = ['Survived']

#x variables or the features
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
#coded values
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare']
data1_xy = Target + data1_x
print('Original x,y: ', data1_xy, '\n')

#change the x variables with bin features to remove continuous variables
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin x,y: ', data1_xy_bin, '\n')

#gets x and y dummy variables of the original features
#get_dummies() converts categorical varialbes into dummy/indicator variables
data1_dummy = pd.get_dummies(data1[data1_x])
#gets the columns of the data dummy and converts it into a list
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy x,y: ', data1_xy_dummy, '\n')

#returns top 5 rows of the data
data1_dummy.head()




#double checking cleaned data
print('Train columns with null values: \n', data1.isnull().sum())
print("-" * 10)
#.info() gets concise information of the dataframe
print(data1.info())
print("-" * 10)

print('Test columns with null values: \n', test.isnull().sum())
print('-' * 10)
print(test.info())
print('-' * 10)
#.describe()generate descriptive statistics (mean, median, std, etc)
train.describe(include = 'all')




#split training and test data
#model_selection.train_test_Split: splits arrays or matrices into random train and test subsets
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target], random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)

#.format specifies value and inserts them in a the {}
print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

train1_x_bin.head()




#performing explatory analysis with statistics

for x in data1_x:
    if data1[x].dtype != 'float64':
        print('Survival Correlation by:', x)
        #grouping by series of columns and as_index is group labels as the index
        #mean averages out all the percentage of survived from 1 being survived
        print(data1[[x, Target[0]]].groupby(x, as_index = False).mean())
        print('-' * 10, '\n')

#crosstab computes a frequency table of the factors: title and survived
print(pd.crosstab(data1['Title'], data1[Target[0]]))

#graphing distributions:
#figure creates a new figure to plot
#figsize is the width and the height in inches
plt.figure(figsize = [16,12])
#adds subplot to figure, 2 3 1 describes the position of the subplot (2, 3, 1)
plt.subplot(231)
#creates boxplot of 'Fare' with mean and the mealine
plt.boxplot(x = data1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(x = data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (years)')

plt.subplot(233)
plt.boxplot(x = data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('FamilySize (#)')

plt.subplot(234)
#plots histogram
plt.hist(x = [data1[data1['Survived'] == 1]['Fare'], data1[data1['Survived'] == 0]['Fare']],
stacked = True, color = ['g', 'r'], label = ['Survived', 'Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [data1[data1['Survived'] == 1]['Age'], data1[data1['Survived'] == 0]['Age']], stacked = True,
color = ['g', 'r'], label = ['Survived', 'Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of passengers')
plt.legend()

plt.subplot(236)
plt.hist(x = [data1[data1['Survived'] == 1]['FamilySize'], data1[data1['Survived'] == 0]['FamilySize']],
stacked = True, color = ['g', 'r'], label = ['Survived', 'Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()
#shows all the plots in a separate tab
#plt.show()


#multivariable plots
fig, saxis = plt.subplots(2, 3, figsize = (16,12))
#barplot shows estimates and confidence intervals as rectangular bars
#ax = axes of where the plot will be
#order is to plot categorical levels in
sns.barplot(x = 'Embarked', y = 'Survived', data = data1, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order = [1, 2, 3], data = data1, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order = [1, 0], data = data1, ax = saxis[0, 2])
#shows point estimates and confidence intervals using scatter plot graphs
sns.pointplot(x = 'FareBin', y = 'Survived', data = data1, ax = saxis[1, 0])
sns.pointplot(x = 'AgeBin', y = 'Survived', data = data1, ax = saxis[1, 1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data = data1, ax = saxis[1, 2])
#show the seaborn plots
#plt.show()

#graphing of pclass data:
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (14,12))
#hue is the variable we are looking for
sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data1, ax = axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')

sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)
axis2.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x = 'Pclass', y = 'FamilySize', hue = 'Survived', data = data1, ax = axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison')
#shows all the plots above
#plt.show()


#graphing of sex data:
fig, qaxis = plt.subplots(1, 3, figsize = (14, 12))
sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data = data1, ax = qaxis[0])
axis1.set_title('Sex vs Embarked Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data = data1, ax = qaxis[1])
axis1.set_title('Sex vs Pclass Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data = data1, ax = qaxis[2])
axis1.set_title('Sex vs IsAlone Survival Comparison')

#plt.show()


fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize = (14, 12))
#how family size factor with sex and survival compare
sns.pointplot(x = 'FamilySize', y = 'Survived', hue = 'Sex', data = data1,
palette = {"male": "blue", "female": "pink"}, markers = ["*", "o"], linestyles = ["-", "--"], ax = maxis1)

#how does class factor with sex and survival compare
sns.pointplot(x = 'Pclass', y = 'Survived', hue = 'Sex', data = data1,
palette = {"male": "blue", "female": "pink"}, markers = ["*", "o"], linestyles = ["-", "--"], ax = maxis2)


#how embark factor with class, sex, and survival compare
#facetgrid is multi-plot grid for plotting conditional relationships
#col is the variable
e = sns.FacetGrid(data1, col = 'Embarked')
#map applies plotting function to each facet's subset of data
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci = 95.0, palette = 'deep')
e.add_legend()


#plot distributions of age of passengers of people who survived vs did not survive
a = sns.FacetGrid(data1, hue = 'Survived', aspect = 4)
#kdeplot fits and plots a univariate or bivariate kernel density estimate
a.map(sns.kdeplot, 'Age', shade = True)
#xlim is the axis limits from 0 to the maximum age
a.set(xlim = (0, data1['Age'].max()))
a.add_legend()


#histogram comparison of sex, class and age by survival
h = sns.FacetGrid(data1, row = 'Sex', col = 'Pclass', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = 0.75)
h.add_legend()

#plt.show()
#pair plots of the entire dataset
#plot pairwise relationships in a dataset
#diag_kind is the kind of plot for diagonal subplots
#diag_kws is the argument for inserting, and dict is the list of values mapping
    #pp = sns.pairplot(data1, hue = 'Survived', palette = 'deep', size = 1.2, diag_kind = 'kde', diag_kws = dict(shade = True), plot_kws = dict(s = 10))
#xticklabel plots the column names in the data frame
    #pp.set(xticklabels = [])

#correlation heatmap of the dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize = (14,12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    _ = sns.heatmap(df.corr(), cmap = colormap, square = True, cbar_kws = {'shrink': .9}, ax = ax, annot = True, linewidth = 0.1
    ,vmax = 1.0, linecolor = 'white', annot_kws = {'fontsize': 12})
    plt.title('Pearson Correlation of Features', y = 1.05, size = 15)
correlation_heatmap(data1)

#plt.show()


#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    #Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),


    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()
    ]

#splitting dataset into 60 training and 30 split
#ShuffleSplit splits into training and test sets
#n_splits is number of re-shuffling and splitting iterations
#test size is proportion of dataset in test split
#train size is proportion of dataset in train split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = 0.3, train_size = 0.6, random_state = 0)

#Machine learning algorithm metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#comparing MLA predictions
MLA_predict = data1[Target]

#saving the performance into a table
row_index = 0
for alg in MLA:
    #accesses a class
    MLA_name = alg.__class__.__name__
    #.loc accesses a group of row and columns by label
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    #evaluates metrics by cross validation and records fit/score times
    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv  = cv_split, return_train_score = True)
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    #captures 3 standard deviations which is 99.7% of the data
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std() * 3
    #fitting is equal to training
    alg.fit(data1[data1_x_bin], data1[Target])
    #.predict is predicting the new values
    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])

    row_index += 1

#by = name or list to sort by
# ascending = false means its in descending order
#inplace is performing operation in place
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
print(MLA_compare)


#barplots using seaborn
sns.barplot(x = 'MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')



#Evaluating model performances

#coin flip with random 1 surviving and 0 died
#iterrows() iterates dataframe as (index,series) pairs
for index, row in data1.iterrows():
    #random() creates a random number from 0 to 1
    if random.random() > .5:
        #at[] is the location of the dataframe and changes the value of that index
        data1.at[index, 'Random_Predict'] = 1
    else:
        data1.at[index, 'Random_Predict'] = 0
data1['Random_Score'] = 0
#set 1 if the prediction is correct
data1.loc[(data1['Survived'] == data1['Random_Predict']), 'Random_Score'] = 1
#{:.2f}% prints float of 2 decimal places
#format puts placeholders in {:.2f}
print('Coin Flip Accuracy: {:.2f}%'.format(data1['Random_Score'].mean() * 100))
print('Coin Flip Model Accuracy w/SciKit: {:.2f}%'.format(metrics.accuracy_score(data1['Survived'], data1['Random_Predict'])*100))

#pivot table
#only has female data and has columns in sex, pclass, that order
pivot_female = data1[data1.Sex == 'female'].groupby(['Sex', 'Pclass', 'Embarked', 'FareBin_Code'])['Survived'].mean()
print('Survival Decision Tree with Female Node: \n', pivot_female)

pivot_male = data1[data1.Sex == 'male'].groupby(['Sex', 'Title'])['Survived'].mean()
print('Survival Decision Tree with Male Node: \n', pivot_male)




#data model using brain power
def mytree(df):
    #initiliazing tables to store predictions
    Model = pd.DataFrame(data = {'Predict':[]})
    male_title = ['Master']
    for index, row in df.iterrows():
        #question 1: were you on titanic; majority died
        Model.loc[index, 'Predict'] = 0
        #are you female, majority survived
        if (df.loc[index, 'Sex'] == 'female'):
            Model.loc[index, 'Predict'] = 1
        #set anything less than 0.5 in female node decision tree to 0
        if ((df.loc[index, 'Sex'] == 'female') & (df.loc[index, 'Pclass'] == 3) & (df.loc[index, 'Embarked'] == 'S'), (df.loc[index, 'Fare'] > 8)):
            Model.loc[index, 'Predict'] = 0
        #set anything greater than 0.5 to 1 for majority Survived
        if ((df.loc[index, 'Sex'] == 'male') & (df.loc[index, 'Title'] in male_title)):
            Model.loc[index, 'Predict'] = 1
    return Model

#Model Data
Tree_Predict = mytree(data1)
print('Decision Tree Model Accuracy/Precision Score: {:.2f}%\n'.format(metrics.accuracy_score(data1['Survived'], Tree_Predict) * 100))
#accuracy summary report
#classifcation_report shows the main classifcation metrics
print(metrics.classification_report(data1['Survived'], Tree_Predict))


#plot accuracy summary
import itertools
#creates a confusion matrix
#normalize can be set to true
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion Matrix, without normalization")
    print(cm)
    #imshow displays data as an image on a 2d master
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    #returns evenly spaced values with a given inteerval
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = 'center', color = 'white' if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

#computing confusion Matrix
cnf_matrix = metrics.confusion_matrix(data1['Survived'], Tree_Predict)
np.set_printoptions(precision = 2)

class_names = ['Dead', 'Survived']

#plotting non-normalize confusion Matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = class_names, title = 'Confusion Matrix without normalization')

#plotting normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = class_names, normalize = True, title = 'Normalized confusion matrix')

#plt.show()



#tuning model with hyper Parameters
#before and after decision tree statistics
dtree = tree.DecisionTreeClassifier(random_state = 0)
base_results = model_selection.cross_validate(dtree, data1[data1_x_bin], data1[Target], cv = cv_split, return_train_score = True)
dtree.fit(data1[data1_x_bin], data1[Target])

print('Before DT Parameters: ',dtree.get_params())
print('Before DT Training w/bin score mean: {:.2f}'.format(base_results['train_score'].mean() * 100))
print('Before DT Test w/bin score mean: {:.2f}'.format(base_results['test_score'].mean() * 100))
print('Before DT Test w/bin score 3*std: +/- {:.2f}'.format(base_results['test_score'].std() * 100 * 3))
print('-' * 10)


#tune hyper parameters:
param_grid = {'criterion' : ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10, None], 'random_state': [0]}
#exhaustive search over specified parameter values for an estimator
tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid = param_grid, scoring = 'roc_auc', cv = cv_split, return_train_score = True)
tune_model.fit(data1[data1_x_bin], data1[Target])

print('After DT Parameters: ',tune_model.best_params_)
print('After DT Training w/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_] * 100))
print('After DT Test w/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_] * 100))
print('After DT Test w/bin score 3*std: +/- {:.2f}'.format(tune_model.cv_results_['std_test_score'][tune_model.best_index_] * 100 * 3))
print('-' * 10)



#tuning model with feature Selection
#using sklearn's recursive feature elimination with cross validation
print('Before DT RFE Training Shape Old: ', data1[data1_x_bin].shape)
print('Before DT RFE Training Columns Old: ', data1[data1_x_bin].columns.values)
print('Before DT RFE Training w/bin score mean: {:.2f}'.format(base_results['train_score'].mean() * 100))
print('Before DT RFE Test w/bin score mean: {:.2f}'.format(base_results['test_score'].mean() * 100))
print('Before DT RFE Test w/bin score 3*std: +/- {:.2f}'.format(base_results['test_score'].std() * 3 * 100))
print('-' * 10)

#feature Selection
dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy', cv = cv_split)
dtree_rfe.fit(data1[data1_x_bin], data1[Target])
X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()]
rfe_results = model_selection.cross_validate(dtree, data1[X_rfe], data1[Target], cv = cv_split, return_train_score = True)

#print dtree_rfe.grid_scores_
print('After DT RFE Training Shape New: ', data1[X_rfe].shape)
print('After DT RFE Training Columns New: ', X_rfe)
print('After DT RFE Training w/bin score mean: {:.2f}'.format(rfe_results['train_score'].mean() * 100))
print('After DT RFE Test w/bin score mean: {:.2f}'.format(rfe_results['test_score'].mean() * 100))
print('After DT RFE Test w/bin score 3*std: +/- {:.2f}'.format(rfe_results['test_score'].std() * 100 * 3))
print('-' * 10)



#tuning rfe model
rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid = param_grid, scoring = 'roc_auc', cv = cv_split, return_train_score = True)
rfe_tune_model.fit(data1[X_rfe], data1[Target])

print('After DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)
print('After DT RFE Tuned Training w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_] * 100))
print('After DT RFE Tuned Test w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_] * 100))
print('After DT RfE Tuned Test w/bin score 3*std: +/- {:.2f}'.format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_] * 100 * 3))
print('-' * 10)



#graphing of a decision tree
import graphviz
dot_data = tree.export_graphviz(dtree, out_file = None, feature_names = data1_x_bin, class_names = True, filled = True, rounded = True)
graph = graphviz.Source(dot_data)
graph





#preparing for submission using the validation data
correlation_heatmap(MLA_predict)

#using a voting classifier to pick one model
vote_est = [
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc', ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    ('lr', linear_model.LogisticRegressionCV()),
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    ('knn', neighbors.KNeighborsClassifier()),
    ('svc', svm.SVC(probability = True)),
    ('xgb', XGBClassifier())
]

#hard voting/majority rules
#voting classifier is soft voting/majority rules classifer for unfitted estimators
vote_hard = ensemble.VotingClassifier(estimators = vote_est, voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, data1[data1_x_bin], data1[Target], cv = cv_split, return_train_score = True)
vote_hard.fit(data1[data1_x_bin], data1[Target])
print("Hard voting Training w/bin score mean: {:.2f}".format(vote_hard_cv['train_score'].mean() * 100))
print('Hard voting Test w/bin score mean: {:.2f}'.format(vote_hard_cv['test_score'].mean() * 100))
print('Hard voting Test w/bin score 3*std: +/- {:.2f}'.format(vote_hard_cv['test_score'].std() * 3 * 100))
print('-' * 10)

#soft voting or weighted possibilities
vote_soft = ensemble.VotingClassifier(estimators = vote_est, voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, data1[data1_x_bin], data1[Target], cv = cv_split, return_train_score = True)
vote_soft.fit(data1[data1_x_bin], data1[Target])
print("Soft voting Training w/bin score mean: {:.2f}".format(vote_soft_cv['train_score'].mean() * 100))
print('Soft voting Test w/bin score mean: {:.2f}'.format(vote_soft_cv['test_score'].mean() * 100))
print('Soft voting Test w/bin score 3*std: +/- {:.2f}'.format(vote_soft_cv['test_score'].std() * 3 * 100))
print('-' * 10)



#hyperparameter tune with GridSearchCV
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

grid_param = [
            [{
            #AdaBoostClassifier
            'n_estimators' : grid_n_estimator,
            'learning_rate': grid_learn,
            'random_state': grid_seed
            }],
            [{
            #BaggingClassifier
            'n_estimators': grid_n_estimator,
            'max_samples': grid_ratio,
            'random_state': grid_seed
            }],
            [{
            #ExtraTreesClassifier
            'n_estimators': grid_n_estimator,
            'criterion': grid_criterion,
            'max_depth': grid_max_depth,
            'random_state': grid_seed
            }],
            [{
            #GradientBoostingClassifier
            'learning_rate': [.05],
            'n_estimators': [300],
            'max_depth': grid_max_depth,
            'random_state': grid_seed
            }],
            [{
            #RandomForestClassifier
            'n_estimators': grid_n_estimator,
            'criterion': grid_criterion,
            'max_depth': grid_max_depth,
            'oob_score': [True],
            'random_state': grid_seed
            }],
            [{
            #GaussianProcessClassifier
            'max_iter_predict': grid_n_estimator,
            'random_state': grid_seed
            }],
            [{
            #LogisticRegressionCV
            'fit_intercept': grid_bool,
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'random_state': grid_seed
            }],
            [{
            #BernoulliNB
            'alpha':grid_ratio,
            }],
            #GaussianNB
            [{}],
            [{
            #KNeighborsClassifier
            'n_neighbors': [1,2,3,4,5,6,7],
            'weights':['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }],
            [{
            #SVC
            'C':[1,2,3,4,5],
            'gamma': grid_ratio,
            'decision_function_shape': ['ovo', 'ovr'],
            'probability': [True],
            'random_state': grid_seed
            }],
            [{
            #XGBClassifier
            'learning_rate': grid_learn,
            'max_depth': [1,2,4,6,8,10],
            'n_estimators': grid_n_estimator,
            'seed': grid_seed
            }]
]

#perf_counter provides various time-related functions
start_total = time.perf_counter()
for clf, param in zip(vote_est, grid_param):
    start = time.perf_counter()
    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc',return_train_score = True)
    best_search.fit(data1[data1_x_bin], data1[Target])
    run = time.perf_counter() - start
    best_param = best_search.best_params_
    print('The best parameter for {} is {} with runtime of {:.2f} seconds'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param)

run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total / 60))
print('-' * 10)


#hard vote or majority rules w/Tuned Hyperparameters
grid_hard = ensemble.VotingClassifier(estimators = vote_est, voting = 'hard')
grid_hard_cv = model_selection.cross_validate(grid_hard, data1[data1_x_bin], data1[Target], cv = cv_split, return_train_score = True)
grid_hard.fit(data1[data1_x_bin], data1[Target])
print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))
print('-'*10)

#soft vote or weighted probabilities w/tuned Hyperparameters
grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
grid_soft_cv = model_selection.cross_validate(grid_soft, data1[data1_x_bin], data1[Target], cv  = cv_split, return_train_score = True)
grid_soft.fit(data1[data1_x_bin], data1[Target])
print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))
print('-'*10)




#preparing data for modeling
print(test.info())
print('-' * 10)
#handmade decision tree - submission score
test['Survived'] = mytree(test).astype(int)
test['Survived'] = grid_hard.predict(test[data1_x_bin])


#submitting files
submit = test[['PassengerId', 'Survived']]
submit.to_csv(r'/Users/ervinli117/Downloads/titanic/submit.csv' , index = False)
print('Validation Data Distribution: \n', test['Survived'].value_counts(normalize = True))
submit.sample(10)
