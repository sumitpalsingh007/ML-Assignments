# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:23:16 2018

@author: SUMIT PAL SINGH
"""

#Read the data and divide it into training and testing:
 
import numpy as np
import pandas as pd
from sklearn import preprocessing
# Import the split & the various classifiers
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


## set the seed...
seed = 110
np.random.seed(seed) ## set the seed to stop random behaviour.


#Set location of the input data 
Location = r'/Users/sumitpal.singh/ML/data.csv'
# Split the data into columns and read
datainput = pd.read_csv(Location, names = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8','a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16','a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26','a27', 'a28', 'a29', 'a30','a31', 'a32', 'a33', 'a34', 'a35', 'a36','a37', 'a38', 'a39', 'a40','a41','a42'])

#Print before removing the outliers
print("Original data is : {}".format(datainput.shape))


#Outlier Removal¶
#Dataset<(Q1−1.5∗IQR)|Dataset>(Q3+1.5∗IQR)Dataset<(Q1−1.5∗IQR)|Dataset>(Q3+1.5∗IQR)
first_quartile = datainput.quantile(0.25)
third_quartile = datainput.quantile(0.75)
IQR = third_quartile - first_quartile
## lets use the above formula to remove the outliers and filter the dataset..
clean_data = datainput[~((datainput < (first_quartile - 1.5 * IQR)) | (datainput > (third_quartile + 1.5 * IQR))).any(axis=1)]

#Outcome of the data
y_value = datainput['a42']
#Set the outcome and dedlete it
del datainput['a42']

#print after removing the outliers
print("Data after removal of outliers is : {}".format(clean_data.shape))


# Split data into Test & Training set where test data is 30% & raining data is 70%
x_train, x_test, y_train, y_test = train_test_split(datainput, y_value, test_size=0.3, random_state = 24)


#Scaling for featrure normalization for setting values between 0 & 1
scaling = preprocessing.MinMaxScaler(feature_range=(0, 1))


# Minmax scaling of training & test data
x_train_minmax=scaling.fit_transform(x_train)
x_test_minmax=scaling.fit_transform(x_test)


# Check the split
print ("The data after split:- Test Data is {} Training data is {}".format(x_train_minmax.shape, x_test_minmax.shape))\
    
# Use Descision tree classifier on the training data===========================================
print('---------------------------------------------- ')
classify1 = DecisionTreeClassifier()
#Train the model
classify1.fit(x_train_minmax, y_train)
# Use the model on the test data
y_predicted1 = classify1.predict(x_test_minmax)
print ("The accuracy score using the Decision Tree is : {}".format(metrics.accuracy_score(y_test, y_predicted1)) )
print('Confusion Matrix is : {}'.format(confusion_matrix(y_test, y_predicted1)))
print('Accuracy : {}'.format(accuracy_score(y_test, y_predicted1)))
print('Classification Report : {}'.format(classification_report(y_test, y_predicted1)))
print('---------------------------------------------- ')


# Next use KNearest Neighbours ============================================================
classify2 = KNeighborsClassifier()
#Train the model
classify2.fit(x_train_minmax, y_train)
# Use the model on the test data
y_predicted2 = classify2.predict(x_test_minmax)
print ("The accuracy score using the K Nereast Neighbour is : {}".format(metrics.accuracy_score(y_test, y_predicted2)))
print('Confusion Matrix : {}'.format(confusion_matrix(y_test, y_predicted2)))
print('Accuracy : {}'.format(accuracy_score(y_test, y_predicted2)))
print('Classification Report : {}'.format(classification_report(y_test, y_predicted2)))
print('---------------------------------------------- ')


#Next use NaiveBayes Classifier =========================================================================
classify3 = BernoulliNB()
#Train the model
classify3.fit(x_train_minmax, y_train)
# Use the model on the test data
y_predicted3 = classify3.predict(x_test_minmax)
print ("The accuracy score using the Naive Bayes Classifier is  : {}".format(metrics.accuracy_score(y_test, y_predicted3)) )
print('Confusion Matrix : {}'.format(confusion_matrix(y_test, y_predicted3)))
print('Accuracy  : {}'.format(accuracy_score(y_test, y_predicted3)))
print('\nClassification Report  : {}'.format(classification_report(y_test, y_predicted3)))
print('---------------------------------------------- ')


# Next use RandomForest Classifier ===========================================================
classify4 = RandomForestClassifier()
#Train the model
classify4.fit(x_train_minmax, y_train)
# Use the model on the test data
y_predicted4 = classify4.predict(x_test_minmax)
print ("The accuracy score using the RandomForest is  : {}".format(metrics.accuracy_score(y_test, y_predicted4)) )
print('Confusion Matrix  : {}'.format(confusion_matrix(y_test, y_predicted4)))
print('Accuracy  : {}'.format(accuracy_score(y_test, y_predicted4)))
print('Classification Report  : {}'.format(classification_report(y_test, y_predicted4)))
print('---------------------------------------------- ')



# Next use SVM===============================================================================
classify5 = SVC()
#Train the model
classify5.fit(x_train_minmax, y_train)
# Use the model on the test data
y_predicted5 = classify5.predict(x_test_minmax)
print ("The accuracy score using the svm is  : {}".format(metrics.accuracy_score(y_test, y_predicted5)) )
print('Confusion Matrix  : {}'.format(confusion_matrix(y_test, y_predicted5)))
print('Accuracy  : {}'.format(accuracy_score(y_test, y_predicted5)))
print('Classification Report  : {}'.format(classification_report(y_test, y_predicted5)))
print('---------------------------------------------- ')



# Next use PCA ===============================================================================
#               Reference - https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# default solver is incredibly slow which is why it was changed to 'lbfgs'

# Make an instance of the Model
pca = PCA(.95)

#Fit the training data
pca.fit(x_train_minmax)

#transofrm the test & training data
x_train_pca = pca.transform(x_train_minmax)
x_test_pca = pca.transform(x_test_minmax)
#print the number of reduced attributes
print ("The data after PCA transformatio is----------------------------->")
print(x_train_pca.shape)
print(x_test_pca.shape)


#Apply Logistic Regression to the Transformed Data
classify6 = LogisticRegression(solver = 'lbfgs')
#Train the model
classify6.fit(x_train_pca, y_train)
# Use the model on the test data
y_predicted6 = classify6.predict(x_test_pca)
print ("The accuracy score using the PCA+Logist Regression is  : {}".format(metrics.accuracy_score(y_test, y_predicted6)) )
print('Confusion Matrix  : {}', confusion_matrix(y_test, y_predicted6))
print('Accuracy  : {}', accuracy_score(y_test, y_predicted6))
print('Classification Report  : {}', classification_report(y_test, y_predicted6 ))
print('---------------------------------------------- ')

