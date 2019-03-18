# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 00:32:15 2018

@author: Siddhartha Banerjee
"""

#Read the data and divide it into training and testing:
import numpy.random as numrandom
import pandas as pd
# Import the split & the various classifiers
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn import metrics

#Set location of the input data 
Location = r'/Users/sumitpal.singh/ML/data.csv'
# Split the data into columns and read
datainput = pd.read_csv(Location, names = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8','a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16','a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26','a27', 'a28', 'a29', 'a30','a31', 'a32', 'a33', 'a34', 'a35', 'a36','a37', 'a38', 'a39', 'a40','a41','a42'])
#Set the outcome and dedlete it
y = datainput['a42']
del datainput['a42']
# Split data into Test & Training set where test data is 30% & raining data is 70%
x_train, x_test, y_train, y_test = train_test_split(datainput, y, test_size = 0.3)

# Use Descision tree classifier on the training data
print('---------------------------------------------- ')
classify1 = DecisionTreeClassifier()
#Train the model
classify1.fit(x_train, y_train)
# Use the model on the test data
predicted1 = classify1.predict(x_test)
print ("The accuracy score using the Decision Tree is ->" )
print (metrics.accuracy_score(y_test, predicted1))
print('---------------------------------------------- ')

# Next use KNearest Neighbours 
classify2 = KNeighborsClassifier()
#Train the model
classify2.fit(x_train, y_train)
# Use the model on the test data
predicted2 = classify2.predict(x_test)
print ("The accuracy score using the K Nereast Neighbour is ->" )
print (metrics.accuracy_score(y_test, predicted2))  
print('---------------------------------------------- ')

# Next use NaiveBayes Classifier 
classify3 = BernoulliNB()
#Train the model
classify3.fit(x_train, y_train)
# Use the model on the test data
predicted3 = classify3.predict(x_test)
print ("The accuracy score using the Naive Bayes Classifier is ->" )
print (metrics.accuracy_score(y_test, predicted3)) 
print('---------------------------------------------- ')

# Next use RandomForest Classifier 
classify4 = RandomForestClassifier()
#Train the model
classify4.fit(x_train, y_train)
# Use the model on the test data
predicted4 = classify4.predict(x_test)
print ("The accuracy score using the RandomForest is ->" )
print (metrics.accuracy_score(y_test, predicted4)) 
print('---------------------------------------------- ')

# Next use SVM
classify5 = SVC()
#Train the model
classify5.fit(x_train, y_train)
# Use the model on the test data
predicted5 = classify5.predict(x_test)
print ("The accuracy score using the svm is ->" )
print (metrics.accuracy_score(y_test, predicted5)) 
print('---------------------------------------------- ')


