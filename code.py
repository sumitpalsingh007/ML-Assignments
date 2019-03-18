from aetypes import end

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
import csv
import glob


path = './data'

def construct_tree(data_file):
    data = pd.read_csv(data_file)
    mask = data.bug > 0
    data.loc[mask, 'bug'] = 1
    drop_columns = list(set(data.columns.values) - set(["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm",
                                                        "amc","max_cc","avg_cc","bug"]))
    data = data.drop(drop_columns, axis=1)

    for column in ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm",
                   "amc","max_cc","avg_cc"]:
        temp = pd.DataFrame(data[column])
        q1 = np.array(temp.quantile([0.25]))
        q3 = np.array(temp.quantile([0.75]))
        quantiled_data = np.array([q1[0],q3[0]])
        data_set = np.array(temp)
        #print(data_set)
        kmeans = KMeans(n_clusters = 2, random_state = 0, max_iter=600)
        #Execute KMEANS alogorithm - Fit
        y_kmeans_output = kmeans.fit(data_set)
        #Execute KMEANS predict
        y_kmeans_labels = kmeans.predict(data_set)
        #Store back the cluster value
        data[column] = y_kmeans_labels
        #extract the Kmeans centers & print     
        y_kmeans_centers = kmeans.cluster_centers_
        print(y_kmeans_centers)
    y_actual = data['bug']
    dtree = data.drop(['bug'], axis=1)
    #Split the data into test & training set - 70% training data, 30% test data, set random seed value = 0
    x_training, x_test, y_training, y_test = train_test_split(dtree, y_actual, test_size=0.3, random_state=0)
    #Calculate the entropy of the descision tree
    entropy = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    #execute the entropy on the training data
    entropy.fit(x_training, y_training)
    #test it on the test data
    y_entropy = entropy.predict(x_test)

    #calculate the various values
    matrix_entropy = metrics.confusion_matrix(y_test, y_entropy)
    accuracy_entropy = round(metrics.accuracy_score(y_test, y_entropy) * 100, 2)
    precision_entropy = round(metrics.precision_score(y_test, y_entropy),2)
    recall_entropy = round(metrics.recall_score(y_test, y_entropy),2)
    fmeasure_entropy = round(metrics.f1_score(y_test, y_entropy),2)

    print("Entropy")
    print("CF_Matrix -\n{0}".format(matrix_entropy))
    print("Accuracy = {0} %\nPrecision = {1}\nRecall = {2}\nFMeasure = {3}".format(accuracy_entropy,
                                                                                   precision_entropy,
                                                                                   recall_entropy,
                                                                                   fmeasure_entropy))
    #Gini Index
    gini = DecisionTreeClassifier()
    gini.fit(x_training, y_training)
    y_gini = gini.predict(x_test)

    #calculate the various values
    matrix_gini = metrics.confusion_matrix(y_test, y_gini)
    accuracy_gini = round(metrics.accuracy_score(y_test, y_gini) * 100, 2)
    precision_gini = round(metrics.precision_score(y_test, y_gini),2)
    recall_gini = round(metrics.recall_score(y_test, y_gini),2)
    fmeasure_gini = round(metrics.f1_score(y_test, y_gini),2)

    print("GINI")
    print("CFMatrix -\n{0}".format(matrix_entropy))
    print("Accuracy = {0} %\nPrecision = {1}\nRecall = {2}\nFMeasure = {3}".format(accuracy_gini,
                                                                                   precision_gini,
                                                                                   recall_gini,
                                                                                   fmeasure_gini))
    perfdata = [file, matrix_entropy, accuracy_entropy,precision_entropy,recall_entropy,fmeasure_entropy,matrix_gini, accuracy_gini,precision_gini,recall_gini,fmeasure_gini]
    with open('2017HT13042.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(perfdata)
data_files = glob.glob(path + '/*.csv')
csvData = ['FileName','matrix_entropy', 'accuracy_entropy','precision_entropy','recall_entropy','fmeasure_entropy','matrix_gini', 'accuracy_gini','precision_gini','recall_gini','fmeasure_gini']
with open('2017HT13042.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(csvData)
for file in data_files:
    construct_tree(file)