# -*- coding: utf-8 -*-
"""
Import all seasons for the entire NBA
and attempt to predict playoff success using a MLP Classifier
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


#################### IMPORT DATASET ##############################
data = pd.read_csv('all_time_normal.csv', 
                   index_col=False,
                   skipinitialspace=True,
                   na_values="nan"
                   )
thisYear = pd.read_csv('2019_normal.csv', 
                   index_col=False,
                   skipinitialspace=True,
                   na_values="nan"
                   )

# fill in missing 3pt data
data.fillna(0, inplace=True)
og_data = data.copy()

# get the columns we're interested in
real_data = data.drop(columns=['Season','Lg','Tm','Playoffs','year'])
real_thisYear = thisYear.drop(columns=['Season','Lg','Tm','Year'])
teams= thisYear['Tm']
targets = data['Playoffs']

X = real_data.values
y = targets.values
ty = real_thisYear.values


############################ BINNING ##################################
X_d= np.zeros((len(X),len(X[0])))
for i in range(len(X)):
    for j in range(len(X[0])):
        if X[i][j] > -0.5:
            X_d[i][j] = -1
        elif X[i][j] > 0.5:
            X_d[i][j] = 0
        else:
            X_d[i][j] = 1
ty_d= np.zeros((len(ty),len(ty[0])))
for i in range(len(ty)):
    for j in range(len(ty[0])):
        if ty[i][j] > -0.5:
            ty_d[i][j] = -1
        elif ty[i][j] > 0.5:
            ty_d[i][j] = 0
        else:
            ty_d[i][j] = 1


##################### TRAINING ####################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y, test_size=0.25)

mlp = MLPClassifier(hidden_layer_sizes=(30,), learning_rate_init=0.01, max_iter=10000)
mlp.fit(X_train, y_train)

gnb = GaussianNB()
gnbpredict= gnb.fit(X_train, y_train).predict(X_test)

neigh = KNeighborsClassifier(n_neighbors=4)
knnpredict= neigh.fit(X_train, y_train).predict(X_test)

clf = SVC(gamma='auto')
svmpredict = clf.fit(X_train, y_train).predict(X_test)

dtc = DecisionTreeClassifier(random_state=0, splitter='random')
dtcpredict= dtc.fit(X_train_d, y_train_d).predict(X_test_d)

#################### PREDICTION ###################################

print("\n\n NEURAL NET RESULTS")
predictions = mlp.predict(X_test)
print(f"Accuracy is {accuracy_score(y_test, predictions)}")
inaccuracy= 0 
for x in range(len(predictions)):
    if predictions[x] != y_test[x]:
        inaccuracy += abs(predictions[x]-y_test[x])
        #print("predicted ", predictions[x], "Acutally", y_test[x])
print("Distance Accuracy is", 1-(inaccuracy/(len(predictions)*5)))


print("\n\n NAIVE BAYES RESULTS")
print(f"Accuracy is {accuracy_score(y_test, gnbpredict)}")
inaccuracy= 0 
for x in range(len(gnbpredict)):
    if gnbpredict[x] != y_test[x]:
        inaccuracy += abs(gnbpredict[x]-y_test[x])
        #print("predicted ", predictions[x], "Acutally", y_test[x])
print("Distance Accuracy is", 1-(inaccuracy/(len(gnbpredict)*5)))


print("\n\n K Nearest Neighbors RESULTS")
print(f"Accuracy is {accuracy_score(y_test, knnpredict)}")
inaccuracy= 0 
for x in range(len(knnpredict)):
    if knnpredict[x] != y_test[x]:
        inaccuracy += abs(knnpredict[x]-y_test[x])
        #print("predicted ", predictions[x], "Acutally", y_test[x])
print("Distance Accuracy is", 1-(inaccuracy/(len(knnpredict)*5)))


print("\n\n Support Vector Machine RESULTS")
print(f"Accuracy is {accuracy_score(y_test, svmpredict)}")
inaccuracy= 0 
for x in range(len(svmpredict)):
    if svmpredict[x] != y_test[x]:
        inaccuracy += abs(svmpredict[x]-y_test[x])
        #print("predicted ", predictions[x], "Acutally", y_test[x])
print("Distance Accuracy is", 1-(inaccuracy/(len(svmpredict)*5)))


print("\n\n Decision Tree RESULTS")
print(f"Accuracy is {accuracy_score(y_test_d, dtcpredict)}")
inaccuracy= 0 
for x in range(len(dtcpredict)):
    if dtcpredict[x] != y_test[x]:
        inaccuracy += abs(dtcpredict[x]-y_test[x])
        #print("predicted ", predictions[x], "Acutally", y_test[x])
print("Distance Accuracy is", 1-(inaccuracy/(len(dtcpredict)*5)))


# Predict for this year
print("\n\n Predictions for 2019 Playoffs")
print("\nTEAM: NN, NB,KNN,SVM, DT")
playoffs1= mlp.predict(real_thisYear)
playoffs2= gnb.predict(real_thisYear)
playoffs3= neigh.predict(real_thisYear)
playoffs4= clf.predict(real_thisYear)
playoffs5= dtc.predict(ty_d)
for k in range(len(teams)):
    print(teams[k], ":", playoffs1[k], ",", playoffs2[k], ",", playoffs3[k], ",", playoffs4[k], ",", playoffs5[k])
path= dtc.decision_path(ty_d)
