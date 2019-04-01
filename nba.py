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
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from statsmodels import robust

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
# brute force
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

# Binning version 2
# discretize the data
def discretize(X, n_scale=1):

    for c in X.columns:
        loc = X[c].median()

        # median absolute deviation of the column
        scale = robust.mad(X[c])

        bins = [-np.inf, loc - (scale * n_scale),
                loc + (scale * n_scale), np.inf]
        X[c] = pd.cut(X[c], bins, labels=[-1, 0, 1])

    return X

# bin the data
binned_data = discretize(real_data)

##################### TRAINING ####################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y, test_size=0.25)

mlp = MLPClassifier(hidden_layer_sizes=(30,6), learning_rate_init=0.01, max_iter=10000)
mlp.fit(X_train, y_train)

gnb = GaussianNB()
gnbpredict= gnb.fit(X_train, y_train).predict(X_test)

neigh = KNeighborsClassifier(n_neighbors=4)
knnpredict= neigh.fit(X_train, y_train).predict(X_test)

clf = SVC(gamma='auto')
svmpredict = clf.fit(X_train, y_train).predict(X_test)

dtc = DecisionTreeClassifier(random_state=0, splitter='random')
dtcpredict= dtc.fit(X_train_d, y_train_d).predict(X_test_d)

mlr= MLPRegressor(hidden_layer_sizes=(30,6), learning_rate_init=0.01, max_iter=10000)
mlrpredict= mlr.fit(X_train, y_train).predict(X_test)

rForest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rForestPredict = rForest.fit(X_train, y_train).predict(X_test)

boost = AdaBoostClassifier(n_estimators=50)
boostPredict = boost.fit(X_train, y_train).predict(X_test)
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


print("\n\n MLR RESULTS")
inaccuracy= 0 
temp= 0
seeding= {29:5, 28:4, 27:3, 26:3, 25:2, 24:2, 23:2, 22:2, 21:1, 20:1, 19:1, 18:1, 17:1, 16:1, 15:1, 14:1, 13:0, 12:0, 11:0, 10:0, 9:0, 8:0, 7:0, 6:0, 5:0, 4:0, 3:0, 2:0, 1:0, 0:0}
for x in range(len(mlrpredict)):
    if mlrpredict[x] != y_test[x]:
        inaccuracy += abs(mlrpredict[x]-y_test[x])
        if mlrpredict[x] < 0:
            mlrpredict[x] = 0
        temp += abs(int(round(mlrpredict[x]))-y_test[x])
print("Seeding Accuracy is", 1-(temp/(len(mlrpredict))))
print("Regression Accuracy is", 1-(inaccuracy/(len(mlrpredict)*5)))

playoffs6= mlr.predict(real_thisYear)
playoffs7= np.argsort(np.argsort(mlr.predict(real_thisYear)))
for x in range(len(playoffs7)):
    playoffs6[x]= seeding[playoffs7[x]]
    
    
print("\n\n Random Forest RESULTS")  
print(f"Accuracy is {accuracy_score(y_test, rForestPredict)}")
inaccuracy= 0 
for x in range(len(rForestPredict)):
    if rForestPredict[x] != y_test[x]:
        inaccuracy += abs(rForestPredict[x]-y_test[x])
    
print("\n\n AdaBoost RESULTS")  
print(f"Accuracy is {accuracy_score(y_test, boostPredict)}")
inaccuracy= 0 
for x in range(len(boostPredict)):
    if boostPredict[x] != y_test[x]:
        inaccuracy += abs(boostPredict[x]-y_test[x])

# Predict for this year
print("\n\n Predictions for 2019 Playoffs")
print("\nTEAM: NN , NB ,KNN ,SVM , DT ,MLR      , RF , AB")
playoffs1= mlp.predict(real_thisYear) 
playoffs2= gnb.predict(real_thisYear)
playoffs3= neigh.predict(real_thisYear)
playoffs4= clf.predict(real_thisYear)
playoffs5= dtc.predict(ty_d)
playoffs8= mlr.predict(real_thisYear)
playoffs9= rForest.predict(real_thisYear)
playoffs10= boost.predict(real_thisYear)
for k in range(len(teams)):
    print(teams[k], ":", playoffs1[k], " ,", playoffs2[k], " ,", playoffs3[k], " ,", playoffs4[k], " ,", playoffs5[k], " ,", int(round(playoffs6[k])), " ,", int(round(playoffs8[k])), " ,", playoffs9[k], " ,", playoffs10[k])
