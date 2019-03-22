# -*- coding: utf-8 -*-
"""
Import all seasons for the entire NBA
and attempt to predict playoff success using a MLP Classifier
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
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
real_thisYear = thisYear.drop(columns=['Season','Lg','Tm','year'])
teams= thisYear['Tm']
targets = data['Playoffs']


X = real_data.values
y = targets.values

##################### TRAINING ####################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

mlp = MLPClassifier(hidden_layer_sizes=(30,), learning_rate_init=0.01, max_iter=10000)
mlp.fit(X_train, y_train)

gnb = GaussianNB()
gnbpredict= gnb.fit(X_train, y_train).predict(X_test)

#################### PREDICTION ###################################

print("\n\nNEURAL NET RESULTS")
predictions = mlp.predict(X_test)
print(f"Accuracy is {accuracy_score(y_test, predictions)}")
inaccuracy= 0 
for x in range(len(predictions)):
    if predictions[x] != y_test[x]:
        inaccuracy += abs(predictions[x]-y_test[x])
        #print("predicted ", predictions[x], "Acutally", y_test[x])
print("Distance Accuracy is", 1-(inaccuracy/(len(predictions)*5)))


print("\n\nNAIVE BAYES RESULTS")
print(f"Accuracy is {accuracy_score(y_test, gnbpredict)}")
inaccuracy= 0 
for x in range(len(gnbpredict)):
    if gnbpredict[x] != y_test[x]:
        inaccuracy += abs(gnbpredict[x]-y_test[x])
        #print("predicted ", predictions[x], "Acutally", y_test[x])
print("Distance Accuracy is", 1-(inaccuracy/(len(gnbpredict)*5)))



# Predict for this year
print("\n\nPredictions for 2019 Playoffs")
print("\nTEAM: NN,NB")
playoffs1= mlp.predict(real_thisYear)
playoffs2= gnb.predict(real_thisYear)
for y in range(len(teams)):
    print(teams[y], ":", playoffs1[y], ",", playoffs2[y])
