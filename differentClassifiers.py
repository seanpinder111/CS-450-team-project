import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm


data = pd.read_csv('all_time_normal.csv', 
                   index_col=False,
                   skipinitialspace=True,
                   na_values="nan"
                   )
data.fillna(0, inplace=True)
og_data = data.copy()

thisYear = pd.read_csv('2019_normal.csv', 
                   index_col=False,
                   skipinitialspace=True,
                   na_values="nan"
                   )


real_data = data.drop(columns=['Season','Lg','Tm','Playoffs'])
real_thisYear = thisYear.drop(columns=['Season','Lg','Tm'])
teams = og_data['Tm']
targets = data['Playoffs']


X = real_data.values
y = targets.values

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X,y)

##################### TRAINING ####################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

rForest= RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rForest.fit(X_train, y_train)
#################### PREDICTION ###################################
predictions = rForest.predict(X_test)
print(f"Accuracy is {accuracy_score(y_test, predictions)}")
inaccuracy= 0 
for x in range(len(predictions)):
    if predictions[x] != y_test[x]:
        inaccuracy += abs(predictions[x]-y_test[x])
print("Distance Accuracy is", 1-(inaccuracy/(len(predictions)*5)))

playoffs= rForest.predict(real_thisYear)
for y in range(len(playoffs)):
    print(teams[y], ":", playoffs[y])
    
 ##################### TRAINING ####################################
svmClassifier = svm.SVC(gamma='scale')
svmClassifier.fit(X_train, y_train) 
#################### PREDICTION ###################################
predictions = svmClassifier.predict(X_test)
print(f"Accuracy is {accuracy_score(y_test, predictions)}")
inaccuracy= 0 
for x in range(len(predictions)):
    if predictions[x] != y_test[x]:
        inaccuracy += abs(predictions[x]-y_test[x])
print("Distance Accuracy is", 1-(inaccuracy/(len(predictions)*5)))

playoffs= svmClassifier.predict(real_thisYear)
for y in range(len(playoffs)):
    print(teams[y], ":", playoffs[y])

