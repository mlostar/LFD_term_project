# -*- coding: utf-8 -*-
import csv
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
#Read Data
contents = []
with open('train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',',)
    next(csv_reader)
    for row in csv_reader:
            contents += [row]
cont_np= np.asarray(contents,dtype=np.float64)
contents = []
with open('test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',',)
    next(csv_reader)
    for row in csv_reader:
            contents += [row]

X_submission_test = np.asarray(contents,dtype=np.float64)        
X = cont_np[:,:-1]
Y = cont_np[:,-1]
#Outlier detection and removal
clf = IsolationForest( behaviour = 'new', random_state = 1, contamination= 'auto')
preds = clf.fit_predict(X)
for i in range(0,len(preds)):
    if preds[i] == -1:
        X = np.delete(X,(i),0)
        Y = np.delete(Y,(i),0)
X = cont_np[:-15,:-1]
Y = cont_np[:-15,-1]
X_test = cont_np[-15:,:-1]
Y_test = cont_np[-15:,-1]
#Scaling
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_submission_test = scaler.transform(X_submission_test)
#Feature Selection
svc = SVC(kernel='linear')
rfecv = RFECV(estimator=svc, step=1, cv=ShuffleSplit(n_splits=7, test_size=0.25, random_state=0),
              scoring='accuracy')
X = rfecv.fit_transform(X, Y)
print(max(rfecv.grid_scores_))
print(rfecv.n_features_)
y_pred = rfecv.predict(X_submission_test)
print(rfecv.score(X_test,Y_test))
with open('submissionRfecv.csv',mode= 'w') as output_file:
    output_writer = csv.writer(output_file,delimiter=',')
    output_writer.writerow(["ID","Predicted"])
    for i in range(1,len(y_pred)+1):
        output_writer.writerow([i,int(y_pred[i-1])])




