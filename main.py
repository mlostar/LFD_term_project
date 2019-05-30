# -*- coding: utf-8 -*-
import csv
import numpy as np
from statistics import mean
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,IsolationForest
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from skfeature.function.similarity_based import fisher_score as fs
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

#Scaling
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_submission_test = scaler.transform(X_submission_test)
#Feature Selection
svc = SVC(kernel='linear')
rfecv = RFECV(estimator=svc, step=1, cv=ShuffleSplit(n_splits=10, test_size=0.25, random_state=0),n_jobs = -1,
              scoring='accuracy')
X = rfecv.fit_transform(X, Y)
print(rfecv.n_features_)
y_pred = rfecv.predict(X_submission_test)
with open('submissionRfecv.csv',mode= 'w') as output_file:
    output_writer = csv.writer(output_file,delimiter=',')
    output_writer.writerow(["ID","Predicted"])
    for i in range(1,len(y_pred)+1):
        output_writer.writerow([i,int(y_pred[i-1])])

X_submission_test = rfecv.transform(X_submission_test)
#Cross Valid
selected_kernel = 'linear'
def svc_param_selection(X, y, cv):
    Cs = [0.00000000001,0.0000000001,0.000000001,0.00000001,0.0000001,0.000001,0.000001,0.00001,0.0001,0.001, 0.01, 0.1, 1, 10,100]
    gammas = [0.00000000001,0.0000000001,0.000000001,0.00000001,0.0000001,0.000001,0.000001,0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100,1000]
    coefs = [0.0001,0.001, 0.01, 0.1, 1, 10,100,1000]
    max_iter = [1000000]
    tol = [1e-3]
    param_grid = {'C': Cs, 'gamma' : gammas,'coef0' : coefs,'max_iter' : max_iter,'tol' : tol}
    grid_search = GridSearchCV(SVC(kernel=selected_kernel), param_grid, cv=cv)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_,grid_search.best_score_,grid_search.cv_results_
def bagging_param_selection(X,y,cv,classifier):
    max_samples = [0.3,0.4,0.5,0.6,0.7,0.8]
    max_features = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    param_grid = {'max_samples' : max_samples,'max_features' : max_features}
    grid_search = GridSearchCV(BaggingClassifier(classifier),param_grid,cv = cv)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_,grid_search.best_score_,grid_search.cv_results_

#SVC
print("SVC")
best_params,best_score,cv_results = svc_param_selection(X,Y,ShuffleSplit(n_splits=10, test_size=0.25, random_state=0))
print(best_params)
print(best_score)
print(mean(cv_results["mean_test_score"]))
best_svc = SVC(kernel = selected_kernel, C = best_params["C"],gamma = best_params["gamma"])

#Bagging + SVC
print("Bagging + SVC")
best_params,best_score,cv_results = bagging_param_selection(X,Y,ShuffleSplit(n_splits=10, test_size=0.25, random_state=0),best_svc)
print(best_params)
print(best_score)
print(mean(cv_results["mean_test_score"]))

#Bagging + SVC
clf = BaggingClassifier(best_svc,max_samples = best_params["max_samples"],max_features =best_params["max_features"])
clf.fit(X,Y)

#Submission
y_pred = clf.predict(X_submission_test)
with open('submissionBagging.csv',mode= 'w') as output_file:
    output_writer = csv.writer(output_file,delimiter=',')
    output_writer.writerow(["ID","Predicted"])
    for i in range(1,len(y_pred)+1):
        output_writer.writerow([i,int(y_pred[i-1])])


