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

#Remove outliers
clf = IsolationForest( behaviour = 'new', max_samples=120, random_state = 1, contamination= 'auto')
preds = clf.fit_predict(X)
index = []
for i in range(0,len(preds)):
    if preds[i] == -1:
        X = np.delete(X,(i),0)
        Y = np.delete(Y,(i),0)

#PCA
pca = PCA(.40)
X = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
X_submission_test = pca.transform(X_submission_test)

#Scale
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_submission_test = scaler.transform(X_submission_test)
#Cross Validation function for best params
selected_kernel = 'rbf'
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
def rfc_param_selection(X,y,cv,classifier):
   estimators = [10,30,50,70,100]
   max_depth = [2,3,4,5,6,7,8]
   param_grid = {'n_estimators' : estimators,'max_depth' : max_depth}
   grid_search = GridSearchCV(classifier, param_grid, cv=cv)
   grid_search.fit(X, y)
   grid_search.best_params_
   return grid_search.best_params_,grid_search.best_score_,grid_search.cv_results_
def kneigh_param_selection(X,y,cv):
    neighbors = [3,4,5,6,7]
    weights = ['uniform','distance']
    param_grid = {'n_neighbors' : neighbors,'weights' : weights}
    grid_search = GridSearchCV(KNeighborsClassifier(),param_grid,cv= cv)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_,grid_search.best_score_,grid_search.cv_results_
def bagging_param_selection(X,y,cv,classifier):
    max_samples = [0.3,0.4,0.5,0.6,0.7]
    max_features = [0.3,0.4,0.5,0.6,0.7]
    param_grid = {'max_samples' : max_samples,'max_features' : max_features}
    grid_search = GridSearchCV(BaggingClassifier(classifier),param_grid,cv = cv)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_,grid_search.best_score_,grid_search.cv_results_
def sgd_param_selection(X,y,cv):
    alpha = [0.000001,0.00001,0.0001,0.001,0.01,1,10]
    penalty = ['l2','l1','elasticnet']
    loss = ['hinge','log','modified_huber','squared_hinge']
    max_iter = [100000]
    tol = [1e-4]
    param_grid = {'alpha' : alpha,'penalty' : penalty,'loss' : loss,'max_iter' : max_iter,'tol' : tol}
    grid_search = GridSearchCV(linear_model.SGDClassifier(),param_grid,cv = cv)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_,grid_search.best_score_,grid_search.cv_results_

#Cross Validation

#Random forest
print("Random Forest")
best_params,best_score,cv_results = rfc_param_selection(X,Y,ShuffleSplit(n_splits=6, test_size=0.33, random_state=0),RandomForestClassifier())
print(best_params)
print(best_score)
print(mean(cv_results["mean_test_score"]))

#Kneighbors
print("Kneighbors")
best_params,best_score,cv_results = kneigh_param_selection(X,Y,ShuffleSplit(n_splits=6, test_size=0.33, random_state=0))
print(best_params)
print(best_score)
print(mean(cv_results["mean_test_score"]))
bestKN = KNeighborsClassifier(n_neighbors = best_params['n_neighbors'],weights = best_params['weights']) 

#Bagging + Kneighbors
print("Bagging + Kneighbors")
best_params,best_score,cv_results = bagging_param_selection(X,Y,ShuffleSplit(n_splits=6, test_size=0.33, random_state=0),bestKN)
print(best_params)
print(best_score)
print(mean(cv_results["mean_test_score"]))

#SVC
print("SVC")
best_params,best_score,cv_results = svc_param_selection(X,Y,ShuffleSplit(n_splits=6, test_size=0.33, random_state=0))
print(best_params)
print(best_score)
print(mean(cv_results["mean_test_score"]))
best_svc = SVC(kernel = selected_kernel, C = best_params["C"],gamma = best_params["gamma"])

#Bagging + SVC
print("Bagging + SVC")
best_params,best_score,cv_results = bagging_param_selection(X,Y,ShuffleSplit(n_splits=6, test_size=0.33, random_state=0),best_svc)
print(best_params)
print(best_score)
print(mean(cv_results["mean_test_score"]))
#SGD
print("SGD")
best_params,best_score,cv_results = sgd_param_selection(X,Y,ShuffleSplit(n_splits=6, test_size=0.33, random_state=0))
print(best_params)
print(best_score)
print(mean(cv_results["mean_test_score"]))
best_sgd = linear_model.SGDClassifier(alpha = best_params["alpha"],penalty = best_params["penalty"],loss = best_params["loss"])
#Bagging + SGD
print("Bagging + SGD")
best_params,best_score,cv_results = bagging_param_selection(X,Y,ShuffleSplit(n_splits=6, test_size=0.33, random_state=0),best_sgd)
print(best_params)
print(best_score)
print(mean(cv_results["mean_test_score"]))

#Submission
'''
#SVC - Best 6.125
classifier = SVC(kernel =selected_kernel,C =best_params["C"],gamma=best_params["gamma"],probability=True)
classifier.fit(X,Y)
'''
'''
#Random Forest - Best 0.55
classifier = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
classifier.fit(X,Y)
'''

#KNeighbors
classifier = KNeighborsClassifier(n_neighbors=4,weights='uniform')
classifier.fit(X, Y)

'''
#Bagging + Neigbors
classifier = BaggingClassifier(bestKN,max_samples = best_params["max_samples"],max_features =best_params["max_features"])
classifier.fit(X,Y)
'''
'''
#GaussianNB
classifier = GaussianNB()
classifier.fit(X,Y)
'''
'''
#Bagging + SVC
classifier = BaggingClassifier(best_svc, max_samples = best_params["max_samples"],max_features =best_params["max_features"])
classifier.fit(X,Y)
'''
'''
#Bagging + SGD
classifier = BaggingClassifier(best_sgd,max_samples = best_params["max_samples"],max_features =best_params["max_features"])
classifier.fit(X,Y)
'''
y_pred = classifier.predict(X_submission_test)
with open('submission.csv',mode= 'w') as output_file:
    output_writer = csv.writer(output_file,delimiter=',')
    output_writer.writerow(["ID","Predicted"])
    for i in range(1,len(y_pred)+1):
        output_writer.writerow([i,int(y_pred[i-1])])


