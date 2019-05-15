# -*- coding: utf-8 -*-
import csv
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
contents = []
#Read Data
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
X_test = np.asarray(contents,dtype=np.float64)
X_train = cont_np[:,:-1]
Y_train = cont_np[:,-1]
test_id = np.arange(0,X_test.shape[0])
#Shuffle
#np.random.shuffle(test_id)
#X_test = X_test[test_id]

#Normalize
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)
#PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
#SVM
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train,Y_train)
y_pred = svclassifier.predict(X_test)

with open('submission.csv',mode= 'w') as output_file:
    output_writer = csv.writer(output_file,delimiter=',')
    output_writer.writerow(["ID","Predicted"])
    for i in range(1,len(y_pred)+1):
        output_writer.writerow([i,int(y_pred[i-1])])
#Plot
'''
indices_0 = np.where(Y_train == 0)
indices_1 = np.where(Y_train == 1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[indices_0,0],X_train[indices_0,1],X_train[indices_0,2],c='r')
ax.scatter(X_train[indices_1,0],X_train[indices_1,1],X_train[indices_1,2],c='b')
plt.show()
'''

