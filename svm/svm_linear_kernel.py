from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

file_name='train_features.txt'
f=open(file_name)
feas=f.readlines()
feature_list=[]
label_list=[]
for fea in feas:
    fe_sp=fea.split('][')
    feature=np.array(fe_sp[0].strip('[').split(',')).astype(np.float32)
    label=np.array(fe_sp[1].replace(']','')).astype(np.int64)
    feature_list.append(feature)
    label_list.append(label)
features=np.array(feature_list).astype(np.float32)
labels=np.array(label_list).astype(np.int64)



X = features
y = labels

svc = SVC()
parameters = [
              {
              'C': [1,3,5,7,9,11,13,15,17,19],
              'gamma': [0.00001,0.0001,0.001,0.01,0.1,1,10,100],
              'kernel': ['rbf']
              },
              {
              'C': [1,3,5,7,9,11,13,15,17,19],
              'kernel': ['linear']
              }
              ]
clf1 = GridSearchCV(svc, parameters, cv=5, n_jobs=-1)

clf1.fit(X,y)




print(clf1.best_params_)
best_model = clf1.best_estimator_
joblib.dump(best_model,'svm_test')

