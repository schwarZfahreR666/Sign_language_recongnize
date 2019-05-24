from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt

file_name='test_features.txt'
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
s=0
right=0
clf1=joblib.load('svm_test')
Z=clf1.predict(X)
for n in range(len(y)):
    s=s+1
    if Z[n]==y[n]:
        right=right+1
a=right/s*100
print('accuracy:%.2f%%' % a)






