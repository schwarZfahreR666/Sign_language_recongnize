from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt

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



X = np.array(features)
y = np.array(labels)


clf1 = SVC(kernel='rbf')
clf1.fit(X,y)




def plot_estimator(estimator, X, y):
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.plot()
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:,0], X[:,1],c=y, cmap=plt.cm.brg)
    plt.xlabel('Petal.Length')
    plt.ylabel('Petal.Width')
    plt.show()
joblib.dump(clf1,'svm_test')

