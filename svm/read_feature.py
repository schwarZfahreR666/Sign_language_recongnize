import numpy as np
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
print(labels.shape)
print(features.shape)
f.close()

