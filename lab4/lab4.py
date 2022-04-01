#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
print(data_breast_cancer['DESCR'])


# In[2]:


#NOWOTWÓR PIERSI


# In[3]:


X = data_breast_cancer['data']
print(X)


# In[4]:


y = data_breast_cancer['target']
print(y)


# In[5]:


#Zadanie 1


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


#Zadanie 2


# In[8]:


# tylko dla cech area, smoothness


# In[9]:


X = X.loc[:,["mean area","mean smoothness"]]


# In[10]:


import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#1. z funkcja straty hinge, bez skalowania
svm_clf_scalling_off = LinearSVC(C=1, loss="hinge", random_state=42)
svm_clf_scalling_off.fit(X_train.loc[:,["mean area","mean smoothness"]],y_train)
#1. po skalowaniu
svm_clf_scalling_on = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1,
                                 loss="hinge",
                                 random_state=42)),
    ])
svm_clf_scalling_on.fit(X_train.loc[:,["mean area","mean smoothness"]],y_train)


# In[11]:


from sklearn.metrics import accuracy_score


# In[12]:


acc_train_scall_off = svm_clf_scalling_off.score(X_train.loc[:,["mean area","mean smoothness"]],y_train)
acc_test_scall_off = svm_clf_scalling_off.score(X_test.loc[:,["mean area","mean smoothness"]],y_test)
acc_train_scall_on = svm_clf_scalling_on.score(X_train.loc[:,["mean area","mean smoothness"]],y_train)
acc_test_scall_on = svm_clf_scalling_on.score(X_test.loc[:,["mean area","mean smoothness"]],y_test)


# In[13]:


acc_list = [acc_train_scall_off, acc_test_scall_off, acc_train_scall_on, acc_test_scall_on ]


# In[14]:


print(acc_list)


# In[15]:


import pickle as pkl
fileObject = open("bc_acc.pkl", 'wb')
pkl.dump(acc_list, fileObject)
fileObject.close()


# In[16]:


#IRYSY


# In[17]:


data_iris = datasets.load_iris(as_frame=True)
print(data_iris['DESCR'])


# In[18]:


X = data_iris['data']
y = (data_iris['target'] == 2).astype(np.int8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


print(X)


# In[20]:


#1. z funkcja straty hinge, bez skalowania
svm_clf_scalling_off = LinearSVC(C=1, loss="hinge", random_state=42)
svm_clf_scalling_off.fit(X_train.iloc[:,[2,3]],y_train) #długość i szerokość płatka
#1. po skalowaniu
svm_clf_scalling_on = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1,
                                 loss="hinge",
                                 random_state=42)),
    ])
svm_clf_scalling_on.fit(X_train.iloc[:,[2,3]],y_train)


# In[21]:


acc_train_scall_off = svm_clf_scalling_off.score(X_train.iloc[:,[2,3]],y_train)
acc_test_scall_off = svm_clf_scalling_off.score(X_test.iloc[:,[2,3]],y_test)
acc_train_scall_on = svm_clf_scalling_on.score(X_train.iloc[:,[2,3]],y_train)
acc_test_scall_on = svm_clf_scalling_on.score(X_test.iloc[:,[2,3]],y_test)


# In[22]:


acc_list = [acc_train_scall_off, acc_test_scall_off, acc_train_scall_on, acc_test_scall_on ]


# In[23]:


print(acc_list)


# In[24]:


fileObject = open("iris_acc.pkl", 'wb')
pkl.dump(acc_list, fileObject)
fileObject.close()


# In[ ]:




