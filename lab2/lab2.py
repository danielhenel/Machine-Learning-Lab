#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Przygotowanie danych
import numpy as np
import pickle as pkl
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


# In[2]:


mnist = fetch_openml('mnist_784', version=1)


# In[3]:


X = mnist['data']
y = mnist['target'].astype(np.uint8)


# In[4]:


print(X)


# In[5]:


X.info()


# In[6]:


print(y)


# In[7]:


y.info()


# In[8]:


print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[9]:


#Zbiór uczący i testowy


# In[13]:


#Posortowanie danych
y = y.sort_values()
index = y.index
X = X.reindex(index=index)


# In[14]:


#losowy podział
shuffle_index = np.random.permutation(70000)
X = X.reindex(index=shuffle_index)
y = y.reindex(index=shuffle_index)


# In[15]:


X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[16]:


y_train.values


# In[17]:


y_test.values


# In[18]:


#Uczenie - jedna klasa


# In[19]:


y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)
print(y_train_0)
print(np.unique(y_train_0))
print(len(y_train_0))


# In[20]:


sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)


# In[21]:


#4 - policz dokładność (accurancy)


# In[22]:


scoreTrain = sgd_clf.score(X_train,y_train_0)
scoreTest = sgd_clf.score(X_test,y_test_0)

acc = list()
acc.append(scoreTrain)
acc.append(scoreTest)

print(scoreTrain, scoreTest)
fileObject = open("sgd_acc.pkl", 'wb')
pkl.dump(acc, fileObject)
fileObject.close()


# In[23]:


#4 - walidacja krzyżowa dokładności modelu


# In[24]:


score = cross_val_score(sgd_clf, X_train, y_train_0,
cv=3, scoring="accuracy")


# In[25]:


print(score)
fileObject = open("sgd_cva.pkl", 'wb')
pkl.dump(score, fileObject)
fileObject.close()


# In[26]:


#Uczenie - wiele klas


# In[27]:


sgd_clf.fit(X_train, y_train)


# In[28]:


score = cross_val_score(sgd_clf, X_train, y_train,
cv=3, scoring="accuracy")
print(score)


# In[29]:


y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)


# In[30]:


conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)


# In[31]:


print(type(conf_mx))


# In[32]:


fileObject = open("sgd_cmx.pkl", 'wb')
pkl.dump(conf_mx, fileObject)
fileObject.close()


# In[ ]:





# In[ ]:




