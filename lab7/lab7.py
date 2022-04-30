#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml 
import numpy as np
mnist = fetch_openml('mnist_784', version=1, as_frame=False) 
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[2]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
silhouette_scores = []
kmeans_10 = None

for i in range(8,13):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit_predict(X)
    if i == 10:
        kmeans_10 = kmeans
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))


# In[3]:


print(silhouette_scores)


# In[4]:


import pickle as pkl
fileObject = open("kmeans_sil.pkl", 'wb')
pkl.dump(silhouette_scores, fileObject)
fileObject.close()


# In[5]:


from sklearn.metrics import confusion_matrix
conMatrix = confusion_matrix(y, kmeans_10.predict(X))
theBiggest = set()
for row in conMatrix:
    theBiggest.add(np.argmax(row))
theBiggest = list(theBiggest)
theBiggest.sort()
print(theBiggest)
print(conMatrix)


# In[6]:


fileObject = open("kmeans_argmax.pkl", 'wb')
pkl.dump(theBiggest, fileObject)
fileObject.close()


# In[7]:


#DBSCAN


# In[8]:


#Policz odległości dla pierwszych 300 elementów ze zbioru X
eps = set()
for i in range(300):
    #for j in range (300):
    for j in range(len(X)):
        if i!=j:
            eps.add(np.linalg.norm(X[i]-X[j]))
eps = list(eps)
eps.sort()
eps = eps [0:10]
print(eps)  

fileObject = open("dist.pkl", 'wb')
pkl.dump(eps, fileObject)
fileObject.close()


# In[11]:


dbscan_len = []
from sklearn.cluster import DBSCAN
s = (eps[0] + eps[1] + eps[2]) / 3
i = s
while i <= s+0.10*s:
    dbscan = DBSCAN(eps=i)
    dbscan.fit(X)
    dbscan_len.append(len(set(dbscan.labels_)))
    i += 0.04*s
    
fileObject = open("dbscan_len.pkl", 'wb')
pkl.dump(dbscan_len, fileObject)
fileObject.close()


# In[12]:


print(dbscan_len)

