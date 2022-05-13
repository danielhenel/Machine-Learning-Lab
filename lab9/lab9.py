#!/usr/bin/env python
# coding: utf-8

# In[1]:


#2.1
import pickle as pkl


# In[2]:


from sklearn.datasets import load_iris 
iris = load_iris(as_frame=True)


# In[3]:


import pandas as pd
pd.concat([iris.data, iris.target], axis=1).plot.scatter(
    x='petal length (cm)',
    y='petal width (cm)',
    c='target',
    colormap='viridis'
)


# In[ ]:





# In[4]:


X = iris.data.iloc[:, 2:4]
y = iris.target


# In[5]:


X


# In[6]:


y


# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


import numpy as np
from sklearn.linear_model import Perceptron

per_acc = []
per_wght = []

for i in range(3):
    Xi_train = X_train
    yi_train = (y_train == i).astype(int)
    Xi_test = X_test
    yi_test = (y_test == i).astype(int)
    per_clf = Perceptron()
    per_clf.fit(Xi_train, yi_train)
    per_acc.append((per_clf.score(Xi_train,yi_train),per_clf.score(Xi_test,yi_test)))
    per_wght.append((per_clf.intercept_[0],per_clf.coef_[0,0],per_clf.coef_[0,1]))


# In[9]:


per_acc
fileObject = open("per_acc.pkl", 'wb')
pkl.dump(per_acc, fileObject)
fileObject.close()


# In[10]:


per_wght
fileObject = open("per_wght.pkl", 'wb')
pkl.dump(per_wght, fileObject)
fileObject.close()


# In[11]:


#2.2


# In[12]:


X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
y = np.array([0,1, 1, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


#uczenie
per_clf = Perceptron()
per_clf.fit(X_train, y_train)
print(per_clf.predict([[0,0]]))
print(per_clf.predict([[0,1]]))
print(per_clf.predict([[1,0]]))
print(per_clf.predict([[1,1]]))


# In[14]:


per_clf.coef_


# In[15]:


#2.3
import tensorflow as tf
from tensorflow import keras
model = keras.models.Sequential()


# In[16]:


#warstwa wejściowa
model.add(tf.keras.layers.Flatten(input_dim = 2))


# In[17]:


#jedna warstwa ukryta
model.add(tf.keras.layers.Dense(2,input_dim = 2,  activation='tanh', use_bias=True))


# In[18]:


#warstwa wyjściowa
model.add(tf.keras.layers.Dense(1, activation="sigmoid", use_bias=True))


# In[19]:


model.summary()


# In[20]:


model.compile(loss='binary_crossentropy',
optimizer='SGD')


# In[21]:


history = model.fit(X_train, y_train, epochs=100, verbose=False)
print(history.history['loss'])


# In[22]:


model.get_weights()


# In[23]:


model.predict(X)


# In[24]:


model.predict(X)[1][0]


# In[25]:


act=['sigmoid','tanh','relu']
#opt = ['SGD','Adam']
loss_f = ['binary_crossentropy','MAE']


# In[26]:


def check_solution(y_pred,y):
    for i in range(len(y)):
        if y[i]==0:
            if y_pred[i][0] >= 0.05 or y_pred[i][0]<= -0.05: return False
        if y[i]==1:
            if y_pred[i][0] <= 0.95 or y_pred[i][0] >= 1.05: return False
    return True


# In[27]:


from random import randrange
while not check_solution(model.predict(X),y):
    model = keras.models.Sequential()
    #warstwa wejściowa
    model.add(tf.keras.layers.Flatten(input_dim = 2))
    #jedna warstwa ukryta
    model.add(tf.keras.layers.Dense(2,input_dim = 2,  activation=act[randrange(len(act))], use_bias=True))
    #warstwa wyjściowa
    model.add(tf.keras.layers.Dense(1, activation=act[randrange(len(act))], use_bias=True))
    model.compile(loss=loss_f[randrange(len(loss_f))], optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    model.fit(X, y, epochs=100, verbose=False)        
    print(model.predict(X))


# In[28]:


model.predict(X)


# In[29]:


mlp_xor_weights = model.get_weights()


# In[31]:


print(mlp_xor_weights)


# In[32]:


fileObject = open("mlp_xor_weights.pkl", 'wb')
pkl.dump(mlp_xor_weights, fileObject)
fileObject.close()


# In[ ]:




