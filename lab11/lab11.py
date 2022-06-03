#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import time
import os
import pickle as pkl


# In[2]:


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()


# In[3]:


def build_model(n_hidden=1, n_neurons=25, optimizer="sgd", learning_rate=0.00001, momentum=0): 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=X_train.shape[1:]))
    
    for i in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    
    model.add(tf.keras.layers.Dense(1))
    opt = None
    
    #optimizers
    if optimizer == "sgd":
        opt = tf.keras.optimizers.get("sgd")
        opt.learning_rate = learning_rate
    elif optimizer == "nesterov":
        opt = tf.keras.optimizers.get("sgd")
        opt.learning_rate = learning_rate
        opt.nesterov = True
    elif optimizer == "momentum":
        opt = tf.keras.optimizers.get("sgd")
        opt.learning_rate = learning_rate
        opt.momentum = momentum
    elif optimizer == "adam":
        opt = tf.keras.optimizers.get("adam")
        opt.learning_rate = learning_rate
    
    model.compile(loss="mean_squared_error",
              optimizer=opt,
              metrics=["mean_absolute_error"])

    return model


# In[4]:


tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


# In[20]:


#eksperyment 1
lr_results = []
model = None
for lr in [0.000001,0.00001,0.0001]:
    model = build_model(learning_rate = lr)
    
    ts = str(int(time.time()))
    logdir = os.path.join(os.curdir, "tb_logs/{ts}_lr_{lr}".format(ts=ts,lr=str(lr))) 
    tensorboard_cb = tf.keras.callbacks.TensorBoard(logdir)
    
    es = tf.keras.callbacks.EarlyStopping(patience=10,min_delta=1.00,verbose=1)

    history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
    
    lr_results.append((lr,history.history['loss'][-1],history.history['mean_absolute_error'][-1]))
    
fileObject = open("lr.pkl", 'wb')
pkl.dump(lr_results, fileObject)
fileObject.close()
model.summary()


# In[21]:


lr_results


# In[8]:


## analiza tensorboard


# In[10]:


#eksperyment 2
hl_results = []
for hl in [0,1,2,3]:
    model = build_model(n_hidden = hl)
    
    ts = str(int(time.time()))
    logdir = os.path.join(os.curdir, "tb_logs/{ts}_hl_{hl}".format(ts=ts,hl=str(hl))) 
    tensorboard_cb = tf.keras.callbacks.TensorBoard(logdir)
    
    es = tf.keras.callbacks.EarlyStopping(patience=10,min_delta=1.00,verbose=1)

    history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
    
    hl_results.append((hl,history.history['loss'][-1],history.history['mean_absolute_error'][-1]))

fileObject = open("hl.pkl", 'wb')
pkl.dump(hl_results, fileObject)
fileObject.close()


# In[11]:


hl_results


# In[12]:


#eksperyment 3
nn_results = []
for nn in [5,25,125]:
    model = build_model(n_neurons = nn)
    
    ts = str(int(time.time()))
    logdir = os.path.join(os.curdir, "tb_logs/{ts}_nn_{nn}".format(ts=ts,nn=str(nn))) 
    tensorboard_cb = tf.keras.callbacks.TensorBoard(logdir)
    
    es = tf.keras.callbacks.EarlyStopping(patience=10,min_delta=1.00,verbose=1)

    history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
    
    nn_results.append((nn,history.history['loss'][-1],history.history['mean_absolute_error'][-1]))

fileObject = open("nn.pkl", 'wb')
pkl.dump(nn_results, fileObject)
fileObject.close()


# In[17]:


nn_results


# In[13]:


#eksperyment 4
opt_results = []
for opt in ['sgd', 'nesterov', 'momentum', 'adam']:
    model = build_model(optimizer = opt, momentum = 0.5)
    
    ts = str(int(time.time()))
    logdir = os.path.join(os.curdir, "tb_logs/{ts}_opt_{opt}".format(ts=ts,opt=opt)) 
    tensorboard_cb = tf.keras.callbacks.TensorBoard(logdir)
    
    es = tf.keras.callbacks.EarlyStopping(patience=10,min_delta=1.00,verbose=1)

    history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
    
    opt_results.append((opt,history.history['loss'][-1],history.history['mean_absolute_error'][-1]))

fileObject = open("opt.pkl", 'wb')
pkl.dump(opt_results, fileObject)
fileObject.close()


# In[18]:


opt_results


# In[14]:


#eksperyment 5
mom_results = []
for mom in [0.1,0.5,0.9]:
    model = build_model(momentum = mom)
    
    ts = str(int(time.time()))
    logdir = os.path.join(os.curdir, "tb_logs/{ts}_mom_{mom}".format(ts=ts,mom=str(mom))) 
    tensorboard_cb = tf.keras.callbacks.TensorBoard(logdir)
    
    es = tf.keras.callbacks.EarlyStopping(patience=10,min_delta=1.00,verbose=1)

    history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
    
    mom_results.append((mom,history.history['loss'][-1],history.history['mean_absolute_error'][-1]))

fileObject = open("mom.pkl", 'wb')
pkl.dump(mom_results, fileObject)
fileObject.close()


# In[19]:


mom_results


# In[23]:


param_distribs = {
"model__n_hidden": [0,1,2,3],
"model__n_neurons": [5,25,125],
"model__learning_rate": [0.000001,0.00001,0.0001],
"model__optimizer": ['sgd', 'nesterov', 'momentum', 'adam'],
"model__momentum": [0.1,0.5,0.9]
}


# In[24]:


import scikeras
from scikeras.wrappers import KerasRegressor

es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)

keras_reg = KerasRegressor(build_model, callbacks=[es])


# In[25]:


from sklearn.model_selection import RandomizedSearchCV
rnd_search_cv = RandomizedSearchCV(keras_reg,
param_distribs,
n_iter=30,
cv=3,
verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_split=0.1)


# In[26]:


rnd_search_cv.best_params_


# In[27]:


fileObject = open("rnd_search.pkl", 'wb')
pkl.dump(rnd_search_cv.best_params_, fileObject)
fileObject.close()

