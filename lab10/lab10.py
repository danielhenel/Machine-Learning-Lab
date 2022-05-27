#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() 
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# In[3]:


X_train= X_train / 255.0
X_test = X_test / 255.0


# In[4]:


import matplotlib.pyplot as plt 
plt.imshow(X_train[142], cmap="binary") 
plt.axis('off')
plt.show()


# In[5]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
               "sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[142]]


# In[6]:


import os
root_logdir = os.path.join(os.curdir, "image_logs") 
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") 
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[7]:


from tensorflow import keras
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])




# In[8]:


model.summary()


# In[9]:


tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)


# In[ ]:


history = model.fit(X_train, y_train, epochs=20,
                    validation_split=0.1, callbacks=[tensorboard_cb])


# In[ ]:


image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()


# In[ ]:


#get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


#get_ipython().run_line_magic('tensorboard', '--logdir ./image_logs')


# In[ ]:


model.save('fashion_clf.h5')


# In[ ]:


#regression


# In[ ]:


from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()


# In[ ]:


X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,y_train_full, random_state=42)


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# In[ ]:


model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation="softmax", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(1))

model.compile(loss="mean_squared_error",
              optimizer="sgd")


# In[ ]:


model.summary()


# In[ ]:


root_logdir = os.path.join(os.curdir, "housing_logs") 
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[ ]:


es = tf.keras.callbacks.EarlyStopping(patience=5,min_delta=0.01,verbose=1)


# In[ ]:


history = model.fit(X_train, y_train, epochs=100,validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[ ]:


model.save('reg_housing_1.h5')


# In[ ]:


#get_ipython().run_line_magic('tensorboard', '--logdir ./housing_logs')


# In[ ]:


# model v2


# In[ ]:


model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation="softmax", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(100, activation="softmax", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(1))

model.compile(loss="mean_squared_error",
              optimizer="sgd")

model.summary()

root_logdir = os.path.join(os.curdir, "housing_logs/housing_logs_2") 
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

es = tf.keras.callbacks.EarlyStopping(patience=5,min_delta=0.01,verbose=1)

history = model.fit(X_train, y_train, epochs=100,validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[ ]:


model.save('reg_housing_2.h5')


# In[ ]:


#get_ipython().run_line_magic('tensorboard', '--logdir ./housing_logs/housing_logs_2')


# In[ ]:


# model v3


# In[ ]:


model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation="softmax", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(300, activation="relu", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(100, activation="tanh", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(1))

model.compile(loss="mean_squared_error",
              optimizer="sgd")

model.summary()

root_logdir = os.path.join(os.curdir, "housing_logs/housing_logs_3") 
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

es = tf.keras.callbacks.EarlyStopping(patience=5,min_delta=0.01,verbose=1)

history = model.fit(X_train, y_train, epochs=100,validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[ ]:


model.save('reg_housing_3.h5')


# In[ ]:


#get_ipython().run_line_magic('tensorboard', '--logdir ./housing_logs/housing_log')


# In[ ]:


# model v4


# In[ ]:


model = keras.models.Sequential()
model.add(keras.layers.Dense(300, activation="relu", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(3000, activation="sigmoid", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(1000, activation="relu", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(1))

model.compile(loss="mean_squared_error",
              optimizer="sgd")

model.summary()

root_logdir = os.path.join(os.curdir, "housing_logs/housing_logs_4") 
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

es = tf.keras.callbacks.EarlyStopping(patience=5,min_delta=0.01,verbose=1)

history = model.fit(X_train, y_train, epochs=100,validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[ ]:


model.save('reg_housing_4.h5')


# In[ ]:


#get_ipython().run_line_magic('tensorboard', '--logdir ./housing_logs/housing_logs_4')


# In[ ]:




