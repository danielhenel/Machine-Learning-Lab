#!/usr/bin/env python
# coding: utf-8

# In[1]:


#2.1


# In[2]:


import tensorflow_datasets as tfds
[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
    "tf_flowers",
split=["train[:10%]", "train[10%:25%]", "train[25%:]"], as_supervised=True,
with_info=True)


# In[3]:


info


# In[4]:


class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples


# In[5]:


class_names


# In[6]:


n_classes


# In[7]:


dataset_size


# In[8]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
index = 0
sample_images = train_set_raw.take(9) 
for image, label in sample_images:
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label])) 
    plt.axis("off")
plt.show(block=False)


# In[9]:


#2.2.1


# In[10]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224]) 
    return resized_image, label


# In[11]:


import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


# In[12]:


batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1) 
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[13]:


plt.figure(figsize=(8, 8)) 
sample_batch = train_set.take(1)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1) 
        plt.imshow(X_batch[index]/255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]])) 
        plt.axis("off")
plt.show(block=False)


# In[14]:


#2.2.2


# In[15]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Rescaling(scale=1./127.5, offset=-1))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=7, input_shape=[224, 224, 1],padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=5))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(25,activation="relu"))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))


# In[16]:


opt = tf.keras.optimizers.get("sgd")
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])


# In[17]:


history = model.fit(train_set, epochs=10, validation_data=valid_set)


# In[18]:


acc_train = model.evaluate(train_set)[1]
acc_test = model.evaluate(test_set)[1]
acc_valid = model.evaluate(valid_set)[1]
simple_cnn_acc = (acc_train, acc_valid, acc_test)


# In[19]:


import pickle as pkl
fileObject = open("simple_cnn_acc.pkl",'wb')
pkl.dump(simple_cnn_acc, fileObject)
fileObject.close()
model.summary()


# In[20]:


#2.3 


# In[21]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


# In[22]:


batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[23]:


plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index] / 2 + 0.5)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")
plt.show(block=False)


# In[24]:


#2.3.2


# In[25]:


base_model = tf.keras.applications.xception.Xception(weights="imagenet",
                                                     include_top=False)


# In[26]:


for index, layer in enumerate(base_model.layers):
    print(index, layer.name)


# In[27]:


inputs = tf.keras.Input(shape=(3,))
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(10, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)


# In[28]:


for layer in base_model.layers:
    layer.trainable = False


# In[29]:


opt = tf.keras.optimizers.get("sgd")
model.compile(loss="sparse_categorical_crossentropy",
optimizer=opt, metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=5)


# In[30]:


for layer in base_model.layers:
    layer.trainable = True


# In[31]:


opt = tf.keras.optimizers.get("sgd")
model.compile(loss="sparse_categorical_crossentropy",
optimizer=opt, metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=10)


# In[ ]:


acc_train = model.evaluate(train_set)[1]
acc_test = model.evaluate(test_set)[1]
acc_valid = model.evaluate(valid_set)[1]
xception_acc = (acc_train, acc_valid, acc_test)


# In[ ]:


fileObject = open("xception_acc.pkl",'wb')
pkl.dump(xception_acc, fileObject)
fileObject.close()
model.summary()


# In[ ]:




