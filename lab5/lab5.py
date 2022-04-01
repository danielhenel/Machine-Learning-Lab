#!/usr/bin/env python
# coding: utf-8

# In[1]:


import graphviz 
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
import pickle as pkl
import numpy as np
import pandas as pd


# In[2]:


# KLASYFIKACJA breast_cancer


# In[3]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True) 


# In[4]:


X = data_breast_cancer['data']


# In[5]:


y = data_breast_cancer['target']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


#max_depth = 2
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_train, y_train)
f1Train = f1_score(y_train, tree_clf.predict(X_train))
f1Test = f1_score(y_test, tree_clf.predict(X_test))
print(f1Train)
print(f1Test)


# In[8]:


#max_depth = 3 - NAJLEPSZY WYNIK DLA ZBIORU TESTOWEGO
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)
f1Train = f1_score(y_train, tree_clf.predict(X_train))
f1Test = f1_score(y_test, tree_clf.predict(X_test))
print(f1Train)
print(f1Test)


# In[9]:


#graf dla najlepszego wyniku
f = "data_breast_cancer_tree.dot"
export_graphviz(
tree_clf,
out_file=f,
feature_names=data_breast_cancer.feature_names,
class_names=[str(num)+", "+name
for num,name in
zip(set(data_breast_cancer.target),
data_breast_cancer.target_names)],
rounded=True,
filled=True)
print(f)


graphviz.render('dot', 'png', filepath=f, outfile="bc.png")


# In[10]:


#lista dla najlepszego wyniku
depth = 3
accTrain = accuracy_score(y_train, tree_clf.predict(X_train))
accTest = accuracy_score(y_test, tree_clf.predict(X_test))
bc_list = [depth, f1Train, f1Test, accTrain, accTest]

fileObject = open("f1acc_tree.pkl", 'wb')
pkl.dump(bc_list, fileObject)
fileObject.close()

print(bc_list)


# In[11]:


#max_depth = 4 - wartość f1 dla zbioru testowego spadła
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)
f1Train = f1_score(y_train, tree_clf.predict(X_train))
f1Test = f1_score(y_test, tree_clf.predict(X_test))
print(f1Train)
print(f1Test)


# In[12]:


#max_depth = 5
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)
f1Train = f1_score(y_train, tree_clf.predict(X_train))
f1Test = f1_score(y_test, tree_clf.predict(X_test))
print(f1Train)
print(f1Test)


# In[13]:


#max_depth = 6 - wartość f1 dla zbioru testowego spadła jeszcze bardziej
tree_clf = DecisionTreeClassifier(max_depth=6, random_state=42)
tree_clf.fit(X_train, y_train)
f1Train = f1_score(y_train, tree_clf.predict(X_train))
f1Test = f1_score(y_test, tree_clf.predict(X_test))
print(f1Train)
print(f1Test)


# In[14]:


#max_depth = 7
tree_clf = DecisionTreeClassifier(max_depth=7, random_state=42)
tree_clf.fit(X_train, y_train)
f1Train = f1_score(y_train, tree_clf.predict(X_train))
f1Test = f1_score(y_test, tree_clf.predict(X_test))
print(f1Train)
print(f1Test)


# In[15]:


data_breast_cancer.feature_names[2:]


# In[16]:


data_breast_cancer.target


# In[17]:


#max_depth = 8
tree_clf = DecisionTreeClassifier(max_depth=8, random_state=42)
tree_clf.fit(X_train, y_train)
f1Train = f1_score(y_train, tree_clf.predict(X_train))
f1Test = f1_score(y_test, tree_clf.predict(X_test))
print(f1Train)
print(f1Test)


# In[18]:


#REGRESJA


# In[19]:


size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1,1)


# In[22]:


#max_depth = 2
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
tree_clf = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_clf.fit(X_train, y_train)
mse1Train = mean_squared_error(y_train, tree_clf.predict(X_train))
mse1Test = mean_squared_error(y_test, tree_clf.predict(X_test))
print(mse1Train)
print(mse1Test)


# In[23]:


import matplotlib.pyplot as plt
plt.plot(X,y,'ro')
temp_X_train = np.sort(X_train, axis=0)
plt.plot(temp_X_train, tree_clf.predict(temp_X_train))


# In[24]:


#max_depth = 3
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
tree_clf = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)
mse1Train = mean_squared_error(y_train, tree_clf.predict(X_train))
mse1Test = mean_squared_error(y_test, tree_clf.predict(X_test))
print(mse1Train)
print(mse1Test)


# In[25]:


plt.plot(X,y,'ro')
temp_X_train = np.sort(X_train, axis=0)
plt.plot(temp_X_train, tree_clf.predict(temp_X_train))


# In[26]:


#max_depth = 4 #NAJLEPSZY WYNIK: najmniejsza wartość MSE dla zbioru testowego
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
tree_clf = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)
mse1Train = mean_squared_error(y_train, tree_clf.predict(X_train))
mse1Test = mean_squared_error(y_test, tree_clf.predict(X_test))
print(mse1Train)
print(mse1Test)


# In[27]:


plt.plot(X,y,'ro')
temp_X_train = np.sort(X_train, axis=0)
plt.plot(temp_X_train, tree_clf.predict(temp_X_train))


# In[28]:


#graf dla najlepszego wyniku
f = "reg_tree.dot"
export_graphviz(
tree_clf,
out_file=f,
#feature_names=data_breast_cancer.feature_names,
#class_names=[str(num)+", "+name
#for num,name in
#zip(set(data_breast_cancer.target),
#data_breast_cancer.target_names)],
#rounded=True,
#filled=True
)
print(f)


graphviz.render('dot', 'png', filepath=f, outfile="reg.png")


# In[29]:


#lista dla najlepszego wyniku
depth = 4
reg_list = [depth, mse1Train, mse1Test]

fileObject = open("mse_tree.pkl", 'wb')
pkl.dump(reg_list, fileObject)
fileObject.close()

print(reg_list)


# In[30]:


#max_depth = 5
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
tree_clf = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)
mse1Train = mean_squared_error(y_train, tree_clf.predict(X_train))
mse1Test = mean_squared_error(y_test, tree_clf.predict(X_test))
print(mse1Train)
print(mse1Test)


# In[31]:


plt.plot(X,y,'ro')
temp_X_train = np.sort(X_train, axis=0)
plt.plot(temp_X_train, tree_clf.predict(temp_X_train))


# In[32]:


#max_depth = 6
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
tree_clf = DecisionTreeRegressor(max_depth=6, random_state=42)
tree_clf.fit(X_train, y_train)
mse1Train = mean_squared_error(y_train, tree_clf.predict(X_train))
mse1Test = mean_squared_error(y_test, tree_clf.predict(X_test))
print(mse1Train)
print(mse1Test)


# In[33]:


plt.plot(X,y,'ro')
temp_X_train = np.sort(X_train, axis=0)
plt.plot(temp_X_train, tree_clf.predict(temp_X_train))


# In[34]:


#max_depth = 7
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
tree_clf = DecisionTreeRegressor(max_depth=7, random_state=42)
tree_clf.fit(X_train, y_train)
mse1Train = mean_squared_error(y_train, tree_clf.predict(X_train))
mse1Test = mean_squared_error(y_test, tree_clf.predict(X_test))
print(mse1Train)
print(mse1Test)


# In[35]:


plt.plot(X,y,'ro')
temp_X_train = np.sort(X_train, axis=0)
plt.plot(temp_X_train, tree_clf.predict(temp_X_train))


# In[36]:


#max_depth = 8
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
tree_clf = DecisionTreeRegressor(max_depth=8, random_state=42)
tree_clf.fit(X_train, y_train)
mse1Train = mean_squared_error(y_train, tree_clf.predict(X_train))
mse1Test = mean_squared_error(y_test, tree_clf.predict(X_test))
print(mse1Train)
print(mse1Test)


# In[37]:


plt.plot(X,y,'ro')
temp_X_train = np.sort(X_train, axis=0)
plt.plot(temp_X_train, tree_clf.predict(temp_X_train))


# In[ ]:




