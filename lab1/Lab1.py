#!/usr/bin/env python
# coding: utf-8

# In[55]:


import os
import tarfile
import matplotlib
import urllib
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# In[22]:


HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
HOUSING_PATH = os.path.join("data")

if not os.path.isdir(HOUSING_PATH):
    os.makedirs(HOUSING_PATH)
tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")
urllib.request.urlretrieve(HOUSING_URL,tgz_path)
housing_tgz = tarfile.open(tgz_path)
housing_tgz.extractall(HOUSING_PATH)
housing_tgz.close()


# In[24]:


df = pd.read_csv('./data/housing.csv')
df


# In[27]:


df.head()


# In[29]:


df.info()


# In[31]:


df['ocean_proximity'].value_counts()


# In[32]:


df['ocean_proximity'].describe()


# In[41]:



df.hist(bins=50, figsize=(20,15))
plt.savefig('plot_1')


# In[42]:




df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.1, figsize=(7,4))
plt.savefig('plot_2')


# In[43]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.4, figsize=(7,3), colorbar=True,
        s=df["population"]/100, label="population", 
        c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('plot_3')


# In[51]:


df.corr()["median_house_value"].sort_values(ascending=False).to_csv('korelacja')


# In[56]:


sns.pairplot(df)


# In[57]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df,
                                       test_size=0.2,
                                       random_state=42)
len(train_set),len(test_set)


# In[ ]:




