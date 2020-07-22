#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


car_df = pd.read_csv('/Users/skylark/Desktop/DL and ML Practical Tutorials - Package/Project 1/Car_Purchasing_Data.csv', encoding='ISO-8859-1')


# In[10]:


car_df.head()


# In[11]:


sn.pairplot(car_df)


# In[12]:


car_df.isnull().sum()


# In[16]:


from sklearn.preprocessing import MinMaxScaler


# In[17]:


scaler = MinMaxScaler()
x = scaler.fit_transform(car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'],axis=1))
x


# In[18]:


scaler.data_max_


# In[19]:


scaler.data_min_


# In[21]:


y = scaler.fit_transform(car_df['Car Purchase Amount'].values.reshape(-1,1))
y


# In[22]:


scaler.data_max_


# In[23]:


scaler.data_min_


# In[13]:


from sklearn.model_selection import train_test_split


# In[44]:


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=100)


# In[45]:


train_x.shape


# In[46]:


from keras import models, layers, regularizers


# In[47]:


model = models.Sequential()


# In[48]:


model.add(layers.Dense(32, activation='relu', input_shape = (5,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1))
model.summary()


# In[49]:


model.compile(optimizer='rmsprop', loss='mse')


# In[50]:


history = model.fit(train_x, train_y, epochs=10, batch_size=30, validation_split=0.2)


# In[51]:


history.history.keys()


# In[52]:


val_loss = history.history['val_loss']
loss = history.history['loss']


# In[54]:


plt.plot(range(1, 11), val_loss, label = 'validation')
plt.plot(range(1, 11), loss, label = 'training')
plt.legend()
plt.show()


# In[55]:


model.evaluate(test_x, test_y)


# In[ ]:




