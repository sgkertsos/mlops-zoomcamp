#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import pickle
import pandas as pd


# In[4]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[5]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[6]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[7]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# ### Question 1

# In[10]:


import numpy as np


# In[11]:


std_dev = np.std(y_pred)


# In[12]:


std_dev


# ### Question 2

# In[13]:


df['ride_id'] = f'{2023:04d}/{3:02d}_' + df.index.astype('str')


# In[15]:


df_result = pd.DataFrame()


# In[18]:


df_result['ride_id']=df['ride_id']
df_result['y_pred']=y_pred


# In[19]:


df_result


# In[20]:


df_result.to_parquet(
    'output_file.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)


# In[ ]:




