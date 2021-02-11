#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Regression Tree with Gradient Boosting
#import Chefboost as Chef , This is used to run DT algorithms ID3, C4.5, CART, CHAID, Regression along with GradientBoosting, RandomForest, AdaBoost, FeatureImportance
#! pip install Chefboost
import pandas as pd

df = pd.read_csv('golf4.txt')


# In[20]:


import Chefboost as chef


# In[18]:


df.head()


# In[8]:


num_of_instances = df.shape[0]
num_of_instances


# In[9]:


#Since the target variable is interval we use Regression Tree
config = {'algorithm' : 'Regression'}


# In[21]:


chef.fit(df.copy(), config)


# In[22]:


import imp


# In[28]:


moduleName = "outputs/rules/rules"
fp, pathname, description = imp.find_module(moduleName)
myrules = imp.load_module(moduleName, fp, pathname, description)

#to pass a datapoint to the DT use findDecision()
myrules.findDecision(['Sunny',85,85,'Weak'])


# In[29]:


#To compare the prediction with the training data 
mae = 0

for index, instance in df.iterrows():
    actual = instance['Decision']
    prediction = myrules.findDecision(instance.values)
    error = abs(actual - prediction)
    mae = mae + error
    print("Actual : ", actual, " Prediciton : ", prediction, " Error : ", error)

mae = mae / num_of_instances
print(" MAE - ", mae)
    


# In[30]:


#Now we have a Regression Tree with MAE - 3.75 , we can get better performance by Gradient Boosting 

config = {'enableGBM' :  True, 'epochs' : 7, 'learning_rate' : 1}


# In[31]:


chef.fit(df.copy(),config)


# In[32]:


mae = 0

for index, instance in df.iterrows():
    actual = instance['Decision']
    prediction = 0
    
    for j in range(0,7):
            moduleName = "outputs/rules/rules%s" % (j)
            fp, pathname, description = imp.find_module(moduleName)
            myrules = imp.load_module(moduleName, fp, pathname, description)
            prediction = prediction + myrules.findDecision(instance.values)
    
    error = abs(actual - prediction)
    mae = mae + error
    print("Actual : ", actual, " Prediciton : ", prediction, " Error : ", error)

mae = mae / num_of_instances
print(" MAE - ", mae)


# In[ ]:


# MAE is boosted from 3.75 to 0.72

