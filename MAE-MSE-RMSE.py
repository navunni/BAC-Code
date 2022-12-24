#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import scipy.stats as stats

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("Downloads/2010Dataset.csv")
print(data.dtypes)
print(data.shape)


# In[31]:


data.head(100)


# In[4]:


# feature col = ['enre']
genres = list(set(data["General Genre"]))
# print(genres)
d = {}
genrePop = []
new_data = [genres, genrePop]
for genre in genres:
    d["%srank" % genre] = data.loc[data["General Genre"] == (genre)]
    print(genre)
    print((d["%srank" % genre]['pop'].sum())/len(d["%srank" % genre]['pop']))
    genrePop.append((d["%srank" % genre]['pop'].sum())/len(d["%srank" % genre]['pop']))
    
# print(d)
    
pop_df = pd.DataFrame(new_data)
pop_df = pop_df.T
pop_df
# x = data[feature_col]
# y = data.pop
# linreg = LinearRegression()
# linreg.fit(x, y)

feature_list = ['spch', 'acous', 'bpm']
x_mult = data[feature_list]
y_mult = data['pop']
mult_linreg = LinearRegression()
mult_linreg.fit(x_mult, y_mult)


# In[11]:


# find and print y-intercept
feature_col = ['Number']
x = data[feature_col]
y = data['pop']
linreg = LinearRegression()
linreg.fit(x, y)
print("y-intercept is", linreg.intercept_)

y_pred = linreg.predict(x)
print("R^2 is", metrics.r2_score(y, y_pred))
print("MSE is", metrics.mean_squared_error(y, y_pred))


# In[12]:


feature_list = ['spch', 'acous', 'bpm']
x_mult = data[feature_list]
y_mult = data['pop']
mult_linreg = LinearRegression()
mult_linreg.fit(x_mult, y_mult)
print("The y intercept is", mult_linreg.intercept_)


# In[13]:


# coefficients paired with metrics 
list(zip(feature_list, mult_linreg.coef_))


# In[14]:


# Calculation of Coefficient of Determination
y_mult_pred = mult_linreg.predict(x_mult)
print ("R^2 is:", metrics.r2_score(y_mult, y_mult_pred))
# Calculation of Mean Squared Error (MSE)
print ("MSE is:", metrics.mean_squared_error(y, y_mult_pred))


# In[20]:


# MAE/MSE/RMSE on Full Linear-Regression Model

feature_list = ['spch', 'acous', 'bpm', 'nrgy', 'dur', 'val', 'dB','dnce','spch','live']
x_mult = data[feature_list]
y_mult = data['pop']
mult_linreg = LinearRegression()
mult_linreg.fit(x_mult, y_mult)
x_mult_train, x_mult_test, y_mult_train, y_mult_test = train_test_split(x_mult, y_mult, test_size=0.9, random_state=1)
mult_linreg2 = LinearRegression()
mult_linreg2.fit(x_mult_train, y_mult_train)
y_mult_pred_train = mult_linreg2.predict(x_mult_train)
print("Training set MAE is calculated to be",(metrics.mean_absolute_error(y_mult_train, y_mult_pred_train)))
print("Training set MSE is calculated to be",(metrics.mean_squared_error(y_mult_train, y_mult_pred_train)))
print("Training set RMSE is calculated to be",np.sqrt(metrics.mean_squared_error(y_mult_train, y_mult_pred_train)))


# In[21]:


y_mult_pred_test = mult_linreg2.predict(x_mult_test)
print("Test set MAE is calculated to be",(metrics.mean_absolute_error(y_mult_test, y_mult_pred_test)))
print("Test set MSE is calculated to be",(metrics.mean_squared_error(y_mult_test, y_mult_pred_test)))
print("Test set RMSE is calculated to be",np.sqrt(metrics.mean_squared_error(y_mult_test, y_mult_pred_test)))


# In[23]:


data.plot(figsize=(18,5))


# In[24]:


data.isnull().values.any()


# In[25]:


data.hist()


# In[33]:


data.info()


# In[ ]:

