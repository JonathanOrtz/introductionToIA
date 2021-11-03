#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import require Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# Read the CSV file:

data = pd.read_csv(r'U:\introIA\introductionToIA\Fuel.csv')
data.head()


# In[3]:


# Let's select some features to explore more :
data = data[['ENGINESIZE','CO2EMISSIONS']]

# ENGINESIZE VS CO2EMISSIONS:
plt.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'],
color='blue')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()


# In[4]:


# Generating training and testing data from our data:
#We are using 80% data for training.

train = data[:(int((len(data)*0.8)))]
test = data[(int((len(data)*0.2))):]


# In[7]:


# Modeling: 
# using sklearn package to model data:
regr = linear_model.LinearRegression()
train_x = np.array(train[['ENGINESIZE']])
train_y = np.array(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
# The coefficients:
print('coefficient :', regr.coef_)#SLope
print('Intercept : ', regr.intercept_)#Intercept


# In[8]:


# Plotting the regression line:
plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'],
color='green')
plt.plot(train_x, regr.coef_*train_x + regr.intercept_, '-r')
plt.xlabel('Engine size')
plt.ylabel('Emission')


# In[ ]:


# Predicting values:
# Function for predicting future values:

def get_regresion_predictions(input_features, intercept, slope):
    predicted_values = input_features *slope + intercept
    return predicted_values

# Predicting emission for future car:
my_engine_size = 3.5
estimated_emission = get_regresion_predictions(my_engine_size, regr.intercept_[0],
                      regr.coef[0][0])

