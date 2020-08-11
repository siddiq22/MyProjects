#!/usr/bin/env python
# coding: utf-8

# # Task-2. Simple Linear Regression

# Importing relevant libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


# Importing data from the link

# In[2]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data.head()


# Dividing the data into input and output format 

# In[3]:


X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values  


# Plotting our data into a 2-D graph to visualise it manually.

# In[4]:


plt.scatter(X, y, color = 'red')
plt.title('Hours vs Score')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage')  
plt.show()


# Checking co-relation between the input and output of the data by using a heatmap

# In[5]:


sb.heatmap(data.corr() , annot = True)


# It is evident from the above graph that there is a strong positive linear relation (98 % ) between the number of hours studied and the 
# score.

# Splitting the dataset into the Training set and Test set
# 

# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Training the algorithm

# In[7]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


# Comparing the predicted Score with actual Score

# In[8]:


y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual Score': y_test, 'Predicted Score': y_pred})  
df 


# Visualising the Training set results

# In[9]:


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hours vs Score (Training set)')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()


# Visualising the Test set results

# In[10]:


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hours vs Score (Test set)')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()


# Testing new data

# In[11]:


no_of_hours = 9.25
predictor = regressor.predict(np.array([9.25]).reshape(1, 1))
print("No of Hours = {}".format(no_of_hours))
print("Predicted Score = {}".format(predictor[0]))
plt.show()


# Evaluating the model using mean square error.

# In[12]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

