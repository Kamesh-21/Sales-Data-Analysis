#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading Data

# In[90]:


df = pd.read_csv("D:\\joti\\projects\\codealpha_tasks\\sales_prediction\\dataset\\Advertising.csv")


# # Data Cleaning

# In[91]:


df.head()


# In[92]:


df.info()


# In[93]:


df.columns


# In[94]:


df.describe()


# In[95]:


# drop extra columns
df.drop(columns = ["Unnamed: 0"], inplace = True)


# In[96]:


df.columns


# In[97]:


# checking missing values
df.isnull().sum()


# In[98]:


# checking duplicates
df.duplicated().sum()
df.drop_duplicates(inplace=True)


# In[99]:


# checking outliers
sns.boxplot(data=df)
plt.show()


# In[100]:


# remove outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[101]:


sns.boxplot(data=df)
plt.show()


# In[102]:


# save cleaned dataset
df.to_csv("Cleaned_Sales_Prediction.csv", index = False)


# # Exploratory Data Analysis

# In[103]:


plt.figure(figsize=(6,4))
sns.histplot(df["Sales"], bins=20, kde=True)
plt.title("Distribution of Sales",fontweight = "bold")
plt.show()


# In[104]:


plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap",fontweight = "bold")
plt.show()


# In[105]:


sns.pairplot(df)
plt.show()


# In[106]:


df["Total_Ad"] = df["TV"] + df["Radio"] + df["Newspaper"]

plt.figure(figsize=(6,4))
sns.regplot(x=df["Total_Ad"], y=df["Sales"])
plt.title("Total Advertising Budget vs Sales",fontweight = "bold")
plt.show()


# In[ ]:




