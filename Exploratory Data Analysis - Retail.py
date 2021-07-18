#!/usr/bin/env python
# coding: utf-8

# # Name:- Rudra Pratap Singh 
# 
# # Task 5:- Exploratory Data Analysis - Retail
# 
# # Iot & Computer Vision Intern 
# 
# # The Sparks Foundation

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


data_set=pd.read_csv("SampleSuperstore.csv")
data_set


# In[59]:


data_set.shape


# In[60]:


data_set.info()


# In[61]:


data_set.describe()


# In[62]:


data_set.nunique()


# # checking for duplicate values

# In[63]:


data_set.duplicated().sum()


# In[65]:


#dropping duplicate values from the dataset
data_set.drop_duplicates()


# In[66]:


sns.heatmap(data_set.isnull(), cbar=False, yticklabels=False, cmap='viridis')


# # Analysis using Pairplot of each column

# In[67]:


sns.pairplot(data_set, hue = 'Category')


# In[12]:


#Based on Region
sns.pairplot(data_set, hue = 'Region')


# In[68]:


#Based on Segment
sns.pairplot(data_set, hue = 'Segment')


# In[16]:


#Based on Sub-Category
sns.pairplot(data_set, hue = 'Sub-Category')


# In[69]:


#Exploratory Data Analysis
plt.figure(figsize=(8,5))
sns.kdeplot(data_set['Sales'],color='red',label='Sales',shade=True,bw_adjust=20)
sns.kdeplot(data_set['Profit'],color='Blue',label='Profit',shade=True,bw_adjust=20)
plt.xlim([-100,1000])
plt.legend()


# In[70]:


#CATEGORIES VS REGION
plt.figure(figsize=(13,7))
plt.title('CATEGORIES VS REGION')
sns.countplot(x=data_set['Category'],hue=data_set['Region'],palette='rocket')
plt.xticks()


# In[71]:


#Category VS Sub-Category
plt.figure(figsize=(18,9))
plt.title('Category VS Sub-Category')
sns.countplot(x=data_set['Category'],hue=data_set['Sub-Category'],palette='rocket')
plt.xticks()


# In[72]:


#Sales vs Profit
plt.subplots(figsize=(10,6))
plt.scatter(data_set['Sales'],data_set['Profit'],color='green')
plt.xlabel('Sales')
plt.ylabel("Profit")
plt.title('Sales vs Profit',fontsize=20)
plt.show()


# In[73]:


# correlation matrix of the data
data_set.corr()


# In[74]:


# covariance matrix of data
data_set.cov()


# In[75]:


#Heatmap for Correlation
sns.heatmap(data_set.corr(),annot= True,cmap = 'rocket')
plt.figure(figsize=(14,7))


# In[76]:


sns.countplot(data_set['Category'])
plt.figure(figsize=(15,7))


# In[77]:


sns.countplot(data_set['Region'])
plt.figure(figsize=(12,6))


# In[30]:


fig, axs = plt.subplots(nrows = 2,ncols = 2,figsize = (12,6))
sns.countplot(data_set['Category'],ax = axs[0][0],palette='cubehelix_r')
sns.countplot(data_set['Region'],ax = axs[0][1],palette='gist_stern_r')
sns.countplot(data_set['Segment'],ax = axs[1][0],palette='gist_stern_r')
sns.countplot(data_set['Ship Mode'],ax = axs[1][1],palette='cubehelix_r')
axs[0][0].set_title('Category',fontsize=20)
axs[0][1].set_title('Region',fontsize=20)
axs[1][0].set_title('Segment',fontsize=20)
axs[1][1].set_title('Ship Mode',fontsize=20)
plt.tight_layout()


# In[78]:


plt.figure(figsize=(15,7))
sns.countplot(data_set['Sub-Category'], palette='rocket_r')
plt.tight_layout()


# In[79]:


data_set['Quantity'].value_counts()


# In[80]:


plt.figure(figsize=(11,7))
sns.countplot(data_set['Quantity'], palette='gist_stern_r')
plt.tight_layout()


# In[81]:


data_set['State'].value_counts()


# In[82]:


plt.figure(figsize=(20,8))
sns.countplot(data_set['State'], palette='viridis')
plt.xticks(rotation=270)
plt.tight_layout()


# In[83]:


data_set['Discount'].value_counts()


# In[84]:


plt.figure(figsize=(15,8))
sns.countplot(data_set['Discount'], color='Teal')
plt.xticks(rotation=45)
plt.tight_layout()


# In[85]:


fig, axs = plt.subplots(ncols=2, nrows = 2, figsize = (12,10))
sns.distplot(data_set['Sales'], color = 'green',  ax = axs[0][0])
sns.distplot(data_set['Profit'], color = 'orange',  ax = axs[0][1])
sns.distplot(data_set['Quantity'], color = 'red',  ax = axs[1][0])
sns.distplot(data_set['Discount'], color = 'blue',  ax = axs[1][1])
axs[0][0].set_title('Sales Distribution', fontsize = 20)
axs[0][1].set_title('Profit Distribution', fontsize = 20)
axs[1][0].set_title('Quantity distribution', fontsize = 20)
axs[1][1].set_title('Discount Distribution', fontsize = 20)
plt.show()


# In[86]:


df=data_set['State'].value_counts()
df.head(10)


# In[87]:


df_states=data_set.groupby(['State'])[['Sales','Discount','Profit']].mean()
df_states.head(10)


# In[88]:


df_state = data_set.groupby('State')['Quantity'].count().sort_values(ascending=True).plot.bar(figsize=(15,5))
plt.title('state-wise dealing',fontsize=20)
plt.xlabel('States')
plt.ylabel('Total no. of dealing')
plt.show()


# In[89]:


#Profit Analysis Statewise
df1=df_states.sort_values('Profit')
df1['Profit'].plot(kind= 'bar',figsize=(15,5),color='indigo')
plt.xlabel('States')
plt.ylabel('Profit Per State')
plt.title('Profit Analysis Statewise', fontsize=20)
plt.legend()


# In[90]:


df1['Sales'].plot(kind= 'pie',figsize=(22,22),autopct='%1.1f%%',startangle=90)
plt.title('Sales Analysis Statewise', fontsize=15)


# In[91]:


df_discount = data_set.groupby('State')['Discount'].mean().sort_values(ascending=False).plot.bar(figsize=(15,5),color='green')
plt.title('Discount Analyis Statewise',fontsize=20)
plt.xlabel('States')
plt.ylabel('Total Discount')
plt.show()


# In[92]:


segment=data_set.Segment.value_counts().reset_index()
segment.columns=("Segment","Count")
segment


# In[93]:


plt.pie(x="Count",labels="Segment",data=segment,radius=2,autopct="%.2f",pctdistance=0.4)


# In[94]:


#Top 5 cities by Sales,Profit,Discount
total_sales = data_set.groupby('City')['Sales'].sum()
top_5_cities = total_sales.sort_values(ascending = False).iloc[0:5]
top_5_cities.plot(kind = 'barh')
plt.title('Top 5 cities by sales',fontsize=15)
plt.xticks(rotation=270)
plt.xlabel('Sales')
plt.show()


# In[95]:


total_profit = data_set.groupby('City')['Profit'].sum()
top_5_cities = total_profit.sort_values(ascending= False).iloc[0:5]
top_5_cities.plot(kind = 'barh')
plt.title('Top 5 cities by Profit',fontsize=15)
plt.ylabel('Profit')


# In[96]:


total_profit = data_set.groupby('City')['Discount'].sum()
top_5_cities = total_profit.sort_values(ascending= False).iloc[0:5]
top_5_cities.plot(kind = 'bar')
plt.title('Top 5 cities by Discount',fontsize=15)
plt.xticks(rotation=0)
plt.ylabel('Discount')


# In[97]:


total_sales = data_set.groupby('Region')['Sales'].sum()
region_sales = total_sales.sort_values(ascending = False).iloc[0:4]
region_sales.plot(kind ='barh')
plt.title('Sales by Region',fontsize=15)
plt.xlabel('Sales')
plt.show()


# In[98]:


total_profit = data_set.groupby('Segment')['Sales'].sum()
segment_sales = total_profit.sort_values(ascending= False).iloc[0:4]
segment_sales.plot(kind = 'bar',fontsize=10)
plt.title('Sales by Segment')
plt.xticks(rotation=0)
plt.ylabel('Sales')


# # Thank you so much. Please give your valuable feedback in the comment box below. 

# # Thank you..............
