#!/usr/bin/env python
# coding: utf-8

# # Case Study 2
# There is 1 dataset(csv) with 3 years’ worth of customer orders. There are 4 columns in the csv dataset: index, CUSTOMER_EMAIL (unique identifier as hash), Net Revenue, and Year.
# 
# For each year we need the following information:
# - Total revenue for the current year
# - New Customer Revenue e.g., new customers not present in previous year only
# - Existing Customer Growth. To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year
# - Revenue lost from attrition
# - Existing Customer Revenue Current Year
# - Existing Customer Revenue Prior Year
# - Total Customers Current Year
# - Total Customers Previous Year
# - New Customers
# - Lost Customers
# 
# Additionally, generate a few unique plots highlighting some information from the dataset. Are there any interesting observations?
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

import warnings
warnings.filterwarnings('ignore')
from IPython.core.display import display, HTML
display(HTML("<style>div.output_scroll { height: 200em; }</style>"))

import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


# In[2]:


df = pd.read_csv('casestudy.csv', 
            sep = r',', 
            skipinitialspace = True, low_memory=False)


# In[3]:


df


# In[4]:


df['year'].unique()


# In[5]:


for i in df['year'].unique()[1:]:
    print("\n\nCurrent Year: ", i)
    
    print('-'*50)
    
    print("\nNet Revenue in current year\n")
    print(df.loc[df['year'] == i].groupby(['year'])[['net_revenue']].sum())
    
    print('-'*50)
    
    print("\nNew Customers Revenue\n")
    prev = df.loc[df['year'] == i-1]['customer_email'].values
    curr = df.loc[df['year'] == i]['customer_email'].values
    common = np.intersect1d(prev, curr)
    print(df.loc[(df['year']==i) & (~df['customer_email'].isin(common))]['net_revenue'].sum())
    
    print('-'*50)
    
    
    print("\nCustomer Growth\n")
    prev_val = df.loc[(df['year'] == i-1) & (df['customer_email'].isin(common))]['net_revenue'].values
    curr_val = df.loc[(df['customer_email'].isin(common)) & (df['year']==i)]['net_revenue'].values
    print(np.subtract(curr_val,prev_val))
    
    
    print('-'*50)
    
    
    print("\nExisting Customer Revenue Current Year\n")
    print(df.loc[(df['year']==i) & (df['customer_email'].isin(common))]['net_revenue'].sum())
    
    print('-'*50)
    
    
    print("\nExisting Customer Revenue previous Year\n")
    print(df.loc[(df['year']==i-1) & (df['customer_email'].isin(common))]['net_revenue'].sum())
    
    
    print('-'*50)
    
    
    print("\nTotal customers in current year\n")
    print(df.loc[df['year'] == i].groupby(['year'])[['customer_email']].count())
    
    
    print('-'*50)
    
    
    print("\nTotal customers in previous year\n")
    print(df.loc[df['year'] == i-1].groupby(['year'])[['customer_email']].count())
    
    
    print('-'*50)
    
    
    print("\nNew Customers\n")
    print(df.loc[(df['year']==i) & (~df['customer_email'].isin(common))]['customer_email'])
    
    print('-'*50)
    
    
    print("\nLost Customers\n")
    print(df.loc[(df['year']==i-1) & (~df['customer_email'].isin(curr))]['customer_email'])
    
    print('-'*50,"\n\n")
    
    print('*'*100)
    


# ## Some Observations:
# From the below line graph we can infer that the no. of customers fell significantly for the year 2016. This sharp drop of customers, however did not result in a significant drop in the net revenue of the company. The company managed to gain an all time high number of customers for the year 2017 which significantly increased the net revenue.

# In[6]:


x = df['year'].unique()
y = df.groupby("year")['customer_email'].count()
fig = go.Figure(px.line(df, x = x,
              y = y, width=500,
             labels={
                     "x": "Year",
                     "y": "No. of Customers"
                 },
                title="No. of customers"))
fig.update_traces(mode='markers+lines')
fig.update_xaxes()
fig.update_layout(
    xaxis = dict(
        tickangle=45,
                 tickmode = 'array',
                 tickvals = x
    )
)
fig.show()


# In[7]:


x = df['year'].unique()
y = df.groupby('year')['net_revenue'].sum()

layout = go.Layout()
fig1 = go.Figure(px.bar(x = x, y = y, width=500, labels={
                     "x": "Year",
                     "y": "Net Revenue"
                 },
                title="Net Revenue"
                       ))
fig1.update_xaxes(tickangle=45,
                 tickmode = 'array',
                 tickvals = x)
iplot(fig1)

