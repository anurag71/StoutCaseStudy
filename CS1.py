#!/usr/bin/env python
# coding: utf-8

# # Case Study 1
# Below is a data set that represents thousands of loans made through the Lending Club platform, which is a platform that allows individuals to lend to other individuals.
# Perform the following using the language of your choice:
# 1. Describe the dataset and any issues with it.
# 2. Generate a minimum of 5 unique visualizations using the data and write a brief description of your observations. Additionally, all attempts should be made to make the visualizations visually appealing.
# 3. Create a feature set and create a model which predicts interest rate using at least 2 algorithms. Describe any data cleansing that must be performed and analysis when examining the data.
# 4. Visualize the test results and propose enhancements to the model, what would you do if you had more time. Also describe assumptions you made and your approach.
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

import plotly
import plotly.express as px

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing, ensemble
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('loans_full_schema.csv', 
            sep = r',', 
            skipinitialspace = True, low_memory=False)


# The below graph provides an insight on the average interest rate based on the whether the applicant owns a house, is paying mortgage or has rented a place to stay. The surprising fact that we can see from the graph is that avg. interest rate was higher for people who owned a house when compared to the other two categories. This could be thought of a valid scenario because people paying rent on time gives assurane to the lender.

# In[3]:


x = df['homeownership'].unique()
y = df.groupby('homeownership')['interest_rate'].mean()

layout = go.Layout()
fig1 = go.Figure(px.bar(x = x, y = y, width=500, labels={
                     "x": "Home Ownership",
                     "y": "Interest Rate"
                 },
                title="Home Ownership~Interest Rate"
                       ))
fig1.update_xaxes(tickangle=45,
                 tickmode = 'array',
                 tickvals = x)
iplot(fig1)


# The pie chart shown below gives a distribution of the applicants based on whether the source of income was verfiied or not. Out of all the applicants, the lender couldn't verify the source of the applicant's income and thus those applicants had a significantly higher interest rate.

# In[4]:


labels = df['verified_income'].unique()

data = df.groupby(['verified_income'])['verified_income'].count()
sum = data.sum()
final_data = data.groupby(level=0).apply(lambda x:
                                                 100 * x / sum)
trace = go.Pie(labels = labels, values = final_data)
final_data = [trace]
fig = go.Figure(data = final_data)
iplot(fig)


# Below pie chart shows a distribution of the applicants according to the loan grade they were assigned. It also contains the average of the annual income (in M) for that particular grade as well the interest rate.

# In[5]:


labels = df['grade'].unique()
data = df.groupby(['grade', 'verified_income'])['interest_rate'].mean()
sum = data.sum()
final_data = data.groupby(level=0).apply(lambda x:
                                                 100 * x / sum)
trace = go.Pie(labels = labels, values = final_data)
final_data = [trace]
fig = go.Figure(data = final_data)
iplot(fig)


# The below graph provides information about the change of interest rate according to grade and also the annual income for that grade.

# In[6]:


x = df['grade'].unique()
y = df.groupby('grade')['annual_income'].mean()
z = df.groupby('grade')['interest_rate'].mean()
fig=make_subplots(
        specs=[[{"secondary_y": True}]])   

fig.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                  yaxis_domain=[0, 0.94]);

fig.add_trace(
    go.Bar(x=x,
           y=y,
           name="Avg. Annual Income",
          ), secondary_y=False)
fig.add_trace(
    go.Scatter(
               y=z,
               name="Avg. Interest Rate",
               line_color="#ee0000"), secondary_y=True)
fig.data[1].update(xaxis='x2')
fig.update_layout(width=700, height=475)


# The below graph is quite fascinating because it depicts a relationship between the annual income, interest rate and the no. of times a applicant declared bankruptcy. It is quite astonishing to see that the people who had a record of being bankrupt twice had the highest avg annual income and a marginal interest rate. The people with no record of bankruptcy had the lowest annual income but were charged a signnificantly lower interest rate.

# In[7]:


x = df['public_record_bankrupt'].unique()
y = df.groupby('public_record_bankrupt')['annual_income'].mean()
z = df.groupby('public_record_bankrupt')['interest_rate'].mean()
fig=make_subplots(
        specs=[[{"secondary_y": True}]])   

fig.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                  yaxis_domain=[0, 0.94]);

fig.add_trace(
    go.Bar(x=x,
           y=y,
           name="Avg. Annual Income",
          ), secondary_y=False)
fig.add_trace(
    go.Scatter(
               y=z,
               name="Avg. Interest Rate",
               line_color="#ee0000"), secondary_y=True)
fig.data[1].update(xaxis='x2')
fig.update_layout(width=700, height=475)


# In[8]:


df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)


# In[9]:


df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)


# In[10]:


pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', None)


# In[11]:


def null_per():
    for col in df.columns:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing*100)))

def unique():
    for col in df:
      print(col, df[col].unique())


# ## Database Description and Issues
# The dataset contains various details for people who had applied and had their loan sanctioned. It contains attributes like net income, joint income, the type of loan application, interest paid, etc. There are a lot of missing values for the columns annual_income_joint, verification_income_joint and debt_to_income_joint. Since there are values, trying to fill them would cause a lot of disturbance in the model training. For the sake of this problem statement, we will drop these columns. The columns debt_to_income, months_since_last_credit_inquiry and num_accounts_120d_past_due are dropped as well, even though they are intuitively important when deciding the interest_rate, they would need a little bit pre processing to make them fit in range for the ML model and filling out the missing values as well. Other columns that are dropped are the ones that aren't significant factors when it comes to deciding the interest rate.

# In[12]:


df.drop('emp_title', axis=1, inplace=True)
df.drop('emp_length', axis=1, inplace=True)
df.drop('state', axis=1, inplace=True)
df.drop('disbursement_method', axis=1, inplace=True)
df.drop('issue_month', axis=1, inplace=True)
df.drop('verification_income_joint', axis=1, inplace=True)
df.drop('debt_to_income_joint', axis=1, inplace=True)
df.drop('annual_income_joint', axis=1, inplace=True)
df.drop('months_since_90d_late', axis=1, inplace=True)
df.drop('months_since_last_delinq', axis=1, inplace=True)
df.drop('loan_status', axis=1, inplace=True)
df.drop('balance', axis=1, inplace=True)
df.drop('debt_to_income', axis=1, inplace=True)
df.dropna(subset=['months_since_last_credit_inquiry'], inplace=True)
df.dropna(subset=['num_accounts_120d_past_due'], inplace=True)
df.drop('application_type', axis=1, inplace=True)
df.drop('loan_purpose', axis=1, inplace=True)


# In[13]:


null_per()


# Since, the above listed columns don't have any missing values and are also important for the interest rate calculation, that is the **final feature set** that I've chosen.

# Some of the features are categorical and we will need to convert them to numeric values so that the model can take them into account.

# In[14]:


le = preprocessing.LabelEncoder()
df['homeownership'] = le.fit_transform(df['homeownership'].values)
df['verified_income'] = le.fit_transform(df['verified_income'].values)
df['grade'] = le.fit_transform(df['grade'].values)
df['sub_grade'] = le.fit_transform(df['sub_grade'].values)
df['initial_listing_status'] = le.fit_transform(df['initial_listing_status'].values)


# In[15]:


train, test = train_test_split(df, test_size=0.2)

train_target = train['interest_rate']
test_target = test['interest_rate']

train_predictors = train.drop(['interest_rate'], axis=1)
test_predictors = test.drop(['interest_rate'], axis=1)


# In[16]:


train.info()


# We will be using Linear Regression for predicting the interest rate. The root mean square score for the model depicts how well the model fit the data. The actual truth values as well the predicted values are listed below.

# In[17]:


model = LinearRegression()
model.fit(train_predictors, train_target)
prediction = model.predict(test_predictors)

list(zip(train_predictors, model.coef_))
LRscore = r2_score(test_target, prediction)

print("R2 Score: ",LRscore)

data = {'Actual Values': test_target.values, 'Predicted Values': prediction}
df_final = pd.DataFrame(data=data)
df_final


# The below graph provides a visual description about how close the actual and predicted values are for the linear regression model.

# In[18]:


ax1 = sns.distplot(test_target, hist=False, color="r", label="Actual Value")
sns.distplot(prediction, hist=False, color="b", label="Predicted Values" , ax=ax1)


# We will be using Gradient Boosting Regressor for predicting the interest rate. The root mean square score for the model depicts how well the model fit the data. The actual truth values as well the predicted values are listed below.

# In[19]:


params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(train_predictors, train_target)

predictions = reg.predict(test_predictors)

GBRscore = r2_score(test_target, predictions)
print("R2 Score: ",GBRscore)

data = {'Actual Values': test_target.values, 'Predicted Values': predictions}
df_final = pd.DataFrame(data=data)
df_final


# The below graph provides a visual description about how close the actual and predicted values are for the Gradient Boosting regression model.

# In[20]:


ax1 = sns.distplot(test_target, hist=False, color="r", label="Actual Value")
sns.distplot(predictions, hist=False, color="b", label="Predicted Values" , ax=ax1)


# If I had more time, I would have had deep dived into the dataset. I would have handled intuitively important feature coloumns for missing values in a much better way. This would have had created a better feature set and would have yielded better results. Another change I would have made was to fine tune the parameters for the regression models rather than using the default values.
