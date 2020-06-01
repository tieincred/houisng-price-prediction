#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import numpy as nm


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

housing.target


housing = pd.read_csv('C:\\Users\91820\\Downloads\\CaliforniaHousing\\calf_housing.csv')


# In[2]:


housing.tail()


# In[3]:


housing.head()


# In[4]:


housing.describe()


# In[5]:


housing.hist(figsize=(30,30),bins=30)


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


train , test = train_test_split(housing , test_size =0.2)


# In[8]:


train.tail()


# In[9]:


test['median_income'].hist()


# In[10]:


train.size


# In[11]:


housing['median_income'].hist()


# In[12]:


housing['income_cat']= nm.ceil(housing['median_income']/1.5)


# In[13]:


housing['income_cat'].head()


# In[14]:


housing['income_cat'].hist()


# In[15]:


housing['income_cat'].tail()


# In[16]:


housing['median_income'].head()


# In[17]:


housing


# In[18]:


housing['income_cat'].hist()


# In[19]:


housing['income_cat'].where(housing['income_cat']<5, 5.0, inplace=True)     

housing['income_cat']

housing['income_cat'].hist()


# In[20]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=9, test_size=0.2)

print(split)

for train_index, test_index in split.split(housing,housing['income_cat']):
    s_train_set, s_test_set = housing.loc[train_index], housing.loc[test_index]        


# In[21]:


s_train_set['income_cat'].hist()


# In[22]:


housing= s_train_set.copy()


# In[23]:


housing.plot(kind='scatter', x='longitude',y='latitude')


# In[24]:


housing.plot(kind='scatter', x='longitude',y='latitude', alpha=0.3, s=housing['population']/100, c='median_house _value', figsize=(15,12), cmap=plt.get_cmap('coolwarm'))


# In[25]:


housing.corr()


# In[26]:


from pandas.plotting import scatter_matrix
scatter_matrix(housing[['housing_median_age', 'total_rooms', 'median_income', 'median_house _value']], figsize=(12,12))


# In[27]:


housing= s_train_set.copy()


# In[28]:


housing


# In[29]:


housing.drop('median_house _value', axis=1, inplace=True)


# In[30]:


housing


# In[31]:


housing_labels=s_train_set['median_house _value'].copy()


# In[32]:


housing_labels


# In[33]:


housing.tail()


# In[34]:


incomplete_data = housing[housing.isnull().any(axis=1)]


# In[35]:


incomplete_data['total_bedrooms'].fillna(housing['total_bedrooms'].median(), inplace=True)
#this basically changes all non available values into median of that column.


# In[36]:


#same thing in scikit
#( from sklearn.preprocessing import Imputer) this is not working basically maybe a bug!!
from sklearn.impute import SimpleImputer


# In[37]:


get_ipython().run_line_magic('pinfo', 'SimpleImputer')


# In[38]:


imputer = SimpleImputer( strategy='median')


# In[39]:


#numerical_housing = housing.drop('ocean_proximity', axis=1)
# if we had a row which does not have numerical value imouter not gonna apply there so drop to make all numerical.


# In[40]:


numerical_housing = housing


# In[41]:


numerical_housing


# In[42]:


imputer.fit(numerical_housing) #calculation of medians


# In[43]:


imputer.statistics_


# In[44]:


numerical_housing.median().values #filling NaN's with medians


# In[45]:


X = imputer.transform(numerical_housing) #filling NaN's with medians


# In[46]:


transformed_housing = pd.DataFrame(X, columns = numerical_housing.columns, index=list(housing.index.values))


# In[47]:


transformed_housing


# In[48]:


t_data =transformed_housing[transformed_housing.income_cat>=4]


# In[49]:


t_data['income_cat'].where(t_data['income_cat']<4, 4.0, inplace=True) #leave all those who satisfy the cond. and change rest as 
#given


# In[50]:


t_data


# In[51]:


transformed_housing.loc[t_data.index.values]


# In[52]:


import random 
a = ["<1H OCEAN", "NEAR OCEAN", "INLAND", "NEAR BAY", "ISLAND"]
PROX = []
for i in transformed_housing.index:
    k = random.choice(a)
    PROX.append(k)
m = nm.asarray(PROX)


len(m)

transformed_housing["ocean_proximity"] = m


# In[53]:


transformed_housing


# In[54]:


s_test_set


# In[55]:


import random 
a = ["<1H OCEAN", "NEAR OCEAN", "INLAND", "NEAR BAY", "ISLAND"]
PROXI = []
for i in s_test_set.index:
    k = random.choice(a)
    PROXI.append(k)
l = nm.asarray(PROXI)


# In[56]:


len(s_test_set.index)


# In[57]:


s_test_set["ocean_proximity"] = l


# In[58]:


s_test_set


# In[59]:


housing = transformed_housing


# In[60]:


housing


# In[61]:


housing_cat = housing[["ocean_proximity"]]


# In[62]:


from sklearn.preprocessing import LabelEncoder


# In[63]:


le = LabelEncoder()


# In[64]:


le_encode_housing = le.fit_transform(housing_cat)


# In[65]:


le.classes_


# In[66]:


from sklearn.preprocessing import OrdinalEncoder


# In[67]:


ordinal_Encoder = OrdinalEncoder()


# In[68]:


ord_encode_cat = ordinal_Encoder.fit_transform(housing_cat)


# In[69]:


ord_encode_cat


# In[70]:


from sklearn.preprocessing import OneHotEncoder


# In[71]:


One_hotencoded = OneHotEncoder()


# In[72]:


hot_encoded = One_hotencoded.fit_transform(housing_cat)


# In[73]:


hot_encoded


# In[74]:


hot_encoded.toarray()


# In[87]:


housing


# In[88]:


dummies = pd.get_dummies(housing['ocean_proximity'])


# In[89]:


dummies.head()


# In[90]:


housing = housing.merge(dummies,left_index=True,right_index=True)


# In[95]:


housing.drop('ocean_proximity',axis =1,inplace = True)


# In[96]:


housing.drop('<1H OCEAN',axis =1,inplace = True)


# In[97]:


housing


# In[98]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[99]:


X = housing
y = housing_labels


# In[100]:


model.fit(X,y)


# In[101]:


model.score(X,y)


# In[ ]:




