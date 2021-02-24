#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("Desktop/zomato.csv")


# In[3]:


df.head()


# In[4]:


df.drop(['address','url','phone'],axis = 1, inplace = True)


# In[5]:


df.head()


# In[6]:


df.isnull().sum()*100/51717


# In[7]:


df.drop(['dish_liked','listed_in(city)'],axis = 1, inplace = True)


# In[8]:


df.head()


# In[9]:


df.fillna(df.mode())


# In[10]:


df['rate'].value_counts()


# In[11]:


df['rate'].replace({"NEW":"2.5/5"},inplace=True)
df['rate'].replace(np.NaN,'2.5/5')


# In[12]:


df['rate'].unique()


# In[13]:


a = df.loc[df['rate'] == '-'].index
df.drop(a, axis = 0, inplace = True)


# In[14]:


df['rate'].unique()


# In[15]:


df


#  No. of restaurants with and without online order

# In[16]:


l=df.online_order.value_counts()[:30].plot(kind='bar')
plt.xlabel('online order')
plt.ylabel('No. of restaurants ')


# No. of restaurants at a particular location

# In[17]:


l=df.location.value_counts()[:30].plot(kind='bar')
plt.xlabel('Location')
plt.ylabel('No. of restaurants at that location ')


# highest cost of a restaurant

# In[18]:


costs=df['approx_cost(for two people)']
costlist=[]
for i in costs:
    if(str(i)=='nan'):
        costlist.append(int(str(i).replace('nan','0')))
    else:
        costlist.append(int(str(i).replace(',','')))
df['Costs']=costlist


# In[19]:


df['Costs']


# In[20]:


df


# In[21]:


max=0
count=0
list=[]
num=0
for i in df['Costs']:
    if(max<i):
        max=i
    if(max==6000):
        num=count+1
        list.append(num)
    count=count+1

print("The highest cost of a restaurant is:"+str(max))
print("There are "+str(len(list))+" number of restaurants with max cost")


# In[22]:


df.drop(['name','location','cuisines','reviews_list','menu_item','listed_in(type)'], axis = 1, inplace = True)


# In[23]:


df


# In[24]:


df.dropna()


# no. of restaurants with casual dining

# In[25]:


types=df['rest_type']


# In[26]:


count=0
for i in types:
    if(i=='Casual Dining'):
        count=count+1
print("No. of restaurants with casual dining are "+str(count))


# In[27]:


noOfvotes=df['votes']
voteslist=[]
for i in noOfvotes:
    if(str(i)=='nan'):
        voteslist.append(int(str(i).replace('nan','0')))
    else:
        voteslist.append(int(str(i).replace(',','')))
df['Votes']=voteslist


# In[28]:


df.drop(['votes','approx_cost(for two people)'], axis = 1, inplace = True)


# In[29]:


df


# In[30]:


rating=df['rate']
ratinglist=[]
for i in rating:
    if(str(i)=='NaN'):
        ratinglist.append(float(str(i).replace('Nan','0')))
    else:
        ratinglist.append(float(str(i).replace('/5','')))
df['Rating']=ratinglist


# In[31]:


df.drop(['rate','rest_type'],axis = 1, inplace = True)


# In[32]:


df


# Checking the dependency of online_order,book_table,Costs,Votes on Rating

# In[33]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values.ravel()


# In[34]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
y=y.reshape(-1,1)
imputer.fit(y)
y=imputer.transform(y)


# In[35]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0,1])],remainder='passthrough')
x=np.array(ct.fit_transform(x))


# In[36]:


x


# Multiple Linear Regression

# In[37]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[43]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[46]:


y_pred=regressor.predict(x_test)


# Accuracy Test

# In[59]:


regressor.score(x,y)


# Decision Tree Regression

# In[64]:


from sklearn.tree import DecisionTreeRegressor
Dregressor=DecisionTreeRegressor(random_state=0)
Dregressor.fit(x_train,y_train)


# In[65]:


y_pred_DTR=Dregressor.predict(x_test)


# In[66]:


Dregressor.score(x,y)


# In[ ]:





# In[ ]:
